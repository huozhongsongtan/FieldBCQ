# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "Detect",
    "Segment",
    "MPTSSegment",
    "BTQSegment",
    "MQCSegment",
    "BASMQCSegment",
    "BASOnlyMQCSegment",
    "BoundaryDiceMQCSegment",
    "BASMQCSGSegment",
    "BASMQCSGPSMFSegment",
    "BASPSMFSegment",
    "BASBTQSegment",
    "BASBTQDCSCSegment",
    "BASBTQFMSCSegment",
    "BASBTQCFASegment",
    "BASBTQPSMFSegment",
    "BASBTQSSCMoESegment",
    "HARMSegment",
    "Pose",
    "Classify",
    "OBB",
    "RTDETRDecoder",
    "v10Detect",
)


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        # x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x[i]), self.one2one_cv3[i](x[i])), 1) for i in range(self.nl)
        ]
        if hasattr(self, 'cv2') and hasattr(self, 'cv3'):
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class MPTSTopologyHead(nn.Module):
    """Predict parcel boundary and interior-distance maps from mask prototypes."""

    def __init__(self, channels=32):
        super().__init__()
        hidden = max(16, channels // 2)
        self.cv1 = Conv(channels, channels, 3)
        self.cv2 = Conv(channels, hidden, 3)
        self.out = nn.Conv2d(hidden, 2, 1)

    def forward(self, prototypes):
        """Return boundary and distance logits at prototype resolution."""
        return self.out(self.cv2(self.cv1(prototypes)))


class MPTSSegment(Segment):
    """YOLO segment head with a training-only parcel topology branch."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.topology = MPTSTopologyHead(self.nm)

    def forward(self, x):
        """Return standard segment outputs plus topology logits during training."""
        p = self.proto(x[0])
        topology = self.topology(p) if self.training else None
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p, topology
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class BoundaryPointRefiner(nn.Module):
    """Refine the shared prototype bank with a zero-initialized boundary residual."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(16, channels)
        self.features = nn.Sequential(Conv(channels, hidden, 3), Conv(hidden, hidden, 3))
        self.out = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, prototypes):
        """Add a learnable residual while preserving pretrained prototypes at initialization."""
        return prototypes + self.out(self.features(prototypes))


class BTQSegment(Segment):
    """Boundary-refined segment head with mask-IoU and boundary-F1 quality prediction."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.point_refiner = BoundaryPointRefiner(self.nm)
        quality_channels = max(16, ch[0] // 4)
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(x, quality_channels, 3),
                Conv(quality_channels, quality_channels, 3),
                nn.Conv2d(quality_channels, 2, 1),
            )
            for x in ch
        )
        # Start close to neutral score calibration when loading a standard YOLO11-seg checkpoint.
        for branch in self.cv5:
            nn.init.zeros_(branch[-1].weight)
            nn.init.constant_(branch[-1].bias, 4.0)

    def forward(self, x):
        """Return standard segmentation outputs and training-only quality logits."""
        p = self.point_refiner(self.proto(x[0]))
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        quality_logits = torch.cat([self.cv5[i](x[i]).view(bs, 2, -1) for i in range(self.nl)], 2)
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p, quality_logits

        # Geometric fusion penalizes a detection only when either mask or boundary quality is poor.
        quality = quality_logits.sigmoid().prod(1, keepdim=True).sqrt()
        decoded = x if self.export else x[0]
        calibrated = decoded.clone()
        calibrated[:, 4 : 4 + self.nc] *= quality
        output = torch.cat([calibrated, mc], 1)
        return (output, p) if self.export else (output, (x[1], mc, p))


class MQCSegment(Segment):
    """Stage C head with mask-quality calibration and no boundary auxiliary branch."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        quality_channels = max(16, ch[0] // 4)
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(x, quality_channels, 3),
                Conv(quality_channels, quality_channels, 3),
                nn.Conv2d(quality_channels, 2, 1),
            )
            for x in ch
        )
        for branch in self.cv5:
            nn.init.zeros_(branch[-1].weight)
            nn.init.constant_(branch[-1].bias, 4.0)

    def forward(self, x):
        """Return standard segmentation outputs and MQC logits while training."""
        p = self.proto(x[0])
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        quality_logits = torch.cat([self.cv5[i](x[i]).view(bs, 2, -1) for i in range(self.nl)], 2)
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p, quality_logits

        quality = quality_logits.sigmoid().prod(1, keepdim=True).sqrt()
        quality_blend = float(getattr(self, "quality_blend", 1.0))
        quality_factor = (1.0 - quality_blend) + quality_blend * quality
        decoded = x if self.export else x[0]
        calibrated = decoded.clone()
        calibrated[:, 4 : 4 + self.nc] *= quality_factor
        output = torch.cat([calibrated, mc], 1)
        return (output, p) if self.export else (output, (x[1], mc, p))


class BASMQCSegment(Segment):
    """BAS boundary supervision with mask-quality calibration, without BPR."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.boundary_head = nn.Sequential(Conv(self.nm, self.nm, 3), nn.Conv2d(self.nm, 1, 1))
        quality_channels = max(16, ch[0] // 4)
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(x, quality_channels, 3),
                Conv(quality_channels, quality_channels, 3),
                nn.Conv2d(quality_channels, 2, 1),
            )
            for x in ch
        )
        for branch in self.cv5:
            nn.init.zeros_(branch[-1].weight)
            nn.init.constant_(branch[-1].bias, 4.0)

    def _quality_features(self, x):
        """Return the feature maps consumed by the MQC branch."""
        return x

    def forward(self, x):
        """Return BAS boundary logits and MQC quality logits while training."""
        p = self.proto(x[0])
        boundary_logits = self.boundary_head(p) if self.training else None
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        quality_features = self._quality_features(x)
        quality_logits = torch.cat(
            [self.cv5[i](quality_features[i]).view(bs, 2, -1) for i in range(self.nl)], 2
        )
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p, (boundary_logits, quality_logits)

        quality = quality_logits.sigmoid().prod(1, keepdim=True).sqrt()
        # Old checkpoints do not contain this runtime-only attribute, so keep full MQC as the fallback.
        quality_blend = float(getattr(self, "quality_blend", 1.0))
        quality_factor = (1.0 - quality_blend) + quality_blend * quality
        decoded = x if self.export else x[0]
        calibrated = decoded.clone()
        calibrated[:, 4 : 4 + self.nc] *= quality_factor
        output = torch.cat([calibrated, mc], 1)
        return (output, p) if self.export else (output, (x[1], mc, p))


class BASOnlyMQCSegment(BASMQCSegment):
    """Stage B+C head: BAS BCE supervision combined with MQC."""
    pass


class BoundaryDiceMQCSegment(BASMQCSegment):
    """Stage A+C head: Boundary Dice supervision combined with MQC."""
    pass


class BASMQCSGSegment(BASMQCSegment):
    """BAS-MQC with training gradients isolated from the shared feature extractor."""

    def _quality_features(self, x):
        """Stop MQC loss gradients at shared features while keeping the quality head trainable."""
        return [feature.detach() for feature in x] if self.training else x


class PhysicalScaleMetadataFiLM(nn.Module):
    """Condition multi-scale image features on physical scale and sensor metadata."""

    def __init__(self, channels, metadata_dim=11, hidden=32, strength=0.1):
        super().__init__()
        self.metadata_dim = metadata_dim
        self.strength = strength
        self.encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.affine = nn.ModuleList(nn.Linear(hidden, 2 * c) for c in channels)
        for layer in self.affine:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, metadata=None):
        """Apply bounded feature-wise scale and bias with a learned missing-metadata token."""
        if metadata is None:
            metadata = features[0].new_zeros((features[0].shape[0], self.metadata_dim))
            if self.metadata_dim >= 2:
                metadata[:, self.metadata_dim - 2] = 1.0
        if metadata.ndim != 2 or metadata.shape[0] != features[0].shape[0]:
            raise ValueError(
                "PSMF metadata must have shape [batch, metadata_dim], "
                f"received {tuple(metadata.shape)} for batch {features[0].shape[0]}"
            )
        metadata = metadata.to(device=features[0].device, dtype=features[0].dtype)
        if metadata.shape[1] != self.metadata_dim:
            raise ValueError(f"PSMF expects {self.metadata_dim} metadata values, received {metadata.shape[1]}")
        embedding = self.encoder(metadata)
        conditioned = []
        for feature, affine in zip(features, self.affine):
            gamma, beta = affine(embedding).chunk(2, dim=1)
            gamma = self.strength * gamma.tanh().unsqueeze(-1).unsqueeze(-1)
            beta = self.strength * beta.tanh().unsqueeze(-1).unsqueeze(-1)
            conditioned.append(feature * (1.0 + gamma) + beta)
        return conditioned


class BASMQCSGPSMFSegment(BASMQCSGSegment):
    """E3 head: BAS + Boundary Dice + SG-MQC with physical-scale metadata fusion."""

    def __init__(self, nc=80, nm=32, npr=256, metadata_dim=11, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.psmf = PhysicalScaleMetadataFiLM(ch, metadata_dim=metadata_dim)
        self._psmf_metadata = None

    def set_metadata(self, metadata):
        """Attach metadata for the next forward pass."""
        self._psmf_metadata = metadata

    def forward(self, x):
        """Fuse metadata into P3/P4/P5 before the standard E2 segmentation head."""
        metadata = self._psmf_metadata
        self._psmf_metadata = None
        return super().forward(self.psmf(x, metadata))


class BASPSMFSegment(Segment):
    """Parameter-matched E1 head with BAS boundary supervision and optional PSMF metadata."""

    def __init__(self, nc=80, nm=32, npr=256, metadata_dim=11, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.boundary_head = nn.Sequential(Conv(self.nm, self.nm, 3), nn.Conv2d(self.nm, 1, 1))
        self.psmf = PhysicalScaleMetadataFiLM(ch, metadata_dim=metadata_dim)
        self._psmf_metadata = None

    def set_metadata(self, metadata):
        """Attach metadata for the next forward pass."""
        self._psmf_metadata = metadata

    def forward(self, x):
        """Return E1 outputs after optional metadata conditioning."""
        metadata = self._psmf_metadata
        self._psmf_metadata = None
        x = self.psmf(x, metadata)
        p = self.proto(x[0])
        boundary_logits = self.boundary_head(p) if self.training else None
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        y = Detect.forward(self, x)
        if self.training:
            return y, mc, p, boundary_logits
        return (torch.cat([y, mc], 1), p) if self.export else (torch.cat([y[0], mc], 1), (y[1], mc, p))


class BASBTQSegment(Segment):
    """BAS boundary supervision with BPR refinement and MQC calibration."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.point_refiner = BoundaryPointRefiner(self.nm)
        self.boundary_head = nn.Sequential(Conv(self.nm, self.nm, 3), nn.Conv2d(self.nm, 1, 1))
        quality_channels = max(16, ch[0] // 4)
        self.cv5 = nn.ModuleList(
            nn.Sequential(
                Conv(x, quality_channels, 3),
                Conv(quality_channels, quality_channels, 3),
                nn.Conv2d(quality_channels, 2, 1),
            )
            for x in ch
        )
        for branch in self.cv5:
            nn.init.zeros_(branch[-1].weight)
            nn.init.constant_(branch[-1].bias, 4.0)

    def forward(self, x):
        """Return BAS boundary and BTQ quality outputs while training."""
        p = self.point_refiner(self.proto(x[0]))
        boundary_logits = self.boundary_head(p) if self.training else None
        bs = p.shape[0]
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        quality_logits = torch.cat([self.cv5[i](x[i]).view(bs, 2, -1) for i in range(self.nl)], 2)
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p, (boundary_logits, quality_logits)

        quality = quality_logits.sigmoid().prod(1, keepdim=True).sqrt()
        decoded = x if self.export else x[0]
        calibrated = decoded.clone()
        calibrated[:, 4 : 4 + self.nc] *= quality
        output = torch.cat([calibrated, mc], 1)
        return (output, p) if self.export else (output, (x[1], mc, p))


class DomainConditionedStyleCalibration(nn.Module):
    """Calibrate FPN features from image-style statistics with a tiny FiLM branch."""

    def __init__(self, channels, hidden=32, strength=0.1):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.levels = len(self.channels)
        self.strength = float(strength)
        descriptor_dim = self.levels * 3
        self.encoder = nn.Sequential(nn.Linear(descriptor_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.affine = nn.ModuleList(nn.Linear(hidden, 2 * c) for c in self.channels)
        for layer in self.affine:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _descriptor(feature):
        value = feature.float()
        mean = value.mean((1, 2, 3))
        std = value.flatten(1).std(1, unbiased=False)
        local = value - F.avg_pool2d(value, kernel_size=3, stride=1, padding=1)
        texture = local.abs().mean((1, 2, 3))
        return mean, std, texture

    def forward(self, features):
        descriptor = torch.cat([torch.stack(self._descriptor(feature), 1) for feature in features], 1)
        embedding = self.encoder(descriptor.to(dtype=next(self.encoder.parameters()).dtype))
        calibrated = []
        for feature, affine in zip(features, self.affine):
            gamma, beta = affine(embedding).chunk(2, 1)
            gamma = self.strength * gamma.tanh().unsqueeze(-1).unsqueeze(-1)
            beta = self.strength * beta.tanh().unsqueeze(-1).unsqueeze(-1)
            calibrated.append(feature * (1.0 + gamma.to(feature.dtype)) + beta.to(feature.dtype))
        return calibrated


class BASBTQDCSCSegment(BASBTQSegment):
    """F1 plus single-forward domain-conditioned style calibration (DCSC)."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.dcsc = DomainConditionedStyleCalibration(ch)

    def forward(self, x):
        return super().forward(self.dcsc(x))


class FrequencyStyleConsistency(nn.Module):
    """Training-only low/high-frequency perturbation for cross-domain generalization.

    Low-frequency statistics are mixed across samples to simulate radiometric and
    sensor-style shifts. High-frequency texture receives a bounded stochastic
    perturbation to reduce shortcuts tied to resolution and local texture. The
    original path is kept at evaluation time, so this module adds no inference
    branch or deployment cost.
    """

    def __init__(
        self,
        strength=0.20,
        probability=0.5,
        kernel_size=5,
        noise=0.10,
        resolution_strength=0.20,
        resolution_probability=0.75,
        coarse_probability=0.75,
    ):
        super().__init__()
        self.strength = float(strength)
        self.probability = float(probability)
        self.kernel_size = int(kernel_size) if int(kernel_size) % 2 else int(kernel_size) + 1
        self.noise = float(noise)
        self.resolution_strength = float(resolution_strength)
        self.resolution_probability = float(resolution_probability)
        self.coarse_probability = float(coarse_probability)
        self.last_loss = None
        self.last_applied = False

    def configure(
        self,
        strength=None,
        probability=None,
        noise=None,
        resolution_strength=None,
        resolution_probability=None,
        coarse_probability=None,
    ):
        """Update training-time perturbation settings after model construction."""
        if strength is not None:
            self.strength = float(strength)
        if probability is not None:
            self.probability = float(probability)
        if noise is not None:
            self.noise = float(noise)
        if resolution_strength is not None:
            self.resolution_strength = float(resolution_strength)
        if resolution_probability is not None:
            self.resolution_probability = float(resolution_probability)
        if coarse_probability is not None:
            self.coarse_probability = float(coarse_probability)

    @staticmethod
    def _normalize_style(low):
        mean = low.mean((2, 3), keepdim=True)
        std = low.flatten(2).std(2, keepdim=True, unbiased=False).unsqueeze(-1).clamp_min(1e-4)
        return (low - mean) / std, mean, std

    def _perturb_level(self, feature):
        kernel = min(self.kernel_size, feature.shape[-2], feature.shape[-1])
        if kernel % 2 == 0:
            kernel = max(1, kernel - 1)
        low = F.avg_pool2d(feature, kernel_size=kernel, stride=1, padding=kernel // 2)
        high = feature - low
        if feature.shape[0] > 1:
            permutation = torch.randperm(feature.shape[0], device=feature.device)
        else:
            permutation = torch.zeros(1, dtype=torch.long, device=feature.device)
        normalized, mean, std = self._normalize_style(low.float())
        mixed_low = normalized * std[permutation] + mean[permutation]
        alpha = torch.empty(1, device=feature.device).uniform_(0.10, 0.30)
        style_low = low.float() + alpha * (mixed_low - low.float())
        high_std = high.float().flatten(1).std(1, keepdim=True, unbiased=False).view(-1, 1, 1, 1)
        texture_noise = torch.randn_like(high.float()) * high_std * self.noise
        beta = torch.empty(1, device=feature.device).uniform_(0.05, 0.15)
        style_high = high.float() * (1.0 + beta * torch.randn(1, device=feature.device)) + texture_noise
        # Resolution bridge: suppress high-frequency detail for the dominant
        # Sentinel-2-like coarse domain, or boost it for VHR sensor domains.
        if self.resolution_strength > 0.0 and torch.rand((), device=feature.device) < self.resolution_probability:
            smooth_high = F.avg_pool2d(high.float(), kernel_size=3, stride=1, padding=1)
            if torch.rand((), device=feature.device) < self.coarse_probability:
                resolution_view = smooth_high
            else:
                resolution_view = high.float() + 0.5 * (high.float() - smooth_high)
            bridge = min(max(self.resolution_strength, 0.0), 1.0)
            style_high = (1.0 - bridge) * style_high + bridge * resolution_view
        perturbed = style_low + style_high
        return feature + self.strength * (perturbed.to(feature.dtype) - feature)

    def forward(self, features):
        # Never retain a graph-bearing tensor during AMP checks, EMA creation,
        # validation, or inference. The trainer only needs this value on an
        # actual perturbation-enabled training forward.
        self.last_loss = None
        self.last_applied = False
        if not self.training or self.strength <= 0.0 or self.probability <= 0.0:
            return features
        if torch.rand((), device=features[0].device) > self.probability:
            return features
        perturbed = [self._perturb_level(feature) for feature in features]
        consistency = []
        for original, view in zip(features, perturbed):
            teacher = F.normalize(original.float().detach(), dim=1)
            student = F.normalize(view.float(), dim=1)
            consistency.append((1.0 - (teacher * student).sum(1)).mean())
        self.last_loss = torch.stack(consistency).mean()
        self.last_applied = True
        return perturbed


class BASBTQFMSCSegment(BASBTQSegment):
    """A+B with frequency-aware multi-scale style consistency (FMSC)."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.fmsc = FrequencyStyleConsistency()

    def forward(self, x):
        return super().forward(self.fmsc(x))


class CausalFrequencyAdapter(nn.Module):
    """Training-only causal frequency adapter inspired by causal token tuning.

    The local low-frequency component is treated as the stable structural signal,
    while the high-frequency residual is treated as a domain/style carrier. A
    small learnable token bank calibrates the structural component and a bounded
    gate suppresses high-frequency shortcuts. Evaluation keeps the original
    features, so the adapter has no deployment-time branch.
    """

    def __init__(self, channels, strength=0.15, probability=0.75, high_suppression=0.20, token_scale=0.10):
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.strength = float(strength)
        self.probability = float(probability)
        self.high_suppression = float(high_suppression)
        self.token_scale = float(token_scale)
        self.tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, c, 1, 1)) for c in self.channels]
        )
        self.gates = nn.ModuleList()
        for c in self.channels:
            hidden = max(8, min(64, c // 8))
            self.gates.append(
                nn.Sequential(
                    nn.Conv2d(c, hidden, 1),
                    nn.SiLU(),
                    nn.Conv2d(hidden, c, 1),
                )
            )
        self.last_loss = None
        self.last_applied = False

    def configure(self, strength=None, probability=None, high_suppression=None, token_scale=None):
        """Update training-only adapter settings after model construction."""
        if strength is not None:
            self.strength = float(strength)
        if probability is not None:
            self.probability = float(probability)
        if high_suppression is not None:
            self.high_suppression = float(high_suppression)
        if token_scale is not None:
            self.token_scale = float(token_scale)

    @staticmethod
    def _low_high(feature):
        kernel = min(5, feature.shape[-2], feature.shape[-1])
        if kernel % 2 == 0:
            kernel -= 1
        low = F.avg_pool2d(feature, kernel_size=max(1, kernel), stride=1, padding=max(0, kernel // 2))
        return low, feature - low

    def _adapt_level(self, feature, token, gate):
        low, high = self._low_high(feature.float())
        # The token is a learnable structural correction, analogous to a causal
        # token operating on the stable component rather than raw style cues.
        structural = low + self.token_scale * torch.tanh(token) * torch.tanh(low)
        gate_value = torch.sigmoid(gate(F.adaptive_avg_pool2d(low, 1)))
        suppression = self.high_suppression * gate_value
        high_causal = high * (1.0 - suppression)
        adapted = structural + high_causal
        return feature + self.strength * (adapted.to(feature.dtype) - feature)

    def forward(self, features):
        self.last_loss = None
        self.last_applied = False
        if not self.training or self.strength <= 0.0 or self.probability <= 0.0:
            return features
        if torch.rand((), device=features[0].device) > self.probability:
            return features
        adapted = [self._adapt_level(feature, token, gate) for feature, token, gate in zip(features, self.tokens, self.gates)]
        consistency = []
        for original, view in zip(features, adapted):
            stable = F.normalize(original.float().detach(), dim=1)
            causal = F.normalize(view.float(), dim=1)
            consistency.append((1.0 - (stable * causal).sum(1)).mean())
        self.last_loss = torch.stack(consistency).mean()
        self.last_applied = True
        return adapted


class BASBTQCFASegment(BASBTQSegment):
    """A+B with causal frequency adaptation for cross-source generalization."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.cfa = CausalFrequencyAdapter(ch)

    def forward(self, x):
        return super().forward(self.cfa(x))


class BASBTQPSMFSegment(BASBTQSegment):
    """A + B head: F2 boundary/quality refinement conditioned on physical metadata."""

    def __init__(self, nc=80, nm=32, npr=256, metadata_dim=11, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.psmf = PhysicalScaleMetadataFiLM(ch, metadata_dim=metadata_dim)
        self._psmf_metadata = None

    def set_metadata(self, metadata):
        """Attach metadata for the next forward pass."""
        self._psmf_metadata = metadata

    def forward(self, x):
        """Condition F2 features before running the unchanged BTQ head."""
        metadata = self._psmf_metadata
        self._psmf_metadata = None
        return super().forward(self.psmf(x, metadata))


class ScaleStyleResidualMoE(nn.Module):
    """Route scale/style-conditioned residual adapters without replacing shared features."""

    MODES = {"p0", "visual", "metadata", "hybrid"}

    def __init__(self, channels, metadata_dim=11, experts=3, hidden=48, strength=0.25):
        super().__init__()
        if experts < 2:
            raise ValueError("SSC-MoE requires at least two experts")
        self.metadata_dim = int(metadata_dim)
        self.experts = int(experts)
        self.levels = len(channels)
        self.strength = float(strength)
        descriptor_dim = self.levels * 2 + self.metadata_dim
        self.router = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.levels * self.experts),
        )
        self.adapters = nn.ModuleList()
        for channels_i in channels:
            bottleneck = max(16, channels_i // 8)
            level_adapters = nn.ModuleList()
            for _ in range(self.experts):
                output = nn.Conv2d(bottleneck, channels_i, 1, bias=True)
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
                level_adapters.append(
                    nn.Sequential(
                        Conv(channels_i, bottleneck, 1),
                        DWConv(bottleneck, bottleneck, 3),
                        output,
                    )
                )
            self.adapters.append(level_adapters)
        self.routing_mode = "hybrid"
        self.temperature = 1.0
        self.last_routing = None

    def set_routing(self, mode="hybrid", temperature=1.0):
        """Configure which conditioning signals are visible to the router."""
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(f"Unknown SSC-MoE routing mode: {mode}")
        self.routing_mode = mode
        self.temperature = max(float(temperature), 0.05)

    def _visual_descriptor(self, features):
        descriptor = []
        for feature in features:
            reduced = feature.float().flatten(1)
            descriptor.extend((reduced.mean(1), reduced.std(1, unbiased=False)))
        return torch.stack(descriptor, 1)

    def forward(self, features, metadata=None):
        """Apply a bounded mixture of zero-initialized residual experts at each FPN level."""
        batch = features[0].shape[0]
        device, dtype = features[0].device, features[0].dtype
        if metadata is None:
            metadata = torch.zeros((batch, self.metadata_dim), device=device, dtype=dtype)
            if self.metadata_dim >= 2:
                metadata[:, self.metadata_dim - 2] = 1.0
        metadata = metadata.to(device=device, dtype=torch.float32)
        if metadata.shape != (batch, self.metadata_dim):
            raise ValueError(
                f"SSC-MoE metadata must have shape {(batch, self.metadata_dim)}, got {tuple(metadata.shape)}"
            )

        visual = self._visual_descriptor(features)
        if self.routing_mode in {"p0", "metadata"}:
            visual = torch.zeros_like(visual)
        if self.routing_mode in {"p0", "visual"}:
            metadata = torch.zeros_like(metadata)
        router_dtype = next(self.router.parameters()).dtype
        descriptor = torch.cat((visual, metadata), 1).to(dtype=router_dtype)
        logits = self.router(descriptor).view(batch, self.levels, self.experts)
        weights = (logits / self.temperature).softmax(-1).to(dtype=dtype)

        # Encourage dataset-level expert utilization without forcing every image to use all experts.
        utilization = weights.float().mean((0, 1))
        router_balance_loss = self.experts * ((utilization - 1.0 / self.experts) ** 2).mean()
        # Training tensors with a grad_fn must not be stored on the module because
        # ModelEMA deep-copies the complete model before the first optimizer step.
        self.last_routing = None if self.training else weights.detach()

        outputs = []
        for level, feature in enumerate(features):
            residual = torch.zeros_like(feature)
            for expert, adapter in enumerate(self.adapters[level]):
                gate = weights[:, level, expert].view(batch, 1, 1, 1)
                residual = residual + gate * adapter(feature)
            outputs.append(feature + self.strength * residual)
        return outputs, router_balance_loss


class BASBTQSSCMoESegment(BASBTQSegment):
    """A+B head: BAS-BTQ with scale/style-conditioned residual expert routing."""

    def __init__(self, nc=80, nm=32, npr=256, metadata_dim=11, experts=3, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.ssc_moe = ScaleStyleResidualMoE(ch, metadata_dim=metadata_dim, experts=experts)
        self._ssc_metadata = None
        self.routing_mode = "hybrid"
        self.router_temperature = 1.0

    def set_metadata(self, metadata):
        """Attach physical metadata for the next forward pass."""
        self._ssc_metadata = metadata

    def set_routing(self, mode="hybrid", temperature=1.0):
        """Set routing inputs and softmax temperature."""
        self.routing_mode = str(mode)
        self.router_temperature = float(temperature)

    def forward(self, x):
        """Route residual experts before the unchanged A segmentation head."""
        metadata = self._ssc_metadata
        self._ssc_metadata = None
        self.ssc_moe.set_routing(self.routing_mode, self.router_temperature)
        features, router_balance_loss = self.ssc_moe(x, metadata)
        outputs = super().forward(features)
        if not self.training:
            return outputs
        predictions, coefficients, prototypes, auxiliary = outputs
        boundary_logits, quality_logits = auxiliary
        return predictions, coefficients, prototypes, (boundary_logits, quality_logits, router_balance_loss)


class P2LiteFusion(nn.Module):
    """Fuse a compressed stride-4 detail map into mask prototypes."""

    def __init__(self, p2_channels, mask_channels):
        super().__init__()
        self.detail = nn.Sequential(Conv(p2_channels, mask_channels, 1), DWConv(mask_channels, mask_channels, 3))
        self.gate = nn.Conv2d(mask_channels * 2, mask_channels, 1)
        self.residual = nn.Conv2d(mask_channels, mask_channels, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, p2, prototypes):
        """Return fused prototypes and the lightweight detail map used by ROI refinement."""
        detail = self.detail(p2)
        if detail.shape[-2:] != prototypes.shape[-2:]:
            detail = nn.functional.interpolate(detail, prototypes.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat((prototypes, detail), 1)).sigmoid()
        return prototypes + gate * self.residual(detail), detail


class InstanceCascadeRefiner(nn.Module):
    """Refine instance masks in two ROI stages using P2 detail and instance context."""

    def __init__(self, channels=32, hidden_channels=16, coarse_size=56, fine_size=112):
        super().__init__()
        self.coarse_size = int(coarse_size)
        self.fine_size = int(fine_size)
        inputs = channels * 2 + 1
        self.stage1 = nn.Sequential(
            Conv(inputs, hidden_channels, 1),
            DWConv(hidden_channels, hidden_channels, 3),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.stage2 = nn.Sequential(
            Conv(inputs, hidden_channels, 1),
            DWConv(hidden_channels, hidden_channels, 3),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.stage1[-1].weight)
        nn.init.zeros_(self.stage1[-1].bias)
        nn.init.zeros_(self.stage2[-1].weight)
        nn.init.zeros_(self.stage2[-1].bias)

    @staticmethod
    def crop_features(features, boxes, image_size, output_size, image_indices=None):
        """Differentiably crop N boxes from a CHW or NCHW feature map."""
        if features.ndim == 3:
            features = features.unsqueeze(0)
        if image_indices is not None:
            features = features.index_select(0, image_indices.long())
        elif features.shape[0] == 1 and boxes.shape[0] > 1:
            features = features.expand(boxes.shape[0], -1, -1, -1)
        if features.shape[0] != boxes.shape[0]:
            raise ValueError("Feature and box batch dimensions must match for ROI cropping")
        if boxes.numel() == 0:
            return features.new_empty((0, features.shape[1], output_size, output_size))

        image_h, image_w = float(image_size[0]), float(image_size[1])
        steps = torch.linspace(0.0, 1.0, output_size, device=boxes.device, dtype=boxes.dtype)
        grid_y, grid_x = torch.meshgrid(steps, steps, indexing="ij")
        x1, y1, x2, y2 = boxes.unbind(1)
        xs = x1[:, None, None] + grid_x * (x2 - x1)[:, None, None]
        ys = y1[:, None, None] + grid_y * (y2 - y1)[:, None, None]
        grid = torch.stack(
            (2.0 * xs / max(image_w - 1.0, 1.0) - 1.0, 2.0 * ys / max(image_h - 1.0, 1.0) - 1.0),
            -1,
        ).to(dtype=features.dtype)
        return nn.functional.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def forward(self, detail, coarse_logits, boxes, image_size, image_indices=None):
        """Return fine ROI logits with coarse-mask identity initialization."""
        detail_coarse = self.crop_features(
            detail, boxes, image_size, self.coarse_size, image_indices=image_indices
        )
        coarse_roi = self.crop_features(coarse_logits.unsqueeze(1), boxes, image_size, self.coarse_size)
        weights = coarse_roi.sigmoid()
        context = (detail_coarse * weights).sum((2, 3), keepdim=True) / weights.sum((2, 3), keepdim=True).clamp_min(1.0)
        context = context.expand(-1, -1, self.coarse_size, self.coarse_size)
        stage1 = coarse_roi + self.stage1(torch.cat((detail_coarse, coarse_roi, context), 1))

        detail_fine = self.crop_features(
            detail, boxes, image_size, self.fine_size, image_indices=image_indices
        )
        stage1 = nn.functional.interpolate(stage1, (self.fine_size, self.fine_size), mode="bilinear", align_corners=False)
        context = nn.functional.interpolate(context, (self.fine_size, self.fine_size), mode="nearest")
        return stage1 + self.stage2(torch.cat((detail_fine, stage1, context), 1))


class HARMSegment(Segment):
    """High-resolution, assignment-aware, ROI-refined YOLO segmentation head."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        if len(ch) != 4:
            raise ValueError(f"HARMSegment expects channels for P2/P3/P4/P5, received {len(ch)} levels")
        super().__init__(nc, nm, npr, ch[1:])
        self.p2_fusion = P2LiteFusion(ch[0], self.nm)
        self.instance_refiner = InstanceCascadeRefiner(self.nm)

    def refine_instances(self, detail, coarse_logits, boxes, image_size, image_indices=None):
        """Refine N coarse instance logits inside their detection boxes."""
        return self.instance_refiner(detail, coarse_logits, boxes, image_size, image_indices=image_indices)

    def forward(self, x):
        """Detect on P3-P5 while reserving P2 for mask fusion and instance refinement."""
        p2, detection_features = x[0], x[1:]
        p, detail = self.p2_fusion(p2, self.proto(detection_features[0]))
        bs = p.shape[0]
        mc = torch.cat(
            [self.cv4[i](detection_features[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
        )
        detection = Detect.forward(self, detection_features)
        if self.training:
            detail = detail + sum(parameter.sum() * 0.0 for parameter in self.instance_refiner.parameters())
            return detection, mc, p, detail
        output = torch.cat((detection if self.export else detection[0], mc), 1)
        return (output, p) if self.export else (output, (detection[1], mc, p, detail))


class OBB(Detect):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class WorldDetect(Detect):
    """Head for integrating YOLOv8 detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLOv8 detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def switch_to_deploy(self):
        del self.cv2, self.cv3
