# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.atss import ATSSAssigner, generate_anchors
from .metrics import bbox_iou, probiou, bbox_mpdiou, bbox_inner_iou, bbox_focaler_iou, bbox_inner_mpdiou, bbox_focaler_mpdiou, wasserstein_loss, gcd_loss, WiseIouLoss
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

import math
class SlideLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(SlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element

    def forward(self, pred, true, auto_iou=0.5):
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class EMASlideLoss:
    def __init__(self, loss_fcn, decay=0.999, tau=2000):
        super(EMASlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        self.is_train = True
        self.updates = 0
        self.iou_mean = 1.0
    
    def __call__(self, pred, true, auto_iou=0.5):
        if self.is_train and auto_iou != -1:
            self.updates += 1
            d = self.decay(self.updates)
            self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou.detach())
        auto_iou = self.iou_mean
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

class VarifocalLoss_YOLO(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        """Initialize the VarifocalLoss class."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_score, gt_score):
        """Computes varfocal loss."""
        
        weight = self.alpha * (pred_score.sigmoid() - gt_score).abs().pow(self.gamma) * (gt_score <= 0.0).float() + gt_score * (gt_score > 0.0).float()
        with torch.cuda.amp.autocast(enabled=False):
            return F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') * weight

class QualityfocalLoss_YOLO(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred_score, gt_score, gt_target_pos_mask):
        # negatives are supervised by 0 quality score
        pred_sigmoid = pred_score.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_score.shape)
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(pred_score, zerolabel, reduction='none') * scale_factor.pow(self.beta)
        
        scale_factor = gt_score[gt_target_pos_mask] - pred_sigmoid[gt_target_pos_mask]
        with torch.cuda.amp.autocast(enabled=False):
            loss[gt_target_pos_mask] = F.binary_cross_entropy_with_logits(pred_score[gt_target_pos_mask], gt_score[gt_target_pos_mask], reduction='none') * scale_factor.abs().pow(self.beta)
        return loss

class FocalLoss_YOLO(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, gamma=1.5, alpha=0.25):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, label):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss

class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        
        # NWD
        self.nwd_loss = False
        # GCD
        self.gcd_loss = False
        self.iou_ratio = 0.5 # total_iou_loss = self.iou_ratio * iou_loss + (1 - self.iou_ratio) * (nwd_loss or gcd_loss)
        
        # WiseIOU
        self.use_wiseiou = False
        if self.use_wiseiou:
            self.wiou_loss = WiseIouLoss(ltype='WIoU', monotonous=False, inner_iou=False, focaler_iou=False)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, mpdiou_hw=None):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if self.use_wiseiou:
            wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95).unsqueeze(-1)
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'scale':0.0}).unsqueeze(-1) # Wise-ShapeIoU,Wise-Inner-ShapeIoU,Wise-Focaler-ShapeIoU
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'mpdiou_hw':mpdiou_hw[fg_mask]}).unsqueeze(-1) # Wise-MPDIoU,Wise-Inner-MPDIoU,Wise-Focaler-MPDIoU
            loss_iou = (wiou * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            # iou = bbox_inner_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, ratio=0.7)
            # iou = bbox_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask])
            # iou = bbox_inner_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask], ratio=0.7)
            # iou = bbox_focaler_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, d=0.0, u=0.95)
            # iou = bbox_focaler_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask], d=0.0, u=0.95)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
                
        if self.nwd_loss:
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            nwd_loss = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = self.iou_ratio * loss_iou +  (1 - self.iou_ratio) * nwd_loss
        elif self.gcd_loss:
            gcd = gcd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            gcd_l = ((1.0 - gcd) * weight.squeeze()).sum() / target_scores_sum  # 0.0002不需要了，之前的code有尺寸上的问题，现已修复
            loss_iou = self.iou_ratio * loss_iou +  (1 - self.iou_ratio) * gcd_l

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # self.bce = EMASlideLoss(nn.BCEWithLogitsLoss(reduction='none'))  # Exponential Moving Average Slide Loss
        # self.bce = SlideLoss(nn.BCEWithLogitsLoss(reduction='none')) # Slide Loss
        # self.bce = FocalLoss_YOLO(alpha=0.25, gamma=1.5) # FocalLoss
        # self.bce = VarifocalLoss_YOLO(alpha=0.75, gamma=2.0) # VarifocalLoss
        # self.bce = QualityfocalLoss_YOLO(beta=2.0) # QualityfocalLoss
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        if hasattr(m, 'dfl_aux'):
            self.assigner_aux = TaskAlignedAssigner(topk=13, num_classes=self.nc, alpha=0.5, beta=6.0)
            self.aux_loss_ratio = 0.25
        # self.assigner = ATSSAssigner(9, num_classes=self.nc)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        
        # ATSS use        
        self.grid_cell_offset = 0.5
        self.fpn_strides = list(self.stride.detach().cpu().numpy())
        self.grid_cell_size = 5.0

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        if hasattr(self, 'assigner_aux'):
            loss, batch_size = self.compute_loss_aux(preds, batch)
        else:
            loss, batch_size = self.compute_loss(preds, batch)
        return loss.sum() * batch_size, loss.detach()

    def compute_loss(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        feats = feats[:self.stride.size(0)]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # ATSS
        if isinstance(self.assigner, ATSSAssigner):
            anchors, _, n_anchors_list, _ = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
            target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(anchors, n_anchors_list, gt_labels, gt_bboxes, mask_gt, pred_bboxes.detach() * stride_tensor)
        # TAL
        else:
            target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        if isinstance(self.bce, (nn.BCEWithLogitsLoss, FocalLoss_YOLO)):
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        elif isinstance(self.bce, VarifocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                cls_iou_targets.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * cls_iou_targets
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype)).sum() / max(fg_mask.sum(), 1)  # BCE
        elif isinstance(self.bce, QualityfocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                targets_onehot = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * targets_onehot
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                targets_onehot_pos = torch.where(fg_scores_mask > 0, targets_onehot, 0)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot_pos = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype), targets_onehot_pos.to(torch.bool)).sum() / max(fg_mask.sum(), 1)

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask, ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0))

        if isinstance(self.bce, (EMASlideLoss, SlideLoss)):
            if fg_mask.sum():
                auto_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).mean()
            else:
                auto_iou = -1
            loss[1] = self.bce(pred_scores, target_scores.to(dtype), auto_iou).sum() / target_scores_sum  # BCE
        
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return loss, batch_size
    
    def compute_loss_aux(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats_all = preds[1] if isinstance(preds, tuple) else preds
        if len(feats_all) == self.stride.size(0):
            return self.compute_loss(preds, batch)
        feats, feats_aux = feats_all[:self.stride.size(0)], feats_all[self.stride.size(0):]
        
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_distri_aux, pred_scores_aux = torch.cat([xi.view(feats_aux[0].shape[0], self.no, -1) for xi in feats_aux], 2).split((self.reg_max * 4, self.nc), 1)

        pred_scores, pred_distri = pred_scores.permute(0, 2, 1).contiguous(), pred_distri.permute(0, 2, 1).contiguous()
        pred_scores_aux, pred_distri_aux = pred_scores_aux.permute(0, 2, 1).contiguous(), pred_distri_aux.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_bboxes_aux = self.bbox_decode(anchor_points, pred_distri_aux)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux, _ = self.assigner_aux(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)

        # cls loss
        if isinstance(self.bce, nn.BCEWithLogitsLoss):
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
            loss[1] += self.bce(pred_scores_aux, target_scores_aux.to(dtype)).sum() / target_scores_sum_aux * self.aux_loss_ratio  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            target_bboxes_aux /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                            target_scores_sum, fg_mask, ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0))
            aux_loss_0, aux_loss_2 = self.bbox_loss(pred_distri_aux, pred_bboxes_aux, anchor_points, target_bboxes_aux, target_scores_aux,
                                            target_scores_sum_aux, fg_mask_aux, ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0))
            
            loss[0] += aux_loss_0 * self.aux_loss_ratio
            loss[2] += aux_loss_2 * self.aux_loss_ratio

        if isinstance(self.bce, (EMASlideLoss, SlideLoss)):
            auto_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).mean()
            loss[1] = self.bce(pred_scores, target_scores.to(dtype), auto_iou).sum() / target_scores_sum  # BCE
            loss[1] += self.bce(pred_scores_aux, target_scores_aux.to(dtype), -1).sum() / target_scores_sum_aux * self.aux_loss_ratio  # BCE

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # MoE auxiliary loss
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     loss[1] += m.aux_loss * 1.5

        # return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
        return loss, batch_size

class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        train_outputs = preds if isinstance(preds[0], list) else preds[1]
        feats, pred_masks, proto, *aux_outputs = train_outputs
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # ATSS
        if isinstance(self.assigner, ATSSAssigner):
            anchors, _, n_anchors_list, _ = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
            target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(anchors, n_anchors_list, gt_labels, gt_bboxes, mask_gt, pred_bboxes.detach() * stride_tensor)
        else:
            target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        if isinstance(self.bce, (nn.BCEWithLogitsLoss, FocalLoss_YOLO)):
            loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        elif isinstance(self.bce, VarifocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                cls_iou_targets.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * cls_iou_targets
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[2] = self.bce(pred_scores, cls_iou_targets.to(dtype)).sum() / max(fg_mask.sum(), 1)  # BCE
        elif isinstance(self.bce, QualityfocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                targets_onehot = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * targets_onehot
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                targets_onehot_pos = torch.where(fg_scores_mask > 0, targets_onehot, 0)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot_pos = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[2] = self.bce(pred_scores, cls_iou_targets.to(dtype), targets_onehot_pos.to(torch.bool)).sum() / max(fg_mask.sum(), 1)

        auxiliary_loss = proto.sum() * 0.0
        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0)
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )
            if aux_outputs:
                auxiliary_loss = self.calculate_auxiliary_segmentation_loss(
                    aux_outputs[0],
                    fg_mask,
                    masks,
                    target_gt_idx,
                    target_bboxes,
                    pred_bboxes.detach() * stride_tensor,
                    target_scores,
                    batch_idx,
                    proto,
                    pred_masks,
                    imgsz,
                    self.overlap,
                )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if aux_outputs:
                for auxiliary_output in aux_outputs:
                    tensors = auxiliary_output if isinstance(auxiliary_output, (tuple, list)) else (auxiliary_output,)
                    for tensor in tensors:
                        if torch.is_tensor(tensor):
                            loss[1] += (tensor * 0).sum()
        
        if isinstance(self.bce, (EMASlideLoss, SlideLoss)):
            if fg_mask.sum():
                auto_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).mean()
            else:
                auto_iou = -1
            loss[2] = self.bce(pred_scores, target_scores.to(dtype), auto_iou).sum() / target_scores_sum  # BCE

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain
        loss[1] += auxiliary_loss

        # MoE auxiliary loss
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     loss[1] += m.aux_loss * 1.5

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def calculate_auxiliary_segmentation_loss(
        self,
        topology_logits,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        pred_bboxes,
        target_scores,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Return zero for standard segmentation heads without auxiliary supervision."""
        return proto.sum() * 0.0

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8MPTSSegmentationLoss(v8SegmentationLoss):
    """YOLO segmentation loss with parcel boundary, distance, and separation supervision."""

    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.boundary_gain = float(model.args.mpts_boundary)
        self.distance_gain = float(model.args.mpts_distance)
        self.separation_gain = float(model.args.mpts_separation)
        self.scale_normalized = bool(model.args.mpts_scale_normalized)
        self.distance_radius = max(1, int(model.args.mpts_distance_radius))
        self.boundary_width = max(1, int(model.args.mpts_boundary_width))
        self.adjacency_radius = max(1, int(model.args.mpts_adjacency_radius))
        self.warmup_epochs = max(0, int(model.args.mpts_warmup_epochs))
        self.last_mpts_losses = {}

    @staticmethod
    def _instance_label_maps(masks, batch_idx, batch_size, overlap):
        """Convert overlap or per-instance masks into one local instance-ID map per image."""
        if overlap:
            if masks.ndim != 3 or masks.shape[0] != batch_size:
                raise ValueError(f"Expected overlap masks with shape (B,H,W), received {tuple(masks.shape)}")
            return masks.round().long()

        height, width = masks.shape[-2:]
        label_maps = torch.zeros((batch_size, height, width), dtype=torch.long, device=masks.device)
        flat_batch_idx = batch_idx.view(-1)
        for image_idx in range(batch_size):
            instance_masks = masks[flat_batch_idx == image_idx]
            for instance_idx, instance_mask in enumerate(instance_masks):
                label_maps[image_idx][instance_mask > 0.5] = instance_idx + 1
        return label_maps

    @staticmethod
    def _boundary_from_labels(label_maps, width=1):
        """Preserve both outer borders and borders between touching instance IDs."""
        boundary = torch.zeros_like(label_maps, dtype=torch.bool)

        left, right = label_maps[..., :-1], label_maps[..., 1:]
        changes = (left != right) & ((left > 0) | (right > 0))
        boundary[..., :-1] |= changes
        boundary[..., 1:] |= changes

        top, bottom = label_maps[..., :-1, :], label_maps[..., 1:, :]
        changes = (top != bottom) & ((top > 0) | (bottom > 0))
        boundary[..., :-1, :] |= changes
        boundary[..., 1:, :] |= changes

        if width > 1:
            radius = width - 1
            boundary = F.max_pool2d(
                boundary.float().unsqueeze(1), 2 * radius + 1, stride=1, padding=radius
            ).squeeze(1) > 0
        return boundary

    @staticmethod
    def _binary_erode(mask):
        """Apply one 3x3 binary erosion with background padding."""
        padded = F.pad(mask, (1, 1, 1, 1), value=0.0)
        return 1.0 - F.max_pool2d(1.0 - padded, 3, stride=1)

    def _distance_target(self, label_maps, base_boundary):
        """Build a truncated distance transform, optionally normalized by parcel area."""
        region = (label_maps > 0).unsqueeze(1)
        current = (region & ~base_boundary.unsqueeze(1)).float()
        distance = torch.zeros_like(current)
        for _ in range(self.distance_radius):
            distance += current
            current = self._binary_erode(current)

        if not self.scale_normalized:
            return (distance / float(self.distance_radius)).clamp_(0.0, 1.0), region

        scale_map = torch.ones_like(distance)
        for image_idx in range(label_maps.shape[0]):
            ids = torch.unique(label_maps[image_idx])
            for instance_id in ids[ids > 0]:
                instance = label_maps[image_idx] == instance_id
                equivalent_radius = torch.sqrt(instance.sum().float() / math.pi).clamp_min(1.0)
                scale_map[image_idx, 0][instance] = equivalent_radius
        return (distance / scale_map).clamp_(0.0, 1.0), region

    @staticmethod
    def _boundary_loss(logits, target):
        target = target.float().unsqueeze(1)
        positive = target.sum()
        negative = target.numel() - positive
        pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        probability = logits.sigmoid()
        intersection = (probability * target).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + target.sum() + 1.0)
        return bce + dice

    @staticmethod
    def _distance_loss(logits, target, region):
        prediction = logits.sigmoid()
        foreground = region.expand_as(prediction)
        if foreground.any():
            foreground_loss = F.smooth_l1_loss(prediction[foreground], target[foreground])
        else:
            foreground_loss = prediction.sum() * 0.0
        background = ~foreground
        background_loss = (
            F.smooth_l1_loss(prediction[background], torch.zeros_like(prediction[background]))
            if background.any()
            else prediction.sum() * 0.0
        )
        return foreground_loss + 0.1 * background_loss

    def _separation_loss(
        self, label_maps, fg_mask, target_gt_idx, pred_bboxes, target_scores, proto, pred_masks, imgsz
    ):
        """Penalize simultaneous foreground probability around adjacent GT instances."""
        losses = []
        mask_height, mask_width = label_maps.shape[-2:]
        box_scale = torch.tensor(
            [mask_width / imgsz[1], mask_height / imgsz[0], mask_width / imgsz[1], mask_height / imgsz[0]],
            device=proto.device,
            dtype=pred_bboxes.dtype,
        )

        for image_idx in range(label_maps.shape[0]):
            positive_anchors = torch.where(fg_mask[image_idx])[0]
            if positive_anchors.numel() == 0:
                continue

            quality = target_scores[image_idx].sum(-1)
            selected_masks = {}
            local_gt_ids = torch.unique(target_gt_idx[image_idx, positive_anchors])
            for gt_idx in local_gt_ids:
                candidates = positive_anchors[target_gt_idx[image_idx, positive_anchors] == gt_idx]
                anchor_idx = candidates[quality[candidates].argmax()]
                mask_logit = torch.einsum("n,nhw->hw", pred_masks[image_idx, anchor_idx], proto[image_idx])
                mask_probability = mask_logit.sigmoid().unsqueeze(0)
                crop_box = (pred_bboxes[image_idx, anchor_idx] * box_scale).unsqueeze(0)
                selected_masks[int(gt_idx)] = crop_mask(mask_probability, crop_box)[0]

            if len(selected_masks) < 2:
                continue

            radius = self.adjacency_radius
            kernel = 2 * radius + 1
            expanded = {}
            for gt_idx in selected_masks:
                instance = (label_maps[image_idx] == gt_idx + 1).float()[None, None]
                expanded[gt_idx] = F.max_pool2d(instance, kernel, stride=1, padding=radius)[0, 0] > 0

            gt_ids = sorted(selected_masks)
            for position, first_id in enumerate(gt_ids):
                for second_id in gt_ids[position + 1 :]:
                    contact_band = expanded[first_id] & expanded[second_id]
                    if contact_band.any():
                        overlap = selected_masks[first_id] * selected_masks[second_id]
                        losses.append(overlap[contact_band].mean())

        return torch.stack(losses).mean() if losses else proto.sum() * 0.0

    def _warmup_factor(self):
        if self.warmup_epochs == 0:
            return 1.0
        current_epoch = int(getattr(self.model, "mpts_epoch", self.warmup_epochs - 1))
        return min(1.0, max(0.0, (current_epoch + 1) / self.warmup_epochs))

    def calculate_auxiliary_segmentation_loss(
        self,
        topology_logits,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        pred_bboxes,
        target_scores,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        if topology_logits is None:
            return proto.sum() * 0.0
        if topology_logits.shape[1] != 2:
            raise ValueError(f"MPTS topology head must produce 2 channels, received {topology_logits.shape[1]}")
        if topology_logits.shape[-2:] != proto.shape[-2:]:
            topology_logits = F.interpolate(topology_logits, proto.shape[-2:], mode="bilinear", align_corners=False)

        label_maps = self._instance_label_maps(masks, batch_idx, proto.shape[0], overlap)
        base_boundary = self._boundary_from_labels(label_maps, width=1)
        boundary_target = self._boundary_from_labels(label_maps, width=self.boundary_width)
        distance_target, region = self._distance_target(label_maps, base_boundary)

        boundary_loss = self._boundary_loss(topology_logits[:, 0:1], boundary_target)
        distance_loss = self._distance_loss(topology_logits[:, 1:2], distance_target, region)
        separation_loss = (
            self._separation_loss(
                label_maps,
                fg_mask,
                target_gt_idx,
                pred_bboxes,
                target_scores,
                proto,
                pred_masks,
                imgsz,
            )
            if self.separation_gain > 0
            else proto.sum() * 0.0
        )

        warmup = self._warmup_factor()
        total = warmup * (
            self.boundary_gain * boundary_loss
            + self.distance_gain * distance_loss
            + self.separation_gain * separation_loss
        )
        self.last_mpts_losses = {
            "boundary": float(boundary_loss.detach()),
            "distance": float(distance_loss.detach()),
            "separation": float(separation_loss.detach()),
            "warmup": warmup,
        }
        return total


class v8BTQSegmentationLoss(v8SegmentationLoss):
    """Boundary-point refinement and mask/boundary quality calibration loss."""

    def __init__(self, model):
        super().__init__(model)
        self.quality_gain = float(model.args.btq_quality)
        self.point_gain = float(model.args.btq_point)
        self.boundary_width = max(1, int(model.args.btq_boundary_width))
        self.uncertain_points = max(0, int(model.args.btq_uncertain_points))
        self.last_btq_losses = {}

    @staticmethod
    def _binary_boundary(mask, width=1):
        """Extract a two-sided morphological boundary from N binary masks."""
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float().unsqueeze(1)
        radius = max(1, int(width))
        kernel = 2 * radius + 1
        dilated = F.max_pool2d(mask, kernel, stride=1, padding=radius)
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel, stride=1, padding=radius)
        return (dilated - eroded).squeeze(1) > 0

    @staticmethod
    def _soft_iou_target(logits, target, valid):
        """Compute a differentiable-mask IoU target, detached before quality supervision."""
        probability = logits.sigmoid() * valid
        target = target * valid
        intersection = (probability * target).sum((1, 2))
        union = (probability + target - probability * target).sum((1, 2))
        return ((intersection + 1.0) / (union + 1.0)).detach()

    @classmethod
    def _boundary_f1_target(cls, logits, target, valid, tolerance=2):
        """Compute tolerant Boundary-F1 targets for quality calibration."""
        prediction = ((logits.detach() > 0) & valid.bool()).float()
        target = (target > 0.5).float() * valid
        pred_boundary = cls._binary_boundary(prediction, 1)
        target_boundary = cls._binary_boundary(target, 1)
        radius = max(0, int(tolerance))
        if radius:
            kernel = 2 * radius + 1
            pred_band = F.max_pool2d(pred_boundary.float().unsqueeze(1), kernel, 1, radius).squeeze(1) > 0
            target_band = F.max_pool2d(target_boundary.float().unsqueeze(1), kernel, 1, radius).squeeze(1) > 0
        else:
            pred_band, target_band = pred_boundary, target_boundary

        precision = (pred_boundary & target_band).sum((1, 2)).float() / pred_boundary.sum((1, 2)).clamp_min(1)
        recall = (target_boundary & pred_band).sum((1, 2)).float() / target_boundary.sum((1, 2)).clamp_min(1)
        return ((2.0 * precision * recall + 1e-6) / (precision + recall + 1e-6)).detach()

    @staticmethod
    def _uncertain_point_mask(logits, valid, count):
        """Select pixels closest to the decision boundary inside each instance crop."""
        selected = torch.zeros_like(valid, dtype=torch.bool)
        if count <= 0 or logits.numel() == 0:
            return selected
        flat_valid = valid.bool().flatten(1)
        uncertainty = -logits.detach().abs().flatten(1)
        uncertainty = uncertainty.masked_fill(~flat_valid, float("-inf"))
        k = min(int(count), uncertainty.shape[1])
        indices = uncertainty.topk(k, dim=1).indices
        selected.flatten(1).scatter_(1, indices, True)
        return selected & valid.bool()

    def _quality_and_point_loss(self, quality_logits, mask_logits, target, valid):
        """Supervise two quality channels and focus mask BCE on difficult boundary points."""
        mask_iou = self._soft_iou_target(mask_logits, target, valid)
        boundary_f1 = self._boundary_f1_target(
            mask_logits, target, valid, tolerance=self.boundary_width
        )
        quality_target = torch.stack((mask_iou, boundary_f1), 1).to(quality_logits.dtype)
        quality_loss = F.binary_cross_entropy_with_logits(quality_logits, quality_target)

        boundary_points = self._binary_boundary(target, self.boundary_width) & valid.bool()
        uncertain_points = self._uncertain_point_mask(mask_logits, valid, self.uncertain_points)
        selected = boundary_points | uncertain_points
        point_bce = F.binary_cross_entropy_with_logits(mask_logits, target, reduction="none")
        normalizer = selected.sum((1, 2)).clamp_min(1)
        point_loss = (point_bce * selected).sum((1, 2)).div(normalizer).mean()
        return quality_loss, point_loss

    def calculate_auxiliary_segmentation_loss(
        self,
        quality_logits,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        pred_bboxes,
        target_scores,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Train instance mask quality and apply focused supervision to refined prototypes."""
        if quality_logits is None:
            return proto.sum() * 0.0
        if quality_logits.shape[1] != 2:
            raise ValueError(f"BTQ quality head must produce 2 channels, received {quality_logits.shape[1]}")
        quality_logits = quality_logits.permute(0, 2, 1).contiguous()
        mask_h, mask_w = proto.shape[-2:]
        box_scale = imgsz.new_tensor([mask_w, mask_h, mask_w, mask_h])
        box_scale = (box_scale / imgsz[[1, 0, 1, 0]]).to(target_bboxes.dtype)
        quality_losses, point_losses = [], []

        for image_idx in range(proto.shape[0]):
            positive = fg_mask[image_idx]
            if not positive.any():
                continue
            mask_indices = target_gt_idx[image_idx, positive]
            if overlap:
                target = (masks[image_idx] == (mask_indices + 1).view(-1, 1, 1)).float()
            else:
                target = masks[batch_idx.view(-1) == image_idx][mask_indices]

            mask_logits = torch.einsum(
                "in,nhw->ihw", pred_masks[image_idx, positive], proto[image_idx]
            )
            boxes = target_bboxes[image_idx, positive] * box_scale
            valid = crop_mask(torch.ones_like(mask_logits), boxes)
            quality_loss, point_loss = self._quality_and_point_loss(
                quality_logits[image_idx, positive], mask_logits, target, valid
            )
            quality_losses.append(quality_loss)
            point_losses.append(point_loss)

        if not quality_losses:
            return quality_logits.sum() * 0.0
        quality_loss = torch.stack(quality_losses).mean()
        point_loss = torch.stack(point_losses).mean()
        self.last_btq_losses = {
            "quality": float(quality_loss.detach()),
            "point": float(point_loss.detach()),
        }
        return self.quality_gain * quality_loss + self.point_gain * point_loss


class v8HARMSegmentationLoss(v8SegmentationLoss):
    """Mask-aware positive selection and instance-cascade refinement for HARM."""

    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.head = model.model[-1]
        self.maskaware_gain = float(model.args.harm_maskaware)
        self.refine_gain = float(model.args.harm_refine)
        self.boundary_gain = float(model.args.harm_boundary)
        self.topk = max(1, int(model.args.harm_topk))
        self.candidates = max(self.topk, int(model.args.harm_candidates))
        self.max_instances = max(1, int(model.args.harm_max_instances))
        self.quality_size = max(16, int(model.args.harm_quality_size))
        self.warmup_epochs = max(0, int(model.args.harm_warmup_epochs))
        self.last_harm_losses = {}

    def _warmup_factor(self):
        if self.warmup_epochs == 0:
            return 1.0
        epoch = int(getattr(self.model, "harm_epoch", self.warmup_epochs - 1))
        return min(1.0, max(0.0, (epoch + 1) / self.warmup_epochs))

    @staticmethod
    def _mask_iou(logits, target, valid):
        probability = logits.sigmoid() * valid
        target = target * valid
        intersection = (probability * target).sum((1, 2))
        union = (probability + target - probability * target).sum((1, 2))
        return (intersection + 1.0) / (union + 1.0)

    @staticmethod
    def _dice_loss(logits, target):
        probability = logits.sigmoid()
        intersection = (probability * target).sum((1, 2))
        denominator = probability.sum((1, 2)) + target.sum((1, 2))
        return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

    @staticmethod
    def _boundary_loss(logits, target):
        target = target.float().unsqueeze(1)
        dilated = F.max_pool2d(target, 3, 1, 1)
        eroded = 1.0 - F.max_pool2d(1.0 - target, 3, 1, 1)
        boundary = (dilated - eroded).squeeze(1).clamp_(0.0, 1.0)
        error = F.binary_cross_entropy_with_logits(logits, target.squeeze(1), reduction="none")
        normalizer = boundary.sum((1, 2)).clamp_min(1.0)
        return (error * boundary).sum((1, 2)).div(normalizer).mean()

    def _instance_targets(self, masks, batch_idx, image_idx, gt_indices, overlap):
        """Gather unique instance targets without a Python loop over positive anchors."""
        gt_indices = gt_indices.long()
        if overlap:
            return (masks[image_idx].unsqueeze(0) == gt_indices[:, None, None] + 1).float()
        return masks[batch_idx.view(-1) == image_idx].index_select(0, gt_indices)

    @staticmethod
    def _group_topk(scores, groups, k):
        """Return local row indices for the Top-K scores of every group in one tensor operation."""
        unique_groups, inverse = torch.unique(groups.long(), sorted=True, return_inverse=True)
        if not unique_groups.numel():
            return groups.new_empty(0), unique_groups
        score_matrix = scores.new_full((scores.numel(), unique_groups.numel()), -torch.inf)
        rows = torch.arange(scores.numel(), device=scores.device)
        score_matrix[rows, inverse] = scores
        values, selected_rows = score_matrix.topk(min(int(k), scores.numel()), dim=0)
        valid = values.isfinite()
        selected_groups = unique_groups.unsqueeze(0).expand_as(selected_rows)
        return selected_rows[valid], selected_groups[valid]

    def _quality_inputs(self, proto, coefficients, unique_targets, target_inverse, boxes):
        """Build low-resolution logits, targets, and crop masks used only for detached ranking."""
        size = self.quality_size
        if tuple(proto.shape[-2:]) != (size, size):
            proto = F.interpolate(proto.unsqueeze(0), (size, size), mode="bilinear", align_corners=False)[0]
            unique_targets = F.interpolate(
                unique_targets.unsqueeze(1), (size, size), mode="nearest"
            ).squeeze(1)
        logits = torch.einsum("in,nhw->ihw", coefficients, proto)
        valid = crop_mask(torch.ones_like(logits), boxes)
        return logits, unique_targets[target_inverse], valid

    def calculate_segmentation_loss(
        self,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Blend the standard mask loss with per-GT mask-quality Top-K assignment."""
        standard_loss = super().calculate_segmentation_loss(
            fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, overlap
        )
        blend = min(1.0, self.maskaware_gain * self._warmup_factor())
        needs_refine_selection = self.refine_gain > 0 or self.boundary_gain > 0
        self._harm_refine_cache = []
        if blend <= 0 and not needs_refine_selection:
            return standard_loss

        _, _, mask_h, mask_w = proto.shape
        normalized_boxes = target_bboxes / imgsz[[1, 0, 1, 0]]
        areas = xyxy2xywh(normalized_boxes)[..., 2:].prod(2)
        mask_boxes = normalized_boxes * proto.new_tensor([mask_w, mask_h, mask_w, mask_h])
        selected_loss = proto.sum() * 0.0
        selected_count = 0

        for image_idx in range(proto.shape[0]):
            positive = fg_mask[image_idx]
            if not positive.any():
                continue
            anchors = torch.where(positive)[0]
            gt_indices = target_gt_idx[image_idx, anchors]
            unique_gt, inverse = torch.unique(gt_indices.long(), sorted=True, return_inverse=True)
            unique_targets = self._instance_targets(masks, batch_idx, image_idx, unique_gt, overlap)
            quality_boxes = normalized_boxes[image_idx, anchors] * self.quality_size
            logits, quality_targets, valid = self._quality_inputs(
                proto[image_idx], pred_masks[image_idx, anchors], unique_targets, inverse, quality_boxes
            )
            quality = self._mask_iou(logits.detach(), quality_targets, valid)
            if needs_refine_selection:
                self._harm_refine_cache.append(
                    (image_idx, anchors, gt_indices.long(), quality, unique_gt, unique_targets)
                )
            if blend <= 0:
                continue
            selected, _ = self._group_topk(quality, gt_indices, self.topk)
            selected_anchors = anchors[selected]
            selected_loss += self.single_mask_loss(
                unique_targets[inverse[selected]],
                pred_masks[image_idx, selected_anchors],
                proto[image_idx],
                mask_boxes[image_idx, selected_anchors],
                areas[image_idx, selected_anchors],
            )
            selected_count += selected.numel()

        if blend <= 0 or selected_count == 0:
            return standard_loss
        selected_loss = selected_loss / selected_count
        return (1.0 - blend) * standard_loss + blend * selected_loss

    def calculate_auxiliary_segmentation_loss(
        self,
        detail,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        pred_bboxes,
        target_scores,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Select mask-aware positives and supervise the two-stage ROI refiner."""
        if detail is None:
            return proto.sum() * 0.0
        if self.refine_gain <= 0 and self.boundary_gain <= 0:
            return detail.sum() * 0.0

        refine_losses, boundary_losses = [], []
        selected_for_refine = []
        for image_idx, anchors, gt_indices, qualities, unique_gt, unique_targets in self._harm_refine_cache:
            assignment_quality = target_scores[image_idx].sum(-1)
            candidate_local, candidate_gt = self._group_topk(
                assignment_quality[anchors], gt_indices, self.candidates
            )
            best_local, best_gt = self._group_topk(qualities[candidate_local], candidate_gt, 1)
            best_anchors = anchors[candidate_local[best_local]]
            keep = assignment_quality[best_anchors].topk(min(self.max_instances, best_anchors.numel())).indices
            selected_for_refine.append(
                (image_idx, best_anchors[keep], best_gt[keep], unique_gt, unique_targets)
            )

        if self.refine_gain > 0 or self.boundary_gain > 0:
            image_indices, all_targets, all_boxes = [], [], []
            for image_idx, anchors, selected_gt, unique_gt, unique_targets in selected_for_refine:
                target_lookup = (selected_gt[:, None] == unique_gt[None]).long().argmax(1)
                image_indices.append(anchors.new_full((anchors.numel(),), image_idx))
                all_targets.append(unique_targets[target_lookup])
                all_boxes.append(pred_bboxes[image_idx, anchors])

            if image_indices:
                image_indices = torch.cat(image_indices)
                targets = torch.cat(all_targets)
                boxes = torch.cat(all_boxes)
                coarse = torch.cat(
                    [
                        torch.einsum("in,nhw->ihw", pred_masks[image_idx, anchors], proto[image_idx])
                        for image_idx, anchors, *_ in selected_for_refine
                    ]
                )
                refined = self.head.refine_instances(
                    detail, coarse, boxes, imgsz, image_indices=image_indices
                )
                target_roi = self.head.instance_refiner.crop_features(
                    targets.unsqueeze(1), boxes, imgsz, refined.shape[-1]
                ).squeeze(1)
                refine_losses.append(
                    F.binary_cross_entropy_with_logits(refined.squeeze(1), target_roi)
                    + self._dice_loss(refined.squeeze(1), target_roi).mean()
                )
                boundary_losses.append(self._boundary_loss(refined.squeeze(1), target_roi))

        zero = detail.sum() * 0.0
        refine_loss = torch.stack(refine_losses).mean() if refine_losses else zero
        boundary_loss = torch.stack(boundary_losses).mean() if boundary_losses else zero
        warmup = self._warmup_factor()
        total = warmup * (
            self.refine_gain * refine_loss
            + self.boundary_gain * boundary_loss
        )
        self.last_harm_losses = {
            "maskaware_blend": min(1.0, self.maskaware_gain * warmup),
            "refine": refine_loss.detach(),
            "boundary": boundary_loss.detach(),
            "warmup": warmup,
        }
        self._harm_refine_cache = []
        return total


class v8BASSegmentationLoss(v8SegmentationLoss):
    """YOLO segmentation loss with the legacy BAS boundary auxiliary objective."""

    def __init__(self, model):
        super().__init__(model)
        self.boundary_gain = float(getattr(model.args, "boundary", 0.2))

    @staticmethod
    def _boundary_target(masks, batch_idx, batch_size, size, overlap):
        """Build the union-mask boundary target used by the original BAS implementation."""
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim != 3:
            raise ValueError(f"Expected masks with shape (B,H,W) or (N,H,W), got {tuple(masks.shape)}")

        if overlap:
            if masks.shape[0] != batch_size:
                raise ValueError(f"Expected {batch_size} overlap masks, got {masks.shape[0]}")
            union = (masks > 0).float().unsqueeze(1)
        else:
            union = torch.zeros((batch_size, 1, *masks.shape[-2:]), device=masks.device, dtype=masks.dtype)
            image_indices = batch_idx.view(-1).long().to(masks.device)
            for image_idx in range(batch_size):
                image_masks = masks[image_indices == image_idx]
                if image_masks.numel():
                    union[image_idx, 0] = (image_masks > 0).any(0).float()

        if tuple(union.shape[-2:]) != tuple(size):
            union = F.interpolate(union, size=size, mode="nearest")
        dilated = F.max_pool2d(union, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool2d(1.0 - union, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp_(0.0, 1.0)

    @staticmethod
    def _boundary_aux_loss(logits, target):
        """Compute the BCE and Dice objective used to train the BAS branch."""
        if tuple(logits.shape[-2:]) != tuple(target.shape[-2:]):
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probability = logits.sigmoid()
        dims = (1, 2, 3)
        intersection = (probability * target).sum(dims)
        denominator = probability.sum(dims) + target.sum(dims)
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        return bce + dice.mean()

    def calculate_auxiliary_segmentation_loss(
        self,
        boundary_logits,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        pred_bboxes,
        target_scores,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Return the weighted BAS boundary loss through the shared auxiliary-loss hook."""
        if boundary_logits is None:
            return proto.sum() * 0.0
        if boundary_logits.shape[1] != 1:
            raise ValueError(f"BAS boundary head must produce 1 channel, received {boundary_logits.shape[1]}")
        target = self._boundary_target(
            masks, batch_idx, proto.shape[0], proto.shape[-2:], overlap
        )
        return self.boundary_gain * self._boundary_aux_loss(boundary_logits, target)


class v8BoundaryDiceSegmentationLoss(v8BASSegmentationLoss):
    """Pure Boundary Dice supervision used by ablation stage A."""

    @staticmethod
    def _boundary_aux_loss(logits, target):
        """Compute only the soft Dice term for the boundary branch."""
        if tuple(logits.shape[-2:]) != tuple(target.shape[-2:]):
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        probability = logits.sigmoid()
        dims = (1, 2, 3)
        intersection = (probability * target).sum(dims)
        denominator = probability.sum(dims) + target.sum(dims)
        return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


class v8BASOnlySegmentationLoss(v8BASSegmentationLoss):
    """Stage B supervision using only the BAS boundary BCE objective."""

    @staticmethod
    def _boundary_aux_loss(logits, target):
        """Compute only the pixel-wise BCE term for the boundary branch."""
        if tuple(logits.shape[-2:]) != tuple(target.shape[-2:]):
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return F.binary_cross_entropy_with_logits(logits, target)


class v8BASMQCSegmentationLoss(v8BTQSegmentationLoss):
    """Combine BAS boundary Dice supervision with MQC quality calibration, without BPR."""

    def __init__(self, model):
        super().__init__(model)
        self.boundary_gain = float(getattr(model.args, "boundary", 0.2))
        self.point_gain = 0.0
        self.last_bas_mqc_losses = {}

    @staticmethod
    def _boundary_aux_loss(logits, target):
        """Return the A+B boundary objective used by the complete method."""
        return v8BASSegmentationLoss._boundary_aux_loss(logits, target)

    def _quality_aux_loss(
        self,
        quality_logits,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Compute the MQC mask-IoU and boundary-F1 quality objective."""
        if quality_logits.shape[1] != 2:
            raise ValueError(f"MQC quality head must produce 2 channels, received {quality_logits.shape[1]}")

        quality_logits = quality_logits.permute(0, 2, 1).contiguous()
        mask_h, mask_w = proto.shape[-2:]
        box_scale = imgsz.new_tensor([mask_w, mask_h, mask_w, mask_h])
        box_scale = (box_scale / imgsz[[1, 0, 1, 0]]).to(target_bboxes.dtype)
        quality_losses = []
        for image_idx in range(proto.shape[0]):
            positive = fg_mask[image_idx]
            if not positive.any():
                continue
            mask_indices = target_gt_idx[image_idx, positive]
            if overlap:
                target = (masks[image_idx] == (mask_indices + 1).view(-1, 1, 1)).float()
            else:
                target = masks[batch_idx.view(-1) == image_idx][mask_indices]

            mask_logits = torch.einsum(
                "in,nhw->ihw", pred_masks[image_idx, positive], proto[image_idx]
            )
            boxes = target_bboxes[image_idx, positive] * box_scale
            valid = crop_mask(torch.ones_like(mask_logits), boxes)
            mask_iou = self._soft_iou_target(mask_logits, target, valid)
            boundary_f1 = self._boundary_f1_target(
                mask_logits, target, valid, tolerance=self.boundary_width
            )
            quality_target = torch.stack((mask_iou, boundary_f1), 1).to(quality_logits.dtype)
            quality_losses.append(
                F.binary_cross_entropy_with_logits(quality_logits[image_idx, positive], quality_target)
            )

        return torch.stack(quality_losses).mean() if quality_losses else quality_logits.sum() * 0.0

    def calculate_auxiliary_segmentation_loss(
        self,
        auxiliary_outputs,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        pred_bboxes,
        target_scores,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Return Boundary Dice and MQC losses while leaving BPR disabled."""
        if not isinstance(auxiliary_outputs, (tuple, list)) or len(auxiliary_outputs) != 2:
            raise ValueError("BAS-MQC head must return (boundary_logits, quality_logits)")
        boundary_logits, quality_logits = auxiliary_outputs
        if boundary_logits.shape[1] != 1:
            raise ValueError(f"BAS boundary head must produce 1 channel, received {boundary_logits.shape[1]}")

        boundary_target = v8BASSegmentationLoss._boundary_target(
            masks, batch_idx, proto.shape[0], proto.shape[-2:], overlap
        )
        boundary_loss = self._boundary_aux_loss(boundary_logits, boundary_target)
        quality_loss = self._quality_aux_loss(
            quality_logits,
            fg_mask,
            masks,
            target_gt_idx,
            target_bboxes,
            batch_idx,
            proto,
            pred_masks,
            imgsz,
            overlap,
        )
        self.last_bas_mqc_losses = {
            "boundary": float(boundary_loss.detach()),
            "quality": float(quality_loss.detach()),
        }
        return self.boundary_gain * boundary_loss + self.quality_gain * quality_loss


class v8MQCSegmentationLoss(v8BASMQCSegmentationLoss):
    """Stage C supervision using MQC without a boundary auxiliary objective."""

    def calculate_auxiliary_segmentation_loss(
        self,
        quality_logits,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        pred_bboxes,
        target_scores,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Return the MQC quality loss while leaving boundary supervision disabled."""
        if not torch.is_tensor(quality_logits):
            raise ValueError("MQC head must return a quality-logit tensor")
        quality_loss = self._quality_aux_loss(
            quality_logits,
            fg_mask,
            masks,
            target_gt_idx,
            target_bboxes,
            batch_idx,
            proto,
            pred_masks,
            imgsz,
            overlap,
        )
        self.last_bas_mqc_losses = {"quality": float(quality_loss.detach())}
        return self.quality_gain * quality_loss


class v8BASOnlyMQCSegmentationLoss(v8BASMQCSegmentationLoss):
    """Stage B+C supervision using BAS boundary BCE and MQC."""

    @staticmethod
    def _boundary_aux_loss(logits, target):
        return v8BASOnlySegmentationLoss._boundary_aux_loss(logits, target)


class v8BoundaryDiceMQCSegmentationLoss(v8BASMQCSegmentationLoss):
    """Stage A+C supervision using Boundary Dice and MQC."""

    @staticmethod
    def _boundary_aux_loss(logits, target):
        return v8BoundaryDiceSegmentationLoss._boundary_aux_loss(logits, target)


class v8BASBTQSegmentationLoss(v8BTQSegmentationLoss):
    """Combine BAS Boundary Dice with the complete BPR and MQC objectives."""

    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.head = model.model[-1]
        self.boundary_gain = float(getattr(model.args, "boundary", 0.1))
        self.router_balance_gain = float(getattr(model.args, "ssc_balance", 0.0))
        self.last_bas_btq_losses = {}
        self.instance_cache = []

    def __call__(self, preds, batch):
        """Clear per-forward instance state before computing the standard BAS-BTQ loss."""
        self.instance_cache = []
        return super().__call__(preds, batch)

    def _cache_instance_predictions(
        self,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        target_scores,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Cache one best positive-anchor mask per GT for cross-view instance matching."""
        mask_h, mask_w = proto.shape[-2:]
        box_scale = imgsz.new_tensor([mask_w, mask_h, mask_w, mask_h])
        box_scale = (box_scale / imgsz[[1, 0, 1, 0]]).to(target_bboxes.dtype)
        cache = []
        flat_batch_idx = batch_idx.view(-1)
        for image_idx in range(proto.shape[0]):
            positive_indices = torch.where(fg_mask[image_idx])[0]
            if positive_indices.numel() == 0:
                cache.append({"gt_idx": positive_indices.long(), "logits": proto[image_idx, :0], "valid": proto[image_idx, :0], "quality": proto.new_zeros(0)})
                continue

            gt_indices = target_gt_idx[image_idx, positive_indices]
            anchor_scores = target_scores[image_idx, positive_indices].sum(-1)
            chosen_anchors, chosen_gt = [], []
            for gt_idx in gt_indices.unique(sorted=True):
                members = torch.where(gt_indices == gt_idx)[0]
                best = members[anchor_scores[members].argmax()]
                chosen_anchors.append(positive_indices[best])
                chosen_gt.append(gt_idx)
            anchors = torch.stack(chosen_anchors)
            gt_ids = torch.stack(chosen_gt).long()
            logits = torch.einsum("in,nhw->ihw", pred_masks[image_idx, anchors], proto[image_idx])
            boxes = target_bboxes[image_idx, anchors] * box_scale
            valid = crop_mask(torch.ones_like(logits), boxes)
            if overlap:
                target = (masks[image_idx] == (gt_ids + 1).view(-1, 1, 1)).float()
            else:
                image_masks = masks[flat_batch_idx == image_idx]
                target = image_masks[gt_ids]
            quality = self._soft_iou_target(logits, target, valid)
            cache.append({"gt_idx": gt_ids, "logits": logits, "valid": valid, "quality": quality})
        self.instance_cache = cache

    @staticmethod
    def _center_scale_maps(maps, factor, mode="bilinear", fill=0.0):
        """Scale maps about the center and crop/pad back to their original canvas."""
        if maps.numel() == 0:
            return maps
        height, width = maps.shape[-2:]
        scaled_h = max(1, int(round(height * float(factor))))
        scaled_w = max(1, int(round(width * float(factor))))
        align = False if mode in {"bilinear", "bicubic"} else None
        scaled = F.interpolate(
            maps.unsqueeze(1), size=(scaled_h, scaled_w), mode=mode, align_corners=align
        ).squeeze(1)
        if scaled_h >= height:
            top = (scaled_h - height) // 2
            scaled = scaled[:, top : top + height]
        else:
            top = (height - scaled_h) // 2
            canvas = scaled.new_full((scaled.shape[0], height, scaled_w), fill)
            canvas[:, top : top + scaled_h] = scaled
            scaled = canvas
        if scaled_w >= width:
            left = (scaled_w - width) // 2
            scaled = scaled[:, :, left : left + width]
        else:
            left = (width - scaled_w) // 2
            canvas = scaled.new_full((scaled.shape[0], height, width), fill)
            canvas[:, :, left : left + scaled_w] = scaled
            scaled = canvas
        return scaled

    @classmethod
    def cross_scale_instance_loss(
        cls,
        teacher_cache,
        student_cache,
        factors,
        boundary_weight=0.5,
        min_quality=0.35,
        min_pixels=16,
    ):
        """Match GT-indexed masks across scale views and return mask/boundary consistency."""
        losses = []
        zero = factors.sum() * 0.0
        for image_idx, (teacher, student) in enumerate(zip(teacher_cache, student_cache)):
            if teacher["gt_idx"].numel() == 0 or student["gt_idx"].numel() == 0:
                continue
            student_lookup = {int(gt): idx for idx, gt in enumerate(student["gt_idx"].tolist())}
            teacher_rows, student_rows = [], []
            for row, gt_idx in enumerate(teacher["gt_idx"].tolist()):
                if gt_idx in student_lookup and float(teacher["quality"][row]) >= float(min_quality):
                    teacher_rows.append(row)
                    student_rows.append(student_lookup[gt_idx])
            if not teacher_rows:
                continue

            teacher_logits = cls._center_scale_maps(
                teacher["logits"][teacher_rows].detach(), factors[image_idx], mode="bilinear", fill=-12.0
            )
            teacher_valid = cls._center_scale_maps(
                teacher["valid"][teacher_rows].detach(), factors[image_idx], mode="nearest", fill=0.0
            )
            student_logits = student["logits"][student_rows]
            valid = (teacher_valid > 0.5) & (student["valid"][student_rows] > 0.5)
            keep = valid.flatten(1).sum(1) >= int(min_pixels)
            if not keep.any():
                continue
            teacher_prob = teacher_logits[keep].sigmoid()
            student_logits = student_logits[keep]
            valid = valid[keep].float()
            student_prob = student_logits.sigmoid()

            bce = F.binary_cross_entropy_with_logits(student_logits, teacher_prob, reduction="none")
            bce = (bce * valid).sum((1, 2)) / valid.sum((1, 2)).clamp_min(1.0)
            intersection = (teacher_prob * student_prob * valid).sum((1, 2))
            denominator = ((teacher_prob + student_prob) * valid).sum((1, 2))
            dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

            def soft_boundary(probability):
                probability = probability.unsqueeze(1)
                dilated = F.max_pool2d(probability, 3, 1, 1)
                eroded = 1.0 - F.max_pool2d(1.0 - probability, 3, 1, 1)
                return (dilated - eroded).squeeze(1).clamp(0.0, 1.0)

            teacher_boundary = soft_boundary(teacher_prob).detach() * valid
            student_boundary = soft_boundary(student_prob) * valid
            boundary_intersection = (teacher_boundary * student_boundary).sum((1, 2))
            boundary_denominator = (teacher_boundary + student_boundary).sum((1, 2))
            boundary_dice = 1.0 - (2.0 * boundary_intersection + 1.0) / (boundary_denominator + 1.0)
            losses.append((bce + dice + float(boundary_weight) * boundary_dice).mean())
        return torch.stack(losses).mean() if losses else zero

    def calculate_auxiliary_segmentation_loss(
        self,
        auxiliary_outputs,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        pred_bboxes,
        target_scores,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
        overlap,
    ):
        """Add the BAS boundary objective to the standard BPR and MQC losses."""
        if not isinstance(auxiliary_outputs, (tuple, list)) or len(auxiliary_outputs) not in {2, 3}:
            raise ValueError(
                "BAS-BTQ head must return (boundary_logits, quality_logits[, router_balance_loss])"
            )
        boundary_logits, quality_logits = auxiliary_outputs[:2]
        router_loss = auxiliary_outputs[2] if len(auxiliary_outputs) == 3 else proto.sum() * 0.0
        if boundary_logits.shape[1] != 1:
            raise ValueError(f"BAS boundary head must produce 1 channel, received {boundary_logits.shape[1]}")

        boundary_target = v8BASSegmentationLoss._boundary_target(
            masks, batch_idx, proto.shape[0], proto.shape[-2:], overlap
        )
        boundary_loss = v8BASSegmentationLoss._boundary_aux_loss(boundary_logits, boundary_target)
        btq_loss = super().calculate_auxiliary_segmentation_loss(
            quality_logits,
            fg_mask,
            masks,
            target_gt_idx,
            target_bboxes,
            pred_bboxes,
            target_scores,
            batch_idx,
            proto,
            pred_masks,
            imgsz,
            overlap,
        )
        self._cache_instance_predictions(
            fg_mask,
            masks,
            target_gt_idx,
            target_bboxes,
            target_scores,
            batch_idx,
            proto,
            pred_masks,
            imgsz,
            overlap,
        )
        self.last_bas_btq_losses = {
            "boundary": float(boundary_loss.detach()),
            "router_balance": float(router_loss.detach()),
            **self.last_btq_losses,
        }
        return self.boundary_gain * boundary_loss + btq_loss + self.router_balance_gain * router_loss


class v8FMSCSegmentationLoss(v8BASBTQSegmentationLoss):
    """A+B loss with training-only frequency/style consistency regularization."""

    def __init__(self, model):
        super().__init__(model)
        self.fmsc_weight = float(getattr(model.args, "fmsc_consistency", 0.05))
        self.fmsc_warmup_epochs = max(0, int(getattr(model.args, "fmsc_warmup_epochs", 10)))

    def __call__(self, preds, batch):
        total_loss, loss_items = super().__call__(preds, batch)
        fmsc_loss = getattr(self.head.fmsc, "last_loss", None)
        if fmsc_loss is None:
            return total_loss, loss_items
        epoch = int(getattr(self.model, "fmsc_epoch", 0))
        ramp = 1.0 if self.fmsc_warmup_epochs == 0 else min(
            1.0, max(0.0, float(epoch + 1) / self.fmsc_warmup_epochs)
        )
        weighted = self.fmsc_weight * ramp * fmsc_loss
        total_loss = total_loss + weighted * batch["img"].shape[0]
        loss_items[1] = loss_items[1] + weighted.detach()
        self.last_fmsc_loss = fmsc_loss.detach()
        return total_loss, loss_items


class v8CFASegmentationLoss(v8BASBTQSegmentationLoss):
    """A+B loss with causal-frequency feature consistency regularization."""

    def __init__(self, model):
        super().__init__(model)
        self.cfa_weight = float(getattr(model.args, "cfa_consistency", 0.04))
        self.cfa_warmup_epochs = max(0, int(getattr(model.args, "cfa_warmup_epochs", 10)))

    def __call__(self, preds, batch):
        total_loss, loss_items = super().__call__(preds, batch)
        cfa_loss = getattr(self.head.cfa, "last_loss", None)
        if cfa_loss is None:
            return total_loss, loss_items
        epoch = int(getattr(self.model, "cfa_epoch", 0))
        ramp = 1.0 if self.cfa_warmup_epochs == 0 else min(
            1.0, max(0.0, float(epoch + 1) / self.cfa_warmup_epochs)
        )
        weighted = self.cfa_weight * ramp * cfa_loss
        total_loss = total_loss + weighted * batch["img"].shape[0]
        loss_items[1] = loss_items[1] + weighted.detach()
        self.last_cfa_loss = cfa_loss.detach()
        return total_loss, loss_items


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        if isinstance(self.assigner, ATSSAssigner):
            anchors, _, n_anchors_list, _ = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
            target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(anchors, n_anchors_list, gt_labels, gt_bboxes, mask_gt, pred_bboxes.detach() * stride_tensor)
        else:
            target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        if isinstance(self.bce, (nn.BCEWithLogitsLoss, FocalLoss_YOLO)):
            loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        elif isinstance(self.bce, VarifocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                cls_iou_targets.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * cls_iou_targets
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[3] = self.bce(pred_scores, cls_iou_targets.to(dtype)).sum() / max(fg_mask.sum(), 1)  # BCE
        elif isinstance(self.bce, QualityfocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                targets_onehot = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * targets_onehot
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                targets_onehot_pos = torch.where(fg_scores_mask > 0, targets_onehot, 0)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot_pos = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[3] = self.bce(pred_scores, cls_iou_targets.to(dtype), targets_onehot_pos.to(torch.bool)).sum() / max(fg_mask.sum(), 1)

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0)
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )
        
        if isinstance(self.bce, (EMASlideLoss, SlideLoss)):
            if fg_mask.sum():
                auto_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).mean()
            else:
                auto_iou = -1
            loss[3] = self.bce(pred_scores, target_scores.to(dtype), auto_iou).sum() / target_scores_sum  # BCE

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        # MoE auxiliary loss
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     loss[1] += m.aux_loss * 1.5


        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()
        
        # Cls loss
        if isinstance(self.bce, (nn.BCEWithLogitsLoss, FocalLoss_YOLO)):
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        elif isinstance(self.bce, VarifocalLoss_YOLO):
            if fg_mask.sum():
                b, n, c = pred_bboxes.size()
                pos_ious = probiou(pred_bboxes.reshape((b * n, c)), target_bboxes.reshape((b * n, c))).clamp(min=1e-6).detach().reshape((b, n, 1))
                # 10.0x Faster than torch.one_hot
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                cls_iou_targets.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * cls_iou_targets
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype)).sum() / max(fg_mask.sum(), 1)  # BCE
        elif isinstance(self.bce, QualityfocalLoss_YOLO):
            if fg_mask.sum():
                b, n, c = pred_bboxes.size()
                pos_ious = probiou(pred_bboxes.reshape((b * n, c)), target_bboxes.reshape((b * n, c))).clamp(min=1e-6).detach().reshape((b, n, 1))
                # 10.0x Faster than torch.one_hot
                targets_onehot = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * targets_onehot
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                targets_onehot_pos = torch.where(fg_scores_mask > 0, targets_onehot, 0)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
                targets_onehot_pos = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                        dtype=torch.int64,
                                        device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype), targets_onehot_pos.to(torch.bool)).sum() / max(fg_mask.sum(), 1)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # MoE auxiliary loss
        if hasattr(self, 'model'):
             for m in self.model.modules():
                 if hasattr(m, 'aux_loss'):
                     loss[1] += m.aux_loss * 1.5


        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]

class E2EFGIDetectLoss:
    def __init__(self, model):
        self.one2ten = v8DetectionLoss(model, tal_topk=10)
        self.one2five = v8DetectionLoss(model, tal_topk=5)
        self.one2one = v8DetectionLoss(model, tal_topk=1)
    
    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2ten = preds['output_0']
        loss_one2ten = self.one2ten(one2ten, batch)
        one2five = preds['output_1']
        loss_one2five = self.one2five(one2five, batch)
        one2one = preds['output_2']
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2ten[0] + loss_one2five[0] + loss_one2one[0], loss_one2ten[1] + loss_one2five[1] + loss_one2one[1]
