"""Versioned boundary candidate tools. No GT, planner, commits or network access.

Pixel-cell coordinates: pixel centres are (column+.5, row+.5). Image support is
an oriented-gradient proxy, not evidence of independently verified correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
from cvr_agent import polygons
from fieldbqc_selective_repair import regularize_geometry

VERSION = 'boundary-tools-v1.2'


@dataclass(frozen=True)
class ToolSpec:
    task: str
    description: str
    methods: tuple
    # name -> (default, minimum, maximum), numeric and finite only
    parameters: dict
    joint: bool = False


TOL = {'tolerance': (1., .05, 2.)}
IMAGE = {'radius': (3., .5, 4.), 'step': (.5, .25, 1.),
         'simplify': (.35, .05, 1.), 'movement_penalty': (.015, 0., .2)}
REGISTRY = {
    'contour_dp': ToolSpec('simplify', 'Distance-bounded DP baseline',
        ('shapely.simplify(preserve_topology=True)',), TOL),
    'contour_shared_dp': ToolSpec('simplify', 'Simplify free boundary stretches while retaining endpoints and exact segments of existing shared edges; no shared-edge movement',
        ('shapely.simplify(preserve_topology=True)', 'exact_shared_segment_endpoint_preservation'), TOL),
    'contour_vw': ToolSpec('simplify', 'Topology-preserving coverage VW',
        ('coverage_is_valid', 'coverage_simplify'), TOL, True),
    'contour_lines': ToolSpec('simplify', 'Ordered segmentation, TLS and corner intersections',
        ('regularize_geometry', 'regularize_ring'), TOL),
    'contour_collinear': ToolSpec('simplify', 'Remove low-turn redundant points',
        ('bounded_turn_and_chord_distance_deletion',), {**TOL, 'angle': (12., 1., 30.)}),
    'contour_short_edges': ToolSpec('simplify', 'Collapse short-edge vertices with chord protection',
        ('short_edge_priority_deletion', 'point_segment_distance'),
        {**TOL, 'length': (1.5, .1, 4.)}),
    'contour_corridor': ToolSpec('simplify', 'Greedy chord corridor with two fixed cyclic anchors',
        ('cyclic_anchor_split', 'greedy_chord_corridor'), TOL),
    'boundary_line_realign': ToolSpec('realign', 'Image-supported straight-edge offset and rotation',
        ('DP_arc_segmentation', 'oriented_gradient_line_search', 'bounded_line_intersections'),
        {**IMAGE, 'segmentation': (1., .25, 2.), 'rotation': (2., 0., 5.)}),
    'boundary_normal_snap': ToolSpec('realign', 'Independent normal attraction; texture-sensitive control',
        ('uniform_ring_sampling', 'normal_offset_search', 'bounded_post_simplify'), IMAGE),
    'boundary_path_realign': ToolSpec('realign', 'Smooth cyclic normal-offset path via dynamic programming',
        ('uniform_ring_sampling', 'cyclic_viterbi_offset_path', 'bounded_post_simplify'),
        {**IMAGE, 'smoothness': (.12, .01, 1.)}),
    'repair_geometry': ToolSpec('legalize', 'Explicit polygonal make_valid candidate',
        ('shapely.make_valid', 'polygonal_component_extraction'), {}),
}


def catalog():
    """JSON-safe discovery contract usable by rule or model controllers."""
    return {name: dict(task=s.task, description=s.description, methods=list(s.methods),
        joint=s.joint, parameters={k: dict(default=v[0], minimum=v[1], maximum=v[2])
                                  for k, v in s.parameters.items()}) for name, s in REGISTRY.items()}


def parameters(tool, values=None):
    if tool not in REGISTRY:
        raise ValueError('Unknown boundary tool')
    values = {} if values is None else values
    if not isinstance(values, dict) or set(values) - set(REGISTRY[tool].parameters):
        raise ValueError('Unknown or non-object parameters')
    result = {}
    for name, (default, low, high) in REGISTRY[tool].parameters.items():
        value = values.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid parameter: ' + name)
        result[name] = float(value)
    return result


def clean_ring(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Invalid ring coordinates')
    if len(xy) > 1 and np.linalg.norm(xy[0]-xy[-1]) < 1e-9:
        xy = xy[:-1]
    xy = xy[np.linalg.norm(xy-np.roll(xy, 1, axis=0), axis=1) > 1e-9]
    if len(xy) < 3:
        raise ValueError('Ring collapsed')
    return xy


def shared_dp_ring(xy, tolerance, shared):
    coords=clean_ring(xy)
    lookup={tuple(point):index for index,point in enumerate(coords)}
    simplified=Polygon(coords).simplify(tolerance,preserve_topology=True)
    keep={lookup[tuple(point)] for point in list(simplified.exterior.coords)[:-1]}
    if shared is not None:
        for index,point in enumerate(coords):
            segment=LineString([point,coords[(index+1)%len(coords)]])
            if segment.intersection(shared).length>1e-7:
                keep.update([index,(index+1)%len(coords)])
    return coords[sorted(keep)] if len(keep)>=3 else coords


def point_segment(points, a, b):
    v = b-a
    t = np.clip((np.asarray(points)-a) @ v / max(float(v@v), 1e-12), 0., 1.)
    return np.linalg.norm(np.asarray(points)-(a + np.asarray(t)[..., None]*v), axis=-1)


def delete_ring(xy, tolerance, angle=None, length=None):
    xy = list(clean_ring(xy))
    # Full-scene and cumulative error are checked separately by the executor.
    while len(xy) > 3:
        p = np.asarray(xy); a = np.roll(p, 1, axis=0); c = np.roll(p, -1, axis=0)
        u, v = p-a, c-p
        la, lb = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        turns = np.degrees(np.arccos(np.clip(np.sum(u*v, axis=1)/np.maximum(la*lb, 1e-12), -1, 1)))
        residual = np.array([point_segment(q, x, y) for q, x, y in zip(p, a, c)])
        ok = residual <= tolerance
        if angle is not None:
            ok &= turns <= angle
        if length is not None:
            ok &= np.minimum(la, lb) <= length
        indices = np.flatnonzero(ok)
        if not len(indices):
            break
        i = min(indices, key=lambda j: (residual[j], min(la[j], lb[j]), int(j)))
        xy.pop(int(i))
    return np.asarray(xy)


def corridor_ring(xy, tolerance):
    xy = clean_ring(xy)
    first = int(np.lexsort((xy[:, 1], xy[:, 0]))[0])
    xy = np.roll(xy, -first, axis=0)
    middle = int(np.argmax(np.linalg.norm(xy-xy[0], axis=1)))

    def chain(p):
        output = [p[0]]; start = 0
        while start < len(p)-1:
            end = start+1
            while end+1 < len(p) and np.max(point_segment(p[start:end+2], p[start], p[end+1])) <= tolerance:
                end += 1
            output.append(p[end]); start = end
        return output

    result = chain(xy[:middle+1])[:-1] + chain(np.vstack((xy[middle:], xy[:1])))[:-1]
    return clean_ring(result) if len(result) >= 3 else xy


def map_rings(geometry, fn, holes=True):
    result = [Polygon(fn(np.asarray(p.exterior.coords)),
                      [fn(np.asarray(r.coords)) if holes else r.coords for r in p.interiors])
              for p in polygons(geometry)]
    if not result:
        raise ValueError('Expected polygonal geometry')
    return result[0] if len(result) == 1 else MultiPolygon(result)


def sample_ring(xy, spacing=1., maximum=1024):
    p = clean_ring(xy); closed = np.vstack((p, p[:1]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    count = max(8, min(maximum, int(math.ceil(arc[-1]/spacing))))
    t = np.linspace(0., arc[-1], count, endpoint=False)
    return np.column_stack([np.interp(t, arc, closed[:, k]) for k in range(2)])


def normals(p):
    tangent = np.roll(p, -2, axis=0)-np.roll(p, 2, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    return np.column_stack((-tangent[:, 1], tangent[:, 0]))


def cyclic_offsets(cost, offsets, smoothness):
    """Exact finite-state cyclic path; all seam starting labels are considered."""
    n, width = cost.shape
    transition = smoothness*(offsets[:, None]-offsets[None, :])**2
    best_cost = float('inf'); best = None
    for start in range(width):
        scores = np.full(width, np.inf); scores[start] = cost[0, start]
        parent = np.zeros((n, width), dtype=np.int16)
        for i in range(1, n):
            matrix = scores[:, None]+transition
            parent[i] = np.argmin(matrix, axis=0)
            scores = cost[i]+np.min(matrix, axis=0)
        last = int(np.argmin(scores+transition[:, start]))
        value = float(scores[last]+transition[last, start])
        if value < best_cost:
            result = np.empty(n, dtype=int); result[-1] = last
            for i in range(n-1, 0, -1):
                result[i-1] = parent[i, result[i]]
            best_cost, best = value, result
    return best


class BoundaryToolbox:
    def __init__(self, image):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected BGR image')
        self.shape = image.shape[:2]
        gray = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.
        gray = cv2.GaussianBlur(gray, (0, 0), .8)
        self.gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)/8
        self.gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)/8

    def support(self, points, direction):
        shape = np.asarray(points).shape[:-1]
        p = np.asarray(points).reshape(-1, 2)
        # OpenCV remap dimensions must be below 32767, even when flattening a
        # valid N x offset-label candidate array. Chunk independently of shape.
        def interpolate(field):
            chunks = []
            for start in range(0, len(p), 16000):
                q = p[start:start+16000]
                x = (q[:, 0]-.5).astype(np.float32).reshape(1, -1)
                y = (q[:, 1]-.5).astype(np.float32).reshape(1, -1)
                chunks.append(cv2.remap(field, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).ravel())
            return np.concatenate(chunks).reshape(shape)
        gx, gy = interpolate(self.gx), interpolate(self.gy)
        direction = np.asarray(direction)
        return np.clip(4*np.abs(gx*direction[..., 0]+gy*direction[..., 1]), 0., 1.)

    def evidence(self, geometry):
        total = weight = 0.
        for part in polygons(geometry):
            # Correction currently edits exterior rings only. Holes remain fixed.
            p = sample_ring(part.exterior.coords)
            total += float(self.support(p, normals(p)).mean())*part.exterior.length
            weight += part.exterior.length
        return total/max(weight, 1e-9)

    def _offset_ring(self, xy, cfg, smooth):
        p = sample_ring(xy); normal = normals(p)
        offsets = np.unique(np.r_[np.arange(-cfg['radius'], cfg['radius']+1e-8, cfg['step']), 0.])
        choices = p[:, None, :]+normal[:, None, :]*offsets[None, :, None]
        cost = -self.support(choices, normal[:, None, :])+cfg['movement_penalty']*offsets[None, :]**2
        labels = cyclic_offsets(cost, offsets, cfg['smoothness']) if smooth else np.argmin(cost, axis=1)
        moved = p+normal*offsets[labels, None]
        # Do not make_valid or conceal an invalid candidate.
        approximate = cv2.approxPolyDP(moved.astype(np.float32), cfg['simplify'], True).reshape(-1, 2)
        if len(approximate) < 3:
            self.warnings.append('offset_simplification_structural_floor: ring retained')
            return clean_ring(xy)
        return clean_ring(approximate)

    def _line_ring(self, xy, cfg):
        original = clean_ring(xy)
        approximate = cv2.approxPolyDP(original.astype(np.float32), cfg['segmentation'], True).reshape(-1, 2)
        if len(approximate) < 3:
            self.warnings.append('line_segmentation_structural_floor: ring retained')
            return original
        anchors = clean_ring(approximate)
        lines = []
        offsets = np.unique(np.r_[np.arange(-cfg['radius'], cfg['radius']+1e-8, cfg['step']), 0.])
        for a, b in zip(anchors, np.roll(anchors, -1, axis=0)):
            center = (a+b)/2; length = float(np.linalg.norm(b-a)); base = (b-a)/max(length, 1e-9)
            best = (-float('inf'), center, base)
            positions = np.linspace(-.4, .4, max(5, int(length)))*length
            for angle in sorted(set([0., -cfg['rotation'], cfg['rotation']])):
                rad = math.radians(angle); c, s = math.cos(rad), math.sin(rad)
                direction = np.array([c*base[0]-s*base[1], s*base[0]+c*base[1]])
                normal = np.array([-direction[1], direction[0]])
                for offset in sorted(offsets, key=lambda x: (abs(x), x)):
                    if length < 6. and (offset != 0 or angle != 0):
                        continue
                    origin = center+normal*offset
                    support = float(self.support(origin+positions[:, None]*direction, normal).mean())
                    score = support-cfg['movement_penalty']*(offset**2+(length*rad/2)**2)
                    if score > best[0]+1e-12:
                        best = score, origin, direction
            lines.append(best[1:])
        corners = []
        for i, anchor in enumerate(anchors):
            c0, d0 = lines[i-1]; c1, d1 = lines[i]
            determinant = float(d0[0]*d1[1]-d0[1]*d1[0]); corner = anchor
            if abs(determinant) > math.sin(math.radians(5)):
                delta = c1-c0
                distance = (delta[0]*d1[1]-delta[1]*d1[0])/determinant
                point = c0+distance*d0
                if np.linalg.norm(point-anchor) <= cfg['radius']+1e-8:
                    corner = point
            corners.append(corner)
        return clean_ring(corners)

    def propose(self, tool, items, ids, values=None):
        cfg = parameters(tool, values); spec = REGISTRY[tool]
        self.warnings = []
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Expected distinct target IDs')
        old = {p.identifier: p for p in items}
        if any(uid not in old for uid in ids):
            raise ValueError('Unknown target ID')
        if len(ids) > 1 and not spec.joint:
            raise ValueError('Tool requires one target; shared movement needs a joint constructor')
        geoms = [old[uid].geometry for uid in ids]
        if spec.task != 'legalize' and any(g.is_empty or not g.is_valid for g in geoms):
            raise ValueError('Legalize invalid inputs first')
        if tool == 'contour_vw':
            array = np.asarray(geoms, dtype=object)
            if not shapely.coverage_is_valid(array):
                raise ValueError('VW requires a nonoverlapping coverage with matching edges')
            result = list(shapely.coverage_simplify(array, cfg['tolerance']))
        elif tool == 'contour_dp':
            result = [geoms[0].simplify(cfg['tolerance'], preserve_topology=True)]
        elif tool == 'contour_shared_dp':
            shared_parts=[geoms[0].boundary.intersection(other.geometry.boundary)
                for other in items if other.identifier != ids[0]]
            shared_parts=[shared for shared in shared_parts if shared.length>1e-7]
            shared=unary_union(shared_parts) if shared_parts else None
            result=[map_rings(geoms[0],lambda ring:shared_dp_ring(ring,cfg['tolerance'],shared))]
        elif tool == 'contour_lines':
            result = [regularize_geometry(geoms[0], cfg['tolerance'])]
        elif tool == 'repair_geometry':
            result = [unary_union(polygons(shapely.make_valid(geoms[0])))]
        elif tool in ('contour_collinear', 'contour_short_edges'):
            extra = dict(angle=cfg['angle']) if tool == 'contour_collinear' else dict(length=cfg['length'])
            result = [map_rings(geoms[0], lambda p: delete_ring(p, cfg['tolerance'], **extra))]
        elif tool == 'contour_corridor':
            result = [map_rings(geoms[0], lambda p: corridor_ring(p, cfg['tolerance']))]
        elif tool == 'boundary_line_realign':
            result = [map_rings(geoms[0], lambda p: self._line_ring(p, cfg), holes=False)]
        else:
            result = [map_rings(geoms[0], lambda p: self._offset_ring(p, cfg, tool == 'boundary_path_realign'), holes=False)]
        replacements = dict(zip(ids, result))
        candidate = [p.with_geometry(replacements[p.identifier]) if p.identifier in replacements else p for p in items]
        return candidate, dict(tool_version=VERSION, tool=tool, task=spec.task, parameters=cfg,
                              methods=list(spec.methods), target_ids=list(ids), warnings=list(self.warnings),
                              structural_floor='Retain original ring if compression would yield fewer than three vertices')
