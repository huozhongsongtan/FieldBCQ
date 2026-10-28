"""Independent geometry-only sampling baselines, never a DP/LLM fallback."""
import math
from shapely.geometry import Polygon, MultiPolygon


def sample_ring(coordinates, method, parameter):
    """Open vertex list; first/last observed vertices anchor the closing edge.

    Input closure is stripped. The final Polygon supplies closure; an explicit
    duplicate closure is never included in vertex counts. Tiny rings stay raw.
    """
    points=[tuple(map(float,p)) for p in coordinates]
    if len(points)>1 and points[0]==points[-1]:points.pop()
    if len(points)<3:raise ValueError('A ring requires at least three vertices')
    if any(len(p)!=2 or not all(math.isfinite(x) for x in p) for p in points):
        raise ValueError('Finite planar coordinates required')
    if method=='nth_point':
        if type(parameter) is not int or parameter<1:raise ValueError('Positive integer step required')
        selected=points[::parameter]
    elif method=='radial_distance':
        if isinstance(parameter,bool) or not isinstance(parameter,(int,float)) or not math.isfinite(parameter) or parameter<=0:
            raise ValueError('Positive finite radius required')
        selected=[points[0]];threshold=parameter**2
        for p in points[1:]:
            if (p[0]-selected[-1][0])**2+(p[1]-selected[-1][1])**2>threshold:
                selected.append(p)
    else:raise ValueError('Unknown sampling method')
    if selected[-1]!=points[-1]:selected.append(points[-1])
    if len(set(selected))<3:return points,'too_few_ring_vertices'
    return selected,None


def simplify_items(items, method, parameter):
    """Preserve identities, all components and holes; reject invalid parcels.

    A valid sampled ring is used only if its complete parcel stays valid. An
    invalid parcel reverts in full to its original geometry. No make_valid,
    hidden DP, image, label, smoothing or shape-specific parameter is used.
    """
    output=[];events=[]
    for item in items:
        raw=item.geometry
        if raw.geom_type not in ('Polygon','MultiPolygon') or raw.is_empty or not raw.is_valid:
            raise ValueError('Valid nonempty polygon input required: '+item.identifier)
        parts=list(raw.geoms) if raw.geom_type=='MultiPolygon' else [raw]
        sampled=[];ring_fallbacks=0
        for part in parts:
            outer,reason=sample_ring(part.exterior.coords,method,parameter);ring_fallbacks+=bool(reason)
            holes=[]
            for hole in part.interiors:
                selected,reason=sample_ring(hole.coords,method,parameter)
                ring_fallbacks+=bool(reason);holes.append(selected)
            sampled.append(Polygon(outer,holes))
        candidate=MultiPolygon(sampled) if raw.geom_type=='MultiPolygon' else sampled[0]
        reverted=candidate.is_empty or not candidate.is_valid or candidate.area<=0
        if reverted:candidate=raw
        output.append(item.with_geometry(candidate))
        events.append(dict(instance_id=item.identifier,ring_count_fallbacks=ring_fallbacks,
            parcel_reverted=bool(reverted),reason='sampled_parcel_invalid' if reverted else None))
    return output,dict(method=method,parameter=parameter,events=events,
        parcel_reverted=sum(e['parcel_reverted'] for e in events),
        rings_reverted_for_vertex_count=sum(e['ring_count_fallbacks'] for e in events),
        runtime_GT_access=False,image_access=False,model_calls=0)
