"""Reproducible validation sampling and the user's improved-model inference."""
from pathlib import Path
import random
import time
import sys
import numpy as np
from fieldagent.io import load,save,digest,now

EXTENSIONS={'.png','.jpg','.jpeg','.tif','.tiff'}


def select(dataset,out,count,seed,weights,model_source,exclude_manifest=None):
    dataset=Path(dataset).resolve();out=Path(out).resolve()
    if count<=0: raise ValueError('count must be positive')
    excluded_scenes=set();excluded_hashes=set()
    if exclude_manifest:
        doc=load(exclude_manifest)
        entries=doc.get('inputs',doc.get('entries',[]))
        excluded_scenes={e['scene'].casefold() for e in entries}
        excluded_hashes={e['image_sha256'] for e in entries if e.get('image_sha256')}
    rows=[];excluded=[]
    for p in sorted((dataset/'images/val').iterdir()):
        if p.suffix.lower() not in EXTENSIONS: continue
        sha=digest(p)
        if p.stem.casefold() in excluded_scenes or sha in excluded_hashes:
            excluded.append(dict(scene=p.stem,reason='overlaps_excluded_test_by_name_or_content'));continue
        label=dataset/'labels/val'/(p.stem+'.txt')
        if not label.is_file(): raise FileNotFoundError(label)
        rows.append(dict(scene=p.stem,domain=p.stem.split('_')[0],image=str(p),image_sha256=sha,
            label=str(label),label_sha256=digest(label)))
    if len({e['scene'].casefold() for e in rows})!=len(rows): raise ValueError('Duplicate image stems')
    if len(rows)<count: raise ValueError('Insufficient validation images after exclusions')
    groups={d:[e for e in rows if e['domain']==d] for d in sorted({e['domain'] for e in rows})}
    quotas={d:int(count*len(g)/len(rows)) for d,g in groups.items()}
    remainder=sorted(groups,key=lambda d:(-(count*len(groups[d])/len(rows)-quotas[d]),d))
    for d in remainder[:count-sum(quotas.values())]: quotas[d]+=1
    rng=random.Random(seed);chosen=[]
    for d,g in groups.items(): chosen.extend(rng.sample(g,quotas[d]))
    chosen.sort(key=lambda e:e['scene'])
    settings=dict(imgsz=640,conf=.25,iou=.7,max_det=710,retina_masks=True,device='cpu',torch_threads=4)
    record=dict(version='application-validation-data-v1',split='val',dataset=str(dataset),seed=seed,
        count=count,available_by_source={d:len(g) for d,g in groups.items()},selected_by_source=quotas,
        weights=str(Path(weights).resolve()),weights_sha256=digest(weights),
        model_source=str(Path(model_source).resolve()),inference=settings,entries=chosen,
        excluded=excluded,exclude_manifest_sha256=digest(exclude_manifest) if exclude_manifest else None,
        GT_offline_only=True,selection_uses_GT_quality=False,
        independence_note='Validation development sample; source-level/spatial independence is not established')
    out.mkdir(parents=True,exist_ok=True);path=out/'selection.json'
    if path.exists():
        if load(path)!=record: raise ValueError('Frozen validation selection changed; use new data output')
    else: save(path,record)
    return record


def predict(record,out):
    from fieldagent import activate_engine
    activate_engine()
    import torch
    import cv2
    from fieldbqc_agent import parcels_from_ultralytics
    from fieldbqc_geometry import from_parcels,geometry_mask,vertices
    from cvr_agent import save_items
    out=Path(out).resolve()
    if digest(record['weights'])!=record['weights_sha256']: raise ValueError('Weights changed')
    source=record['model_source'];sys.path.insert(0,source)
    from ultralytics import YOLO
    import ultralytics
    if not Path(ultralytics.__file__).resolve().is_relative_to(Path(source).resolve()):
        raise ValueError('Wrong Ultralytics source; custom segmentation layers require the configured source')
    cv2.setNumThreads(1);torch.set_num_threads(record['inference']['torch_threads'])
    model=None;rows=[]
    for number,entry in enumerate(record['entries'],1):
        if digest(entry['image'])!=entry['image_sha256']: raise ValueError('Validation image changed')
        folder=out/'predictions'/entry['scene'];done=folder/'completed.json';vector=folder/'prediction.pixel.geojson'
        if done.exists():
            old=load(done)
            if old['image_sha256']!=entry['image_sha256'] or old['vector_sha256']!=digest(vector):
                raise ValueError('Prediction cache changed')
        else:
            if model is None: model=YOLO(record['weights'],task='segment')
            began=time.perf_counter();settings={k:v for k,v in record['inference'].items() if k!='torch_threads'}
            result=model.predict(source=entry['image'],save=False,verbose=False,**settings)[0]
            parcels=parcels_from_ultralytics(result);items=from_parcels(parcels)
            if len(parcels)!=len(items) or not all(np.array_equal(p.mask,geometry_mask(g.geometry,result.orig_shape)) for p,g in zip(parcels,items)):
                raise ValueError('Prediction vector/mask roundtrip failed')
            save_items(vector,items)
            old=dict(scene=entry['scene'],image_sha256=entry['image_sha256'],vector_sha256=digest(vector),
                instances=len(items),vertices=sum(vertices(p.geometry) for p in items),
                inference_seconds=time.perf_counter()-began,mask_roundtrip=True)
            save(done,old)
        rows.append(dict(entry,prediction_vector=str(vector),prediction_vector_sha256=digest(vector)))
        save(out/'progress.json',dict(completed=number,expected=len(record['entries']),last_scene=entry['scene'],updated_utc=now()))
        print(f'VAL PREDICT {number}/{len(record["entries"])} {entry["scene"]} instances={old["instances"]}',flush=True)
    document=dict(coordinate_space='image_pixels',split='val',selection_sha256=digest(out/'selection.json'),
        labels_offline_only=True,development_only=True,inputs=rows)
    if (out/'inputs.json').exists() and load(out/'inputs.json')!=document: raise ValueError('Input manifest changed')
    save(out/'inputs.json',document)
    return document
