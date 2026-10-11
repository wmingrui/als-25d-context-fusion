from pathlib import Path
import json,shutil,hashlib,numpy as np
Q=Path(__file__).resolve().parent;OLD=Q.with_name('run_v1');D=Q/'data'
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
assert not (Q/'protocol.json').exists()
p=json.loads((OLD/'protocol.json').read_text());p.update(root=str(Q),unet_checkpoint=str(Q/'unet/best.pt'),unet_sha256=None,revision='v2 corrected reflectance, independent repeat sampling, matched optimizer updates',samples_per_epoch=848,updates_per_epoch=53,total_updates=5300,checkpoint_selection='validation mIoU only; no test-guided selection',class_weighting='inverse sqrt expected class frequency under uniform-block sampling; mean normalized and clipped .5..8',metrics=['per_class_F1','mF1','OA','mIoU'],source_hashes={f.name:hashlib.file_digest(f.open('rb'),'sha256').hexdigest() for f in Q.glob('*.py')})
write(Q/'protocol.json',p)
shutil.copy2(OLD/'normalization.json',Q/'normalization.json');shutil.copy2(OLD/'reflectance_stats.json',Q/'reflectance_stats.json')
audit={'repair':'Restore unit-scale reflectance by undoing one erroneous division by65535. Geometry, labels and degradation realization unchanged. Shared normalization fits geometric channels only; stats reused.','maps':0,'cache_blocks':0}
for src in (OLD/'data').glob('maps_25d*_shared_v3/*/*.npz'):
 dest=D/src.relative_to(OLD/'data');dest.parent.mkdir(parents=True,exist_ok=True)
 with np.load(src) as z:payload={k:z[k] for k in z.files}
 for k in ['maps_raw','maps_norm']:
  payload[k]=payload[k].copy();payload[k][6]*=65535.
  assert np.isfinite(payload[k]).all();assert payload[k][6].min()>=-1e-6 and payload[k][6].max()<=1.00001
 payload['reflectance_correction']=np.array('v2 undo duplicate /65535');np.savez_compressed(dest,**payload);audit['maps']+=1
counts=np.zeros(11,float)
for src in (OLD/'data/point_cache').glob('*/*/*'):
 if not src.is_dir():continue
 dest=D/'point_cache'/src.relative_to(OLD/'data/point_cache');dest.mkdir(parents=True,exist_ok=True)
 for k in ['xyz','local','labels']:(dest/(k+'.npy')).symlink_to(src/(k+'.npy'))
 a=np.load(src/'intensity.npy')*65535.;assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1.00001;np.save(dest/'intensity.npy',a);audit['cache_blocks']+=1
 if src.parent.name=='train':
  y=np.load(src/'labels.npy');counts+=np.bincount(y,minlength=11)/len(y)
 assert np.array_equal(np.load(dest/'labels.npy'),np.load(src/'labels.npy'))
(D/'maps_clean').symlink_to(D/'maps_25d_b50_r050_shared_v3',target_is_directory=True);(D/'maps_medium').symlink_to(D/'maps_25d_b50_r050_degraded_v3_medium_shared_v3',target_is_directory=True)
write(Q/'train_class_counts.json',(counts/counts.sum()*100000000).tolist());write(Q/'repair_audit.json',audit);shutil.copy2(OLD/'cache_complete.json',Q/'cache_complete.json')
print(audit)
