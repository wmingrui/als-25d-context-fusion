from pathlib import Path
import os,json,hashlib,shutil,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
Q=Path(__file__).resolve().parent;ROOT=Path('/root/autodl-tmp/whu25d')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def process(item):
 sev,sp,p=item;dest=Q/'data/point_cache'/sev/sp/p.stem;dest.mkdir(parents=True,exist_ok=False)
 with np.load(p) as z:
  arrays={'xyz':z['points'].astype(np.float32),'local':z['points_local'].astype(np.float32),'labels':z['labels'].astype(np.int16),'intensity':z['features'].astype(np.float32)}
  assert arrays['labels'].min()>=0 and arrays['labels'].max()<8
  assert all(np.isfinite(a).all() for a in arrays.values())
  for k,a in arrays.items():np.save(dest/(k+'.npy'),a)
  counts=np.bincount(arrays['labels'].ravel(),minlength=8)
 return sev,sp,counts.tolist()
if __name__=='__main__':
 assert not (Q/'protocol.json').exists()
 sp=json.loads((ROOT/'data/full_blocks_b50/split_info.json').read_text())
 assert not set(sp['train'])&set(sp['val']);assert not (set(sp['train'])|set(sp['val']))&set(sp['test'])
 unet=ROOT/'outputs/restoration_unet_v3_medium/best.pt'
 p={'root':str(ROOT),'class_names':['other','ground','buildings','street_trees','tree_clusters','low_veg','lights','wire'],'split':sp,'unet_checkpoint':str(unet),'unet_sha256':sha(unet),'epochs':100,'seeds':[43,44],'methods':['base','observation','feature','random_feature'],'train_severity':'medium','test_severities':['medium','mild','severe'],'evaluation_sampling_seed':42,'class_protocol':'8-class training and primary selection; secondary 7-class score excludes other GT rows','features':'frozen U-Net dec1 48 channels, learned Linear48to9; random control seed42 matched architecture; observation delta=observed maps','fusion':'input plus stage1; quality modulation and reliability descriptors disabled','precision':'BF16 train, FP32 validation/test','fixed_unseen_degradation':'medium U-Net and segmenter unchanged for every severity','evaluation_extent':'8192 points per block; not full-cloud benchmark','source_hashes':{x.name:sha(x) for x in Q.glob('*.py')}}
 (Q/'protocol.json').write_text(json.dumps(p,indent=2));shutil.copy2(ROOT/'data/shared_25d_norm_v3.json',Q/'normalization.json')
 tasks=[(sev,split,f) for sev in ['medium','mild','severe'] for split in (['train','val','test'] if sev=='medium' else ['test']) for f in sorted((ROOT/'data'/f'degraded_blocks_b50_v3_{sev}'/split).glob('*.npz'))]
 counts={};start=time.time()
 with ProcessPoolExecutor(max_workers=4) as pool:
  for i,f in enumerate(as_completed([pool.submit(process,t) for t in tasks]),1):
   sev,split,c=f.result();key=sev+'/'+split;counts.setdefault(key,np.zeros(8,dtype=np.int64));counts[key]+=c
   if i%50==0:print(i,len(tasks),round(time.time()-start),flush=True)
 assert all(c[1:].sum()>0 for c in counts.values());assert (counts['medium/train']>0).all()
 (Q/'train_class_counts.json').write_text(json.dumps(counts['medium/train'].tolist()))
 (Q/'cache_complete.json').write_text(json.dumps({'state':'complete','files':len(tasks),'seconds':time.time()-start,'counts':{k:v.tolist() for k,v in counts.items()}},indent=2))
