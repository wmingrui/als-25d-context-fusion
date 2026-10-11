"""D1/D3 evaluation only: frozen weights, inherited medium fixed/full protocols."""
import sys,json,hashlib,time,inspect
from pathlib import Path
import numpy as np,torch
RUN=Path('/root/autodl-tmp/cg_upgrade_20260914_run03')
OLD=Path('/root/autodl-tmp/cg_evaluation_20260914_01')
OUT=Path('/root/autodl-tmp/cg_d1d3_evaluation_20260914_01')
sys.path[:0]=[str(RUN),str(OLD)]
import full_evaluation as F
from controls import T,DATA
import d1_controls as D
from d3_runner import PointNextObs
from external_controls import load_external
from torch.utils.data import DataLoader
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def jobs():
 return [dict(id=f'D1_{v}_{s}',path=str(RUN/'train'/f'D1_{v}_{s}'),family='D1',dataset='DALES',variant=v,seed=s) for v in D.VARIANTS for s in [43,44]]+[dict(id=f'D3_{d}_s{s}',path=str(RUN/'d3_runs'/f'D3_{d}_s{s}'),family='D3',dataset=d,variant='PointNeXt_input',seed=s) for d in ['DALES','H3D'] for s in [43,44]]
class BatchModel(torch.nn.Module):
 def __init__(self,m):super().__init__();self.m=m
 def forward(self,b):return self.m(b['coords'][0],b['base'],b['degraded'],b['delta'],b['rel'])
def setup(j,split):
 p=Path(j['path']);cfg=json.loads((p/'config.json').read_text());cp=p/'best.pt'
 expected=json.loads((OUT/'WEIGHT_LOCK.json').read_text())['checkpoints'][j['id']]
 assert sha(cp)==expected
 if j['dataset']=='DALES':
  cls=D.install_dataset();root=Path('/root/autodl-tmp/als-transfer/data') if split=='test' else DATA
  ds=cls(block_dir=root/'degraded_blocks_b50_v3_medium',clean_map_dir=OUT/'FORBIDDEN',deg_map_dir=root/'maps_25d_b50_r050_degraded_v3_medium_shared_v3',restored_map_dir=OUT/'FORBIDDEN',split=split,mode='restored_rgdi',num_points=8192,num_layers=4,num_neighbors=16,sub_sampling_ratio=[4]*4,degradation_tag='deg_v3_medium',seed=42)
  collate=T.randla_collate;nc=8;batch=2
 else:
  m,original,stats=load_external('H3D');ds=original(split,42,'observation','medium');collate=m.collate;nc=11;batch=16
 if j['family']=='D1':model=D.model_class(j['variant'])(**D.model_args(cfg),device=torch.device('cuda'))
 else:model=PointNextObs(nc,cfg['radius_metres'],cfg['embed_ch'],cfg['gate_bias'])
 model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'],strict=True);model.cuda().eval()
 if j['family']=='D3':model=BatchModel(model)
 env=dict(job=dict(j,variant='O_full'),split=split,ds=ds,collate=collate,model=model,nc=nc,batch=batch)
 if j['dataset']=='H3D':env.update(module=m,stats=stats)
 return env
def forward(env,b):
 b=T.move_batch_to_device(b,torch.device('cuda'));z=env['model'](b)
 assert torch.isfinite(z).all();return z
@torch.no_grad()
def evaluate(j,mode):
 split='val' if mode=='val' else 'test'
 if split=='test':
  lock=json.loads((OUT/'PRETEST_LOCK.json').read_text());assert lock['validation_reproduced']==8 and lock['source_sha256']==sha(__file__)
 out=OUT/mode/j['id'];out.mkdir(parents=True,exist_ok=False)
 env=setup(j,split);cm=np.zeros((env['nc'],env['nc']),np.int64);rows=[];start=time.time()
 if mode=='val':
  ds=env['ds'];d=F.load_block(env,0)
  ids=np.random.default_rng(T.stable_seed(42,ds.files[0].name)).choice(len(d['labels']),8192,replace=len(d['labels'])<8192)
  a=ds[0];b=F.sample(env,d,ids)
  for k in ['base','degraded','delta','restored','labels','hard','rel']:np.testing.assert_allclose(a[k],b[k],atol=1e-6,rtol=1e-6,err_msg=k)
  for k in ['coords','neighbor_indices','sub_idx','interp_idx']:
   for x,y in zip(a[k],b[k]):np.testing.assert_array_equal(x,y)
  chunks=list(F.groups(len(d['labels']),ds.files[0].name));np.testing.assert_array_equal(np.sort(np.concatenate([x[:n] for x,n in chunks])),np.arange(len(d['labels'])))
  za=forward(env,env['collate']([a]));zb=forward(env,env['collate']([b]));torch.testing.assert_close(za,zb,atol=1e-6,rtol=1e-6)
  write(out/'adapter_validation.json',dict(status='passed',input_fields_equal=True,hierarchy_equal=True,logits_equal=True,retained_coverage_exact=True))
 if mode!='full':
  dl=DataLoader(env['ds'],batch_size=env['batch'],num_workers=2,pin_memory=True,collate_fn=env['collate'])
  index=0
  for b in dl:
   y=b['labels'].numpy().copy();pr=forward(env,b).argmax(-1).cpu().numpy()
   for yy,pp in zip(y,pr):
    assert yy.min()>=0 and yy.max()<env['nc']
    c=np.bincount((yy*env['nc']+pp).ravel(),minlength=cm.size).reshape(cm.shape);cm+=c
    rows.append(dict(file=env['ds'].files[index].name,points=int(c.sum()),confusion=c.tolist()));index+=1
   write(out/'status.json',dict(status='running',blocks=index,total_blocks=len(env['ds']),seconds=time.time()-start))
  assert cm.sum()==len(env['ds'])*8192
 else:
  for i,p in enumerate(env['ds'].files):
   d=F.load_block(env,i);n=len(d['labels']);pred=np.full(n,255,np.uint8);seen=np.zeros(n,bool)
   for ids,valid in F.groups(n,p.name):
    b=env['collate']([F.sample(env,d,ids)]);z=forward(env,b);keep=ids[:valid]
    assert not seen[keep].any();seen[keep]=True;pred[keep]=z.argmax(-1)[0,:valid].cpu().numpy()
   assert seen.all() and (pred<env['nc']).all()
   c=np.bincount(d['labels'].astype(np.int64)*env['nc']+pred,minlength=cm.size).reshape(cm.shape);cm+=c
   np.savez_compressed(out/(p.stem+'.npz'),prediction=pred,confusion=c)
   h=hashlib.sha256()
   for key in ['xyz','local','scalar','labels']:
    a=np.ascontiguousarray(d[key]);h.update(key.encode());h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
   h.update(np.ascontiguousarray(d['map']['maps_norm'] if j['dataset']=='DALES' else d['raster']).tobytes())
   rows.append(dict(file=p.name,points=n,input_sha256=h.hexdigest(),confusion=c.tolist()))
   write(out/'status.json',dict(status='running',blocks=i+1,total_blocks=len(env['ds']),seconds=time.time()-start))
 assert len(rows)==len(env['ds'])
 expected=json.loads((OUT/'WEIGHT_LOCK.json').read_text())['checkpoints'][j['id']];assert sha(Path(j['path'])/'best.pt')==expected
 result=dict(job=j,mode=mode,split=split,severity='medium',sampling_seed=42,points=int(cm.sum()),blocks=rows,confusion=cm.tolist(),metrics=T.metrics_from_confusion(cm),checkpoint_sha256=expected,seconds=time.time()-start,source_sha256=sha(__file__))
 if mode=='val':
  target=json.loads((Path(j['path'])/'best_val_metrics.json').read_text())['mIoU'];result['validation_reference']=target;result['validation_abs_diff']=abs(result['metrics']['mIoU']-target)
  assert result['validation_abs_diff']<1e-6,(j,result['metrics']['mIoU'],target)
 write(out/'results.json',result);write(out/'status.json',dict(status='complete',blocks=len(rows),seconds=time.time()-start))
if __name__=='__main__':
 j=next(x for x in jobs() if x['id']==sys.argv[1])
 try:evaluate(j,sys.argv[2])
 except Exception as e:write(OUT/sys.argv[2]/j['id']/'failure.json',dict(error=repr(e)));raise
