import hashlib,inspect,json,os,sys,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
Q=Path(__file__).resolve().parent
TRAIN=Path('/root/autodl-tmp/cg_upgrade_20260914_run02')
sys.path.insert(0,str(TRAIN))
from controls import T,DATA,make_dataset
from external_controls import load_external
from pointnext_runner import PointNext
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def jobs():return json.loads((TRAIN/'COMPLETION_AUDIT.json').read_text())['rows']
def identity(job):
    p=job['path'];ispoint=p.startswith('pointnext_runs/')
    d='DALES' if 'DALES' in p else 'WHU' if 'WHU' in p else 'H3D'
    variant='PointNeXt_S' if ispoint else 'O_point' if 'DALES_point' in p else 'O_sampled'
    return d,variant
def construct(job,split,severity):
    p=TRAIN/job['path'];cfg=json.loads((p/'config.json').read_text());d,v=identity(job)
    if d=='DALES':
        root=Path('/root/autodl-tmp/als-transfer/data') if split=='test' else DATA
        cls=T._RESTORATION_CONTROL_ORIGINAL_DATASET if v=='PointNeXt_S' else make_dataset('point' if v=='O_point' else 'sampled')
        ds=cls(block_dir=root/f'degraded_blocks_b50_v3_{severity}',clean_map_dir=Q/'FORBIDDEN',deg_map_dir=root/f'maps_25d_b50_r050_degraded_v3_{severity}_shared_v3',restored_map_dir=Q/'FORBIDDEN',split=split,mode='base' if v=='PointNeXt_S' else 'restored_rgdi',num_points=8192,num_layers=4,num_neighbors=16,sub_sampling_ratio=[4]*4,degradation_tag=f'deg_v3_{severity}',seed=42)
        if split=='test':assert len(ds)==1100
        collate=T.randla_collate;nc=8;batch=2
        if v!='PointNeXt_S':
            model=T.RandLASegmentationModel(mode='restored_rgdi',num_points=8192,num_neighbors=16,num_layers=4,sub_sampling_ratio=[4]*4,dim_output=[16,64,128,256],dim_features=8,embed_ch=32,gate_bias=.3,inject_stages=1,internal_hidden_ch=32,internal_alpha_init=.1,use_explicit_quality_modulation=False,use_reliability_descriptors=False,gate_mode='learned',dropout=.5,device=torch.device('cuda'))
    else:
        m,original,_=load_external(d)
        cls=original if v=='PointNeXt_S' else m.Points
        ds=cls(split,42,'base' if v=='PointNeXt_S' else 'observation',severity)
        collate=m.collate;nc=8 if d=='WHU' else 11;batch=16
        if v!='PointNeXt_S':model=m.Model('observation')
    if v=='PointNeXt_S':model=PointNext(nc,cfg['radius'])
    checkpoint=p/'best.pt';assert sha(checkpoint)==job['checkpoints']['best.pt']['sha256']
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.cuda().eval()
    return ds,collate,model,nc,batch
@torch.no_grad()
def evaluate(job,split,severity):
    if split=='test':
        lock=json.loads((Q/'PRETEST_LOCK.json').read_text());assert lock['validation_reproduced']==14
        assert lock['checkpoints'][job['path']]==job['checkpoints']['best.pt']['sha256']
    out=Q/split/severity/(job['path'].replace('/','__')+'.json')
    if out.exists():raise FileExistsError(out)
    started=time.time();ds,collate,model,nc,batch=construct(job,split,severity)
    dl=DataLoader(ds,batch_size=batch,num_workers=4 if identity(job)[0]=='DALES' else 2,pin_memory=True,collate_fn=collate)
    cm=np.zeros((nc,nc),np.int64);block_cms=[]
    for b in dl:
        labels=b['labels'].numpy().copy()
        if identity(job)[1]=='PointNeXt_S':z=model(b['coords'][0].cuda(non_blocking=True),b['base'].cuda(non_blocking=True))
        else:z=model(T.move_batch_to_device(b,torch.device('cuda')))
        if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite logits')
        pred=z.argmax(-1).cpu().numpy()
        for y,pr in zip(labels,pred):
            assert y.min()>=0 and y.max()<nc
            c=np.bincount((y*nc+pr).reshape(-1),minlength=nc*nc).reshape(nc,nc);cm+=c;block_cms.append(c)
    assert len(block_cms)==len(ds) and cm.sum()==len(ds)*8192
    assert sha(TRAIN/job['path']/'best.pt')==job['checkpoints']['best.pt']['sha256']
    result=dict(dataset=identity(job)[0],variant=identity(job)[1],job=job['path'],split=split,severity=severity,blocks=len(ds),points=int(cm.sum()),sampling_seed=42,checkpoint_sha256=job['checkpoints']['best.pt']['sha256'],metrics=T.metrics_from_confusion(cm),confusion=cm.tolist(),seconds=time.time()-started)
    write(out,result);np.savez_compressed(out.with_suffix('.npz'),block_confusions=np.stack(block_cms),files=np.array([p.name for p in ds.files]))
    print(json.dumps(result),flush=True);return result
if __name__=='__main__':
    job=next(j for j in jobs() if j['path']==sys.argv[1]);evaluate(job,sys.argv[2],sys.argv[3])
