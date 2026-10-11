"""Official PointNeXt-S, point-only inputs, bounded validation-only fitting."""
import argparse,csv,hashlib,json,os,random,sys,time
from pathlib import Path
os.environ['WANDB_MODE']='disabled'
os.environ['WANDB_DISABLED']='true'
import numpy as np,torch
from torch.utils.data import DataLoader
from controls import T,DATA,ROOT,module
Q=Path(__file__).resolve().parent
VENDOR=Q/'vendor/PointNeXt-master'
sys.path.insert(0,str(VENDOR))
from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
from external_controls import ROOTS

class PointNext(torch.nn.Module):
    def __init__(self,classes,radius):
        super().__init__();cfg=EasyConfig();cfg.load(str(VENDOR/'cfgs/s3dis/pointnext-s.yaml'))
        cfg.model.encoder_args.radius=radius;cfg.model.cls_args.num_classes=classes
        self.net=build_model_from_cfg(cfg.model)
    def forward(self,coords,base):
        return self.net({'pos':coords.contiguous(),'x':base.transpose(1,2).contiguous()}).transpose(1,2).contiguous()

def datasets(name,seed):
    if name=='DALES':
        kw=dict(block_dir=DATA/'degraded_blocks_b50_v3_medium',clean_map_dir=DATA/'FORBIDDEN',deg_map_dir=DATA/'FORBIDDEN',restored_map_dir=DATA/'FORBIDDEN',mode='base',num_points=8192,num_layers=4,num_neighbors=16,sub_sampling_ratio=[4]*4,degradation_tag='deg_v3_medium',seed=seed)
        cls=T._RESTORATION_CONTROL_ORIGINAL_DATASET
        return cls(split='train',**kw),cls(split='val',**kw),T.randla_collate,T.compute_class_weights(kw['block_dir']),8
    root=ROOTS[name];m=module('pointnext_data_'+name,root/'train_complete.py')
    counts=np.array(json.loads((root/'train_class_counts.json').read_text()))+1
    w=1/np.sqrt(counts/counts.sum());w=np.clip(w/w.mean(),.5,8)
    return m.Points('train',seed,'base'),m.Points('val',seed,'base'),m.collate,torch.tensor(w,dtype=torch.float32),len(counts)

def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['DALES','WHU','H3D'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--radius',type=float,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--epochs',type=int,required=True);p.add_argument('--job',required=True);a=p.parse_args()
    assert '/' not in a.job and '..' not in a.job
    out=Q/'pointnext_runs'/a.job;out.mkdir(parents=True,exist_ok=False)
    T.seed_everything(a.seed);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ds,vs,collate,weights,nc=datasets(a.dataset,a.seed)
    batch=2 if a.dataset=='DALES' else 16;workers=4 if a.dataset=='DALES' else 2
    gen=torch.Generator().manual_seed(a.seed)
    dl=DataLoader(ds,batch_size=batch,shuffle=True,generator=gen,num_workers=workers,persistent_workers=True,pin_memory=True,drop_last=a.dataset=='DALES',collate_fn=collate)
    vl=DataLoader(vs,batch_size=batch,num_workers=2,persistent_workers=True,pin_memory=True,collate_fn=collate)
    m=PointNext(nc,a.radius).cuda();opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=.0001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=a.lr*.05);weights=weights.cuda();best=-1
    write(out/'config.json',dict(**vars(a),batch=batch,workers=workers,precision='FP32, TF32 disabled',point_inputs='existing normalized XYZ and scalar attribute; metric local coordinates for neighborhoods',additional_augmentation=False,pretrained=False,validation_sampling_seed=42,train_samples_per_epoch=len(ds),test_access=False,parameters=sum(p.numel() for p in m.parameters()),pointnext_commit='b7ec10fd8b9acdb1c3b18e557f0d3465eb919164',openpoints_commit='db31e0d94ede35bc672ef06d1b6e4073a9c269d5',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),resume='not automatic; worker RNG not claimed exactly restorable'))
    started=time.time()
    try:
        for ep in range(1,a.epochs+1):
            if hasattr(ds,'epoch'):ds.epoch.value=ep
            m.train();loss_sum=0;seen=0;t0=time.time()
            for it,b in enumerate(dl):
                x=b['base'].cuda(non_blocking=True);coords=b['coords'][0].cuda(non_blocking=True);y=b['labels'].cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True);z=m(coords,x);loss=torch.nn.functional.cross_entropy(z.reshape(-1,nc),y.reshape(-1),weight=weights)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();loss_sum+=float(loss.detach())*len(y);seen+=len(y)
                if it%20==0:write(out/'status.json',dict(status='running',epoch=ep,batch=it+1,batches=len(dl),elapsed=time.time()-started))
            m.eval();cm=np.zeros((nc,nc),np.int64)
            with torch.no_grad():
                for b in vl:
                    z=m(b['coords'][0].cuda(non_blocking=True),b['base'].cuda(non_blocking=True))
                    if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite validation logits')
                    pred=z.argmax(-1).cpu().numpy();y=b['labels'].numpy();cm+=np.bincount((y*nc+pred).reshape(-1),minlength=nc*nc).reshape(nc,nc)
            assert cm.sum()==len(vs)*8192
            metrics=T.metrics_from_confusion(cm);sched.step()
            row=dict(epoch=ep,train_loss=loss_sum/seen,val_mIoU=metrics['mIoU'],val_oa=metrics['oa'],seconds=time.time()-t0)
            with (out/'history.csv').open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=row);w.writeheader() if f.tell()==0 else None;w.writerow(row)
            ck=dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=ep,metrics=metrics,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=gen.get_state())
            torch.save(ck,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
            if metrics['mIoU']>best:
                best=metrics['mIoU'];torch.save(ck,out/'best.tmp');(out/'best.tmp').replace(out/'best.pt');np.save(out/'best_val_confusion.npy',cm);write(out/'best_val_metrics.json',dict(epoch=ep,**metrics))
            print(json.dumps(row),flush=True)
        write(out/'status.json',dict(status='complete',epochs=a.epochs,best_val_mIoU=best,seconds=time.time()-started))
    except BaseException as e:write(out/'status.json',dict(status='failed',error=repr(e)));raise

if __name__=='__main__':main()
