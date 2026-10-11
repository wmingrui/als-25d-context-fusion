"""WHU-Urban3D ALS extension; original RandLA/fusion topology, isolated data protocol."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.util,csv,argparse,random
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from multiprocessing import Value
from functools import lru_cache
Q=Path(__file__).resolve().parent;D=Q/'data';PROJECT=Path('/root/autodl-tmp/ALS-DiffBridge')
SRC=PROJECT/'scripts/train';sys.path.insert(0,str(SRC));sys.modules['open3d']=None

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
T=module('whu_randla',SRC/'39_train_randlanet_reliability_ablation.py');U=module('whu_unet',SRC/'12_train_unet_25d_restoration.py')
T.NUM_CLASSES=8
PROTOCOL=json.loads((Q/'protocol.json').read_text());T.CLASS_NAMES=PROTOCOL['class_names']
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
@lru_cache(maxsize=32)
def points(folder):return {k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['xyz','local','intensity','labels']}
class Points(Dataset):
 def __init__(self,split,seed,method,severity='medium'):
  self.split=split;self.seed=seed;self.method=method;self.epoch=Value('i',0);self.files=sorted((D/'point_cache'/severity/split).iterdir());self.maps={}
  h=json.loads((Q/'normalization.json').read_text())['stats_sha256']
  for p in self.files:
   with np.load(Path(PROTOCOL['root'])/'data'/f'maps_25d_b50_r050_degraded_v3_{severity}_shared_v3'/split/(p.name+'_25d.npz')) as z:
    assert str(z['normalization_stats_sha256'])==h
    raw=z['maps_norm'].astype(np.float32)
    masks=np.stack([z[k] for k in ['valid_mask','structural_hard_mask','low_density_mask','high_gradient_mask','high_roughness_mask']]).astype(np.float32)
    self.maps[p.name]=(raw,np.concatenate([raw,masks]),float(z['x0']),float(z['y0']))
 def __len__(self):return len(self.files)
 def __getitem__(self,i):
  p=self.files[i];d=points(p);n=len(d['labels']);s=42 if self.split!='train' else self.seed+100003*self.epoch.value
  rng=np.random.default_rng(T.stable_seed(s,p.name));idx=rng.choice(n,8192,replace=n<8192)
  xyz=d['xyz'][idx].copy();coords=d['local'][idx].copy();b=coords.copy();b[:,:2]/=25;b[:,2]=np.clip(b[:,2]/20,-2,2);base=np.concatenate([b,d['intensity'][idx]],1).astype(np.float32)
  labels=d['labels'][idx].astype(np.int64)
  raw,ras,x0,y0=self.maps[p.name];ix=np.clip(np.floor((xyz[:,0]-x0)/.5).astype(np.int64),0,99);iy=np.clip(np.floor((xyz[:,1]-y0)/.5).astype(np.int64),0,99)
  prior=raw[:,iy,ix].T.copy();hier=T.build_randlanet_hierarchy(coords,4,16,[4]*4)
  return dict(base=base,degraded=prior,delta=prior.copy(),restored=np.zeros_like(prior),rel=np.zeros((8192,6),np.float32),labels=labels,hard=ras[10,iy,ix].astype(np.float32),name=p.name,**hier,raster=ras,cell=iy*100+ix)
def collate(items):
 b=T.randla_collate(items);b['name']=[i['name'] for i in items];b['raster']=torch.from_numpy(np.stack([x['raster'] for x in items]));b['cell']=torch.from_numpy(np.stack([x['cell'] for x in items]));return b
class Model(torch.nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  self.point=T.RandLASegmentationModel(mode='base' if method=='base' else 'restored_rgdi',num_points=8192,num_neighbors=16,num_layers=4,sub_sampling_ratio=[4]*4,dim_output=[16,64,128,256],dim_features=8,embed_ch=32,gate_bias=.3,inject_stages=0 if method=='base' else 1,internal_hidden_ch=32,internal_alpha_init=.1,use_explicit_quality_modulation=False,use_reliability_descriptors=False,gate_mode='learned',dropout=.5,device=torch.device('cuda'))
  self.unet=None
  if method in ['feature','random_feature']:
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(42);self.unet=U.UNet2D(in_ch=14,out_ch=9,base_ch=48,dropout=.05)
   if method=='feature':self.unet.load_state_dict(torch.load(Path(PROTOCOL['unet_checkpoint']),map_location='cpu',weights_only=False)['model'])
   self.unet.eval().requires_grad_(False);self.unet.dec1.register_forward_hook(self.capture)
   with torch.random.fork_rng(devices=[]):self.projection=torch.nn.Linear(48,9)
 def capture(self,m,i,o):self.latent=o
 def train(self,mode=True):
  super().train(mode)
  if self.unet is not None:self.unet.eval()
  return self
 def forward(self,b):
  if self.unet is not None:
   with torch.no_grad():self.unet(b['raster'])
   z=self.latent.flatten(2).transpose(1,2);lookup=z.gather(1,b['cell'].unsqueeze(-1).expand(-1,-1,48))
   b=dict(b);b['delta']=self.projection(lookup)
  return self.point(b)
def gpu(b):
 b=T.move_batch_to_device(b,torch.device('cuda'));b['raster']=b['raster'].cuda(non_blocking=True);b['cell']=b['cell'].cuda(non_blocking=True);return b
@torch.no_grad()
def evaluate(model,loader,detailed=False):
 model.eval();cm=np.zeros((8,8),np.int64);hardcm=np.zeros_like(cm);scenes={}
 for b in loader:
  b=gpu(b);logits=model(b);pred=logits.argmax(-1).cpu().numpy();lab=b['labels'].cpu().numpy();T.update_confusion(cm,pred,lab)
  if detailed:
   hard=b['hard'].cpu().numpy()>0.5;T.update_confusion(hardcm,pred[hard],lab[hard])
   for i,name in enumerate(b['name']):
    scene=name.split('_')[0];scenes.setdefault(scene,np.zeros_like(cm));T.update_confusion(scenes[scene],pred[i],lab[i])
 if cm.sum()==0:raise RuntimeError('No valid labels in evaluation')
 metrics=T.metrics_from_confusion(cm);masked=cm.copy();masked[0,:]=0;u=masked.sum(0)+masked.sum(1)-masked.diagonal();iou=np.divide(masked.diagonal(),u,out=np.zeros(8,dtype=float),where=u>0);metrics['mIoU_7_ignore_other']=float(iou[1:].mean());metrics['hard_metrics']=T.metrics_from_confusion(hardcm) if detailed else None;metrics['per_scene']={k:{'metrics':T.metrics_from_confusion(v),'confusion':v.tolist()} for k,v in scenes.items()};metrics['evaluated_points']=int(cm.sum());metrics['valid_nonother_points']=int(masked.sum());return metrics,cm

def train_points(args):
 seed(args.seed);out=Q/('benchmarks' if args.benchmark else 'train')/f's{args.seed}_{args.method}_b{args.batch}' if args.benchmark else Q/'train'/f's{args.seed}_{args.method}';out.mkdir(parents=True,exist_ok=True)
 if (out/'status.json').exists() and not args.resume and not args.benchmark:raise RuntimeError('Existing run; use --resume')
 ds=Points('train',args.seed,args.method);vs=Points('val',args.seed,args.method)
 gen=torch.Generator().manual_seed(args.seed)
 dl=DataLoader(ds,batch_size=args.batch,shuffle=True,generator=gen,num_workers=args.workers,persistent_workers=args.workers>0,pin_memory=True,collate_fn=collate)
 vl=DataLoader(vs,batch_size=args.batch,num_workers=2,persistent_workers=True,pin_memory=True,collate_fn=collate)
 m=Model(args.method).cuda();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=.00005)
 counts=np.array(json.loads((Q/'train_class_counts.json').read_text()))+1;w=1/np.sqrt(counts/counts.sum());w=np.clip(w/w.mean(),.5,8);weight=torch.tensor(w,dtype=torch.float32,device='cuda')
 best=-1.;start=1
 if args.resume:
  ck=torch.load(out/'last.pt',weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);start=ck['epoch']+1;best=ck['best'];gen.set_state(ck['loader_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
 if not args.benchmark:
  write(out/'config.json',{**vars(args),'protocol_sha256':sha(Q/'protocol.json'),'normalization_sha256':sha(Q/'normalization.json'),'class_names':T.CLASS_NAMES,'precision':'BF16 autocast training; FP32 validation','data_protocol':'50m nonoverlapping blocks, 8192 sampled records; 8-class training/main metrics, secondary 7-class metrics ignoring other GT; not full-scene benchmark','source_sha256':sha(__file__),'upstream_sha256':sha(Path(PROTOCOL['unet_checkpoint'])) if args.method=='feature' else None})
 t0=time.time();steps=0;total_examples=0;torch.cuda.reset_peak_memory_stats()
 for ep in range(start,args.epochs+1):
  ds.epoch.value=ep;m.train();loss_sum=0.;n=0;epochstart=time.time()
  for it,b in enumerate(dl):
   b=gpu(b);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    logits=m(b);valid=b['labels']!=-100
    loss=F.cross_entropy(logits.reshape(-1,8),b['labels'].reshape(-1),weight=weight,ignore_index=-100) if valid.any() else logits.sum()*0
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
   if not torch.isfinite(gn):raise RuntimeError('Nonfinite gradients')
   opt.step();loss_sum+=float(loss.detach())*len(b['labels']);n+=len(b['labels']);steps+=1;total_examples+=len(b['labels'])
   if args.benchmark and steps>=args.benchmark:
    torch.cuda.synchronize();print(json.dumps({'steps':steps,'seconds':time.time()-t0,'blocks_per_second':total_examples/(time.time()-t0),'peak_GB':torch.cuda.max_memory_allocated()/1e9,'loss':loss_sum/n}),flush=True);return
   if it%10==0:
    state={'state':'training','epoch':ep,'epochs':args.epochs,'batch':it+1,'batches':len(dl),'loss':float(loss.detach()),'elapsed_seconds':time.time()-t0,'peak_GB':torch.cuda.max_memory_allocated()/1e9}
    write(out/'status.json',state);print(json.dumps(state),flush=True)
  metric,cm=evaluate(m,vl);scheduler.step();row={'epoch':ep,'train_loss':loss_sum/n,'val_mIoU':metric['mIoU'],'val_oa':metric['oa'],'seconds':time.time()-epochstart}
  with (out/'history.csv').open('a',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(row));
   if f.tell()==0:writer.writeheader()
   writer.writerow(row)
  ck={'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':ep,'best':max(best,metric['mIoU']),'loader_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
  torch.save(ck,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
  if metric['mIoU']>best:
   best=metric['mIoU'];torch.save({'model':m.state_dict(),'epoch':ep,'metrics':metric},out/'best.tmp');(out/'best.tmp').replace(out/'best.pt');write(out/'best_val_metrics.json',metric);np.save(out/'best_val_confusion.npy',cm)
  print(json.dumps(row),flush=True)
 write(out/'status.json',{'state':'complete','epochs':args.epochs,'best_val_mIoU':best,'seconds':time.time()-t0,'checkpoint_sha256':sha(out/'best.pt')})

class Maps(Dataset):
 def __init__(self,split):
  original=U.ALS25DRestorationDataset(D/'maps_clean',D/'maps_medium',split,degradation_tag='deg_v3_medium');self.samples=[original[i] for i in range(len(original))]
 def __len__(self):return len(self.samples)
 def __getitem__(self,i):return self.samples[i]
def train_unet(args):
 seed(42);out=Q/'unet';out.mkdir(exist_ok=True);m=U.UNet2D(14,9,48,0).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
 ds=Maps('train');vs=Maps('val');dl=DataLoader(ds,batch_size=32,shuffle=True,pin_memory=True);vl=DataLoader(vs,batch_size=32,pin_memory=True)
 weight=[1,2,1,1.5,2,1.5,.5,1.5,2];best=float('inf');t0=time.time()
 for ep in range(1,51):
  scores=[]
  for training,loader in [(True,dl),(False,vl)]:
   m.train(training);total=0;n=0
   for b in loader:
    x=b['x'].cuda(non_blocking=True);y=b['y'].cuda(non_blocking=True);v=b['clean_valid'].cuda();h=b['clean_hard'].cuda()
    opt.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16,enabled=training):
     pred=m(x);loss=U.masked_l1(pred,y,v,weight)+U.masked_l1(pred,y,v*h,weight)+.2*U.gradient_loss(pred,y,v,1)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite U-Net loss')
    if training:loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    total+=float(loss.detach())*len(x);n+=len(x)
   scores.append(total/n)
  if scores[1]<best:best=scores[1];torch.save({'model':m.state_dict(),'epoch':ep,'val_loss':best},out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
  row={'state':'training','epoch':ep,'train_loss':scores[0],'val_loss':scores[1],'seconds':time.time()-t0};write(out/'status.json',row);print(json.dumps(row),flush=True)
  with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 write(out/'status.json',{'state':'complete','epochs':50,'best_val_loss':best,'checkpoint_sha256':sha(out/'best.pt'),'seconds':time.time()-t0})
def evaluate_best(args):
 out=Q/'train'/f's{args.seed}_{args.method}';ckpath=out/'best.pt';frozen=sha(ckpath)
 ds=Points('test',42,args.method,args.severity);dl=DataLoader(ds,batch_size=args.batch,num_workers=args.workers,pin_memory=True,collate_fn=collate)
 model=Model(args.method).cuda();ck=torch.load(ckpath,weights_only=False);model.load_state_dict(ck['model']);metrics,cm=evaluate(model,dl,True)
 assert sha(ckpath)==frozen
 dest=Q/'eval'/args.severity;dest.mkdir(parents=True,exist_ok=True);name=f's{args.seed}_{args.method}'
 write(dest/(name+'.json'),{'method':args.method,'seed':args.seed,'checkpoint_sha256':frozen,'selected_epoch':ck['epoch'],'split':'test','blocks':len(ds),'sampling_seed':42,'severity':args.severity,'frozen_medium_unet':PROTOCOL['unet_checkpoint'],'metrics':metrics});np.save(dest/(name+'_confusion.npy'),cm)
 print(json.dumps(metrics),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['base','observation','feature','random_feature','unet'],required=True);p.add_argument('--seed',type=int,default=43);p.add_argument('--batch',type=int,default=16);p.add_argument('--severity',choices=['mild','medium','severe'],default='medium');p.add_argument('--workers',type=int,default=4);p.add_argument('--epochs',type=int,default=100);p.add_argument('--benchmark',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--eval-only',action='store_true');a=p.parse_args()
 if a.eval_only:evaluate_best(a)
 elif a.method=='unet':train_unet(a)
 else:train_points(a)
