import torch,sys
from pointnext_runner import PointNext, datasets
torch.set_num_threads(1)
name=sys.argv[1] if len(sys.argv)>1 else 'DALES'
ds,_,collate,_,nc=datasets(name,46)
b=collate([ds[i] for i in range(2 if name=='DALES' else 16)])
x=b['base'].cuda();p=b['coords'][0].cuda();y=b['labels'].cuda()
m=PointNext(nc,.5).cuda();m.train();opt=torch.optim.AdamW(m.parameters(),lr=.001)
losses=[]
for step in range(25):
    opt.zero_grad(set_to_none=True);z=m(p,x)
    assert z.shape==(*y.shape,nc) and torch.isfinite(z).all()
    loss=torch.nn.functional.cross_entropy(z.reshape(-1,nc),y.reshape(-1));loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
assert min(losses[-5:])<losses[0],losses
print({'status':'passed','dataset':name,'points':y.numel(),'classes':nc,'steps':25,'first_loss':losses[0],'last_loss':losses[-1],'parameters':sum(p.numel() for p in m.parameters()),'peak_cuda_GB':torch.cuda.max_memory_allocated()/1e9})
