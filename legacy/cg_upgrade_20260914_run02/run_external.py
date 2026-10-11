import argparse,hashlib,json,os,shutil,time
from pathlib import Path
from types import SimpleNamespace
from external_controls import load_external,ROOTS
Q=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['WHU','H3D'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--benchmark',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    m,_,_=load_external(a.dataset);reference=ROOTS[a.dataset]
    cfg=json.loads((reference/'train/s43_observation/config.json').read_text())
    assert m.sha(reference/'train_complete.py')==cfg['source_sha256']
    out=Q/'external'/a.dataset
    out.mkdir(parents=True,exist_ok=True)
    for name in ['normalization.json','protocol.json','train_class_counts.json']:
        target=out/name
        if target.exists():assert target.read_bytes()==(reference/name).read_bytes()
        else:shutil.copyfile(reference/name,target)
    if a.smoke:
        out=out/'smoke';out.mkdir(exist_ok=False)
        for name in ['normalization.json','protocol.json','train_class_counts.json']:shutil.copyfile(reference/name,out/name)
    m.Q=out
    args=SimpleNamespace(method='observation',seed=a.seed,batch=cfg['batch'],severity='medium',workers=cfg['workers'],epochs=1 if a.smoke else 100,benchmark=a.benchmark,resume=False,eval_only=False)
    statusfile=out/f'cg_s{a.seed}_b{a.benchmark}_status.json'
    if statusfile.exists():raise FileExistsError(statusfile)
    state=dict(status='running',pid=os.getpid(),started=time.time(),dataset=a.dataset,variant='O_sampled',args=vars(args),test_access=False,source_hashes={name:m.sha(Q/name) for name in ['run_external.py','external_controls.py','controls.py']},reference=str(reference))
    m.write(statusfile,state)
    try:m.train_points(args);state.update(status='complete',finished=time.time())
    except BaseException as e:state.update(status='failed',error=repr(e),finished=time.time());raise
    finally:m.write(statusfile,state)
if __name__=='__main__':main()
