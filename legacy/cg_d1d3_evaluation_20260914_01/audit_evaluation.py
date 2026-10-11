"""Independent count, metric, input-provenance and immutable-checkpoint audit."""
from pathlib import Path
import numpy as np,json,hashlib,csv,tarfile
Q=Path('/root/autodl-tmp/cg_d1d3_evaluation_20260914_01')
OLD=Path('/root/autodl-tmp/cg_evaluation_20260914_01/full_retained')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((Q/'WEIGHT_LOCK.json').read_text());pre=json.loads((Q/'PRETEST_LOCK.json').read_text())
assert pre['validation_reproduced']==8 and pre['source_sha256']==sha(Q/'evaluate_remaining.py')==lock['source_sha256']
for n,h in lock['inherited_sources'].items():assert sha(OLD.parent/n)==h
baselines={d:json.loads((OLD/id/'results.json').read_text()) for d,id in [('DALES','old_DALES_s43_base'),('H3D','old_H3D_base_43')]}
rows=[];classes=[];audits=[]
for mode in ['fixed','full']:
 for jid,h in lock['checkpoints'].items():
  p=Q/mode/jid/'results.json';r=json.loads(p.read_text());j=r['job'];cm=np.asarray(r['confusion'],np.int64);nc=len(cm)
  assert r['checkpoint_sha256']==h==sha(Path(j['path'])/'best.pt')
  assert sha(Path(j['path'])/'last.pt')==lock['last_checkpoints'][jid]
  assert sha(Path(j['path'])/'config.json')==lock['configs'][jid]
  assert r['source_sha256']==lock['source_sha256']
  val=json.loads((Q/'val'/jid/'results.json').read_text());assert val['validation_abs_diff']<1e-6
  assert json.loads((Q/'val'/jid/'adapter_validation.json').read_text())['status']=='passed'
  blocks=r['blocks'];assert len({x['file'] for x in blocks})==len(blocks)
  assert np.array_equal(sum(np.asarray(x['confusion'],np.int64) for x in blocks),cm)
  assert sum(x['points'] for x in blocks)==r['points']==int(cm.sum())
  if mode=='fixed':assert r['points']==len(blocks)*8192
  else:
   refs={x['file']:x for x in baselines[j['dataset']]['blocks']};assert set(refs)=={x['file'] for x in blocks}
   for b in blocks:assert b['input_sha256']==refs[b['file']]['input_sha256'] and b['points']==refs[b['file']]['points']
  tp=np.diag(cm);gt=cm.sum(1);pred=cm.sum(0)
  div=lambda a,b:np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
  iou=div(tp,gt+pred-tp);rec=div(tp,gt);precision=div(tp,pred);f1=div(2*tp,gt+pred)
  metrics={'mIoU':float(iou.mean()*100),'OA':float(tp.sum()/cm.sum()*100),'mAcc':float(rec.mean()*100),'mF1':float(f1.mean()*100)}
  for k,key in [('mIoU','mIoU'),('OA','oa'),('mAcc','mAcc')]:assert abs(metrics[k]/100-r['metrics'][key])<1e-10
  row=dict(protocol=mode,dataset=j['dataset'],family=j['family'],method=j['variant'],seed=j['seed'],job=jid,severity='medium',**metrics,points=r['points'],blocks=len(blocks),checkpoint_sha256=h,source_sha256=sha(p));rows.append(row)
  for k in range(nc):classes.append(dict(protocol=mode,dataset=j['dataset'],method=j['variant'],seed=j['seed'],class_id=k,IoU=iou[k]*100,F1=f1[k]*100,Recall=rec[k]*100,Precision=precision[k]*100,GT_points=int(gt[k]),Pred_points=int(pred[k]),TP=int(tp[k])))
  audits.append(dict(job=jid,protocol=mode,blocks=len(blocks),counts_and_metrics_passed=True,input_hashes_matched=mode=='full',weights_unchanged=True))
assert len(rows)==16
for name,data in [('D1_D3_SEEDS.csv',rows),('D1_D3_CLASSES.csv',classes)]:
 with (Q/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(Q/'COMPLETION_AUDIT.json').write_text(json.dumps(dict(status='passed',validation=8,fixed=8,full=8,rows=audits),indent=2))
with tarfile.open(Q/'metadata_complete.tar.gz','w:gz') as t:
 for p in Q.rglob('*'):
  if p.is_file() and p.suffix in ['.json','.csv','.py'] and '__pycache__' not in p.parts:t.add(p,arcname=p.relative_to(Q))
print('PASS: 8 validation, 8 fixed, 8 full; all weights, source, metrics and retained input hashes verified.')
