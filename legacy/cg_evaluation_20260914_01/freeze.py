import csv,json,time
from evaluation import Q,TRAIN,jobs,sha,write
review=[]
for j in jobs():
    p=TRAIN/j['path'];v=json.loads((Q/'val/medium'/(j['path'].replace('/','__')+'.json')).read_text())
    assert abs(v['metrics']['mIoU']-j['best_validation_mIoU'])<1e-6
    hist=list(csv.DictReader((p/'history.csv').open()));values=[float(r['val_mIoU']) for r in hist]
    review.append(dict(job=j['path'],best_epoch=j['checkpoints']['best.pt']['epoch'],epochs=len(hist),best_validation=max(values),last10_mean=sum(values[-10:])/10,previous10_mean=sum(values[-20:-10])/10,decision='Retain predeclared training budget and validation-selected checkpoint; budget-constrained comparison, no claim of fully tuned convergence.'))
    assert sha(p/'best.pt')==j['checkpoints']['best.pt']['sha256']
lock=dict(created=time.time(),validation_reproduced=14,checkpoints={j['path']:j['checkpoints']['best.pt']['sha256'] for j in jobs()},evaluation_source_sha256=sha(Q/'evaluation.py'),sampling_seed=42,severities=['medium','mild','severe'],precision='FP32, TF32 disabled',validation_review=review,test_results_seen=False)
assert not (Q/'PRETEST_LOCK.json').exists();write(Q/'PRETEST_LOCK.json',lock)
print('Locked 14 checkpoints after validation reproduction and trajectory review.')
