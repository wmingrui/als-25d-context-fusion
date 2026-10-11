import csv,json,statistics,re
from collections import defaultdict
from pathlib import Path
Q=Path(__file__).resolve().parent
records=[json.loads(p.read_text()) for p in (Q/'test').glob('*/*.json')]
assert len(records)==42
groups=defaultdict(list)
for r in records:
    assert r['points']==r['blocks']*8192 and r['sampling_seed']==42
    seed=re.search(r'(?:^|[_/])s?(43|44)(?:[_/]|$)',r['job']).group(1)
    groups[(r['dataset'],r['variant'],r['severity'])].append((seed,r['metrics']['mIoU']*100))
rows=[]
for (d,v,s),pairs in sorted(groups.items()):
    assert len(pairs)==2 and {p[0] for p in pairs}=={'43','44'}
    vals=[p[1] for p in pairs]
    rows.append(dict(dataset=d,variant=v,severity=s,mean_mIoU_percent=statistics.mean(vals),sd_mIoU_pp=statistics.stdev(vals),seed_mIoU_percent=dict(sorted(pairs))))
(Q/'FIXED_SUMMARY.json').write_text(json.dumps(dict(evaluations=42,rows=rows),indent=2))
with (Q/'FIXED_SUMMARY.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for r in rows:
    if r['severity']=='medium':print(json.dumps(r))
