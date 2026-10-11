import csv,json,statistics
from pathlib import Path
Q=Path(__file__).resolve().parent;ROOT=Path('/root/autodl-tmp/ALS-DiffBridge/PROJECT_NAV_20260909/05_new_work/results')
summary=json.loads((Q/'FIXED_SUMMARY.json').read_text())['rows'];rows=[]
for d in ['DALES','WHU','H3D']:
    r=dict(dataset=d)
    for method in ['base','observation']:
        vals=[]
        for s in [43,44]:
            if d=='DALES':
                p=ROOT/('medium_seeds43_44_test_20260910_01' if method=='base' else 'observation_seeds43_44_20260910_01/test')/f's{s}_{method}'/'test_metrics.json'
                if method=='base':p=p.parent/'evaluation'/p.name
                metric=json.loads(p.read_text())
            else:
                p=Path('/root/autodl-tmp')/('whu25d/completion_v2' if d=='WHU' else 'h3d25d/run_v2')/'eval/medium'/f's{s}_{method}.json';metric=json.loads(p.read_text())['metrics']
            vals.append(metric['mIoU']*100)
        r[method]=statistics.mean(vals)
    for x in summary:
        if x['dataset']==d and x['severity']=='medium':r[x['variant']]=x['mean_mIoU_percent']
    r['sampled_minus_base_pp']=r['O_sampled']-r['base'];r['full_minus_sampled_pp']=r['observation']-r['O_sampled'];rows.append(r)
(Q/'MEDIUM_COMPARISON.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
