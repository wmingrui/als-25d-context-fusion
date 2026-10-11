"""Pooled-cell terrain scores and explicit scene-level summaries."""
from pathlib import Path
import json,csv,collections,statistics,tarfile
import numpy as np
from terrain_core import error_stats
Q=Path(__file__).resolve().parent;P=Q/'terrain'
meta=json.loads((P/'models.json').read_text());files=sorted((P/'cells').glob('*.npz'));assert len(files)==1100
refs=[];preds=[];commons=[];density=[];slope=[];tiles=[];file_names=[];refcounts=[];predcounts=[]
for p in files:
 with np.load(p) as z:
  refs.append(z['reference']);preds.append(z['predicted']);commons.append(z['common']);density.append(z['density_group']);slope.append(z['slope_group']);tiles.append(str(z['tile']));file_names.append(p.name);refcounts.append(z['reference_count']);predcounts.append(z['predicted_count'])
ref=np.concatenate(refs);pred=np.concatenate(preds,axis=1);common=np.concatenate(commons);dg=np.concatenate(density);sg=np.concatenate(slope)
tile_ids=sorted(set(tiles));tile=np.repeat([tile_ids.index(t) for t in tiles],2500)
assert len(tile_ids)==11 and pred.shape==(15,2750000)
allrows=[]
for label,mask in [('ALL',np.ones(len(ref),bool))]+[(t,tile==i) for i,t in enumerate(tile_ids)]:
 for ji,jid in enumerate(meta['ids']):
  for scope,smask in [('own',mask),('common_all',mask&common)]:
   r=error_stats(ref,pred[ji],smask);r.update(parent=label,method=meta['variants'][ji],seed=meta['seeds'][ji],model=jid,scope=scope,
       total_reference_cells=int((np.isfinite(ref)&mask).sum()))
   r['coverage_of_all_reference']=r['valid_cells']/r['total_reference_cells'] if r['total_reference_cells'] else None
   allrows.append(r)
def save(name,rs):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
save('terrain_per_seed.csv',allrows)
strata=[]
for kind,group in [('reference_density',dg),('reference_slope',sg)]:
 for k in [-1,0,1,2]:
  mask=group==k
  for ji,jid in enumerate(meta['ids']):
   for scope,smask in [('own',mask),('common_all',mask&common)]:
    r=error_stats(ref,pred[ji],smask);r.update(stratum=kind,group=k,method=meta['variants'][ji],seed=meta['seeds'][ji],model=jid,scope=scope,
       total_reference_cells=int((np.isfinite(ref)&mask).sum()))
    r['coverage_of_all_reference']=r['valid_cells']/r['total_reference_cells'] if r['total_reference_cells'] else None
    strata.append(r)
save('terrain_strata.csv',strata)
groups=collections.defaultdict(list)
for r in allrows:groups[(r['parent'],r['method'],r['scope'])].append(r)
summary=[]
for (parent,method,scope),rs in groups.items():
 row=dict(parent=parent,method=method,scope=scope,n_seeds=len(rs))
 for metric in ['mae_m','rmse_m','median_abs_error_m','p95_abs_error_m','bias_m','coverage_of_all_reference','false_positive_occupancy','valid_cells','total_reference_cells']:
  vals=[r[metric] for r in rs if r[metric] is not None]
  row[metric+'_mean']=statistics.mean(vals) if vals else None;row[metric+'_SD']=statistics.stdev(vals) if len(vals)>1 else None
 summary.append(row)
save('terrain_summary.csv',summary)

eligible=[i for i,r in enumerate(refs) if np.isfinite(r).sum()>=1000]
differences=[]
for i in eligible:
 diffs=[]
 for s in [43,44]:
  bi=next(j for j in range(15) if meta['variants'][j]=='Base' and meta['seeds'][j]==s)
  oi=next(j for j in range(15) if meta['variants'][j]=='O_full' and meta['seeds'][j]==s)
  mask=commons[i]
  if mask.any():diffs.append(np.mean(np.abs(preds[i][oi,mask]-refs[i][mask]))-np.mean(np.abs(preds[i][bi,mask]-refs[i][mask])))
 differences.append((statistics.mean(diffs) if diffs else -np.inf,i))
worst=max(differences,key=lambda x:(x[0],-x[1]))
selected=[dict(role='filename-first supported block',file=file_names[eligible[0]]),dict(role='largest O-full minus Base common-cell MAE; explicitly selected failure',file=file_names[worst[1]],delta_mae_m=worst[0])]
json.dump(selected,(P/'figure_selection.json').open('w'),indent=2)
counts=np.concatenate(predcounts,axis=1);reference_counts=np.concatenate(refcounts)
audit=dict(blocks=1100,parent_tiles=11,grid_cells=len(ref),reference_cells=int(np.isfinite(ref).sum()),common_cells=int(common.sum()),
  reference_ground_points=int(reference_counts.sum()),predicted_ground_points_by_model={m:int(counts[i].sum()) for i,m in enumerate(meta['ids'])},
  input_points=sum(r['retained_points'] for r in json.loads((P/'input_audit.json').read_text())),
  no_interpolation=True,no_prediction_edits=True,scopes=['own','common_all'],metrics_aggregation='pool cells within each parent/all; then mean and sample SD across two seeds; oracle has one fixed output')
assert audit['input_points']==77044189
json.dump(audit,(P/'terrain_audit.json').open('w'),indent=2)
with tarfile.open(Q/'terrain_results.tar.gz','w:gz') as t:
 for p in P.glob('*'):
  if p.is_file():t.add(p,arcname='terrain/'+p.name)
 for s in selected:t.add(P/'cells'/s['file'],arcname='terrain/selected_cells/'+s['file'])
 t.add(Q/'TERRAIN_PROTOCOL_LOCK.json',arcname='TERRAIN_PROTOCOL_LOCK.json')
for r in summary:
 if r['parent']=='ALL' and r['scope']=='own':print(r['method'],'MAE',r['mae_m_mean'],'coverage',r['coverage_of_all_reference_mean'],flush=True)
print('SUMMARY COMPLETE',flush=True)
