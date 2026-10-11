"""Post-segmentation terrain analysis from unmodified retained-point predictions."""
from pathlib import Path
import argparse,csv,hashlib,json,time,re
import numpy as np
from terrain_core import median_grid,slope_grid,error_stats
Q=Path(__file__).resolve().parent
TRAIN=Path('/root/autodl-tmp/ALS-DiffBridge/data/processed/dales_objects')
TEST=Path('/root/autodl-tmp/als-transfer/data')
EVAL=Path('/root/autodl-tmp/cg_evaluation_20260914_01/full_retained')

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def block(p):
 with np.load(p,allow_pickle=True) as z:return {k:z[k] for k in z.files}
def alignment(clean,deg):
 ids=deg['kept_indices'];assert len(np.unique(ids))==len(ids) and len(ids)==len(deg['points'])
 assert np.array_equal(np.flatnonzero(deg['keep_mask']),ids)
 assert np.array_equal(clean['points'][ids,:2],deg['points'][:,:2])
 assert np.array_equal(clean['labels'][ids],deg['labels'])
 np.testing.assert_allclose(clean['points'][ids,2]+deg['z_noise'],deg['points'][:,2],rtol=0,atol=1e-5)
 for k in ['x0','y0','block_size','tile_origin_x','tile_origin_y']:assert clean[k]==deg[k],k
 assert float(clean['block_size'])==50
def preflight():
 target=Q/'TERRAIN_PROTOCOL_LOCK.json';assert not target.exists()
 started=time.time();densities=[];slopes=[];hashes={}
 files=sorted((TRAIN/'full_blocks_b50/train').glob('*.npz'));assert len(files)==2400
 for i,p in enumerate(files):
  d=block(p);z,n=median_grid(d['points'][d['labels']==0],float(d['x0']),float(d['y0']));s=slope_grid(z)
  densities.append(n[n>0]);slopes.append(s[np.isfinite(s)]);hashes[str(p)]=sha(p)
  if i%100==0:write(Q/'terrain_status.json',dict(phase='train_thresholds',blocks=i+1,total=2400,seconds=time.time()-started))
 den=np.concatenate(densities);sl=np.concatenate(slopes)
 validations=[];vfiles=sorted((TRAIN/'degraded_blocks_b50_v3_medium/val').glob('*.npz'))
 for i in np.linspace(0,len(vfiles)-1,33,dtype=int):
  p=vfiles[i];d=block(p);cp=TRAIN/'full_blocks_b50/val'/p.name.replace('_deg_v3_medium','');c=block(cp);alignment(c,d)
  z,n=median_grid(c['points'][c['labels']==0],float(c['x0']),float(c['y0']))
  validations.append(dict(file=p.name,clean_sha256=sha(cp),degraded_sha256=sha(p),reference_cells=int((n>0).sum()),retained_points=len(d['points'])))
 write(target,dict(grid_resolution_m=1,grid_shape=[50,50],height_statistic='median',empty='NaN; no interpolation/smoothing',
  reference='clean dataset GT-ground cell median; not independent surveyed DTM',oracle='degraded GT-ground median, not guaranteed best',
  density_tertiles=np.quantile(den,[1/3,2/3]).tolist(),slope_tertiles_deg=np.quantile(sl,[1/3,2/3]).tolist(),
  slope_definition='central differences on reference cell medians, requires finite center and 4 direct neighbors; boundary and unsupported cells unknown',
  threshold_fit=dict(split='train',blocks=2400,density_cells=len(den),slope_cells=len(sl),density_quantiles=np.quantile(den,[0,.1,.5,.9,1]).tolist(),slope_quantiles_deg=np.quantile(sl,[0,.1,.5,.9,.99,1]).tolist()),
  validation_alignment=validations,common_scope='intersection of all 14 DALES model outputs, oracle and reference; report own coverage separately',
  posthoc=True,note='Application analysis protocol fixed after segmentation test metrics were known; no model, grid or threshold selection on terrain test results',
  seeds=[43,44],source_hashes={str(Q/'terrain_core.py'):sha(Q/'terrain_core.py'),str(Path(__file__)):sha(__file__)},train_input_hashes=hashes,seconds=time.time()-started))
 print('PREFLIGHT COMPLETE',json.dumps(dict(density_tertiles=np.quantile(den,[1/3,2/3]).tolist(),slope_tertiles=np.quantile(sl,[1/3,2/3]).tolist())),flush=True)

def evaluate():
 lock=json.loads((Q/'TERRAIN_PROTOCOL_LOCK.json').read_text());assert sha(Q/'terrain_core.py')==lock['source_hashes'][str(Q/'terrain_core.py')]
 out=Q/'terrain';out.mkdir(exist_ok=False);(out/'cells').mkdir()
 results=[json.loads(p.read_text()) for p in sorted(EVAL.glob('*/results.json'))]
 results=[r for r in results if r['job']['dataset']=='DALES'];assert len(results)==14
 jobs=[r['job'] for r in results];ids=[j['id'] for j in jobs]+['oracle'];names=[j['variant'] for j in jobs]+['Oracle']
 seeds=[int(re.findall(r'(?:^|_)s?(43|44)(?:$|_)',j['id'])[0]) for j in jobs]+[None]
 for j in jobs:assert sha(j['checkpoint'])==j['sha256']
 maps=[{b['file']:b for b in r['blocks']} for r in results]
 write(out/'models.json',dict(ids=ids,variants=names,seeds=seeds,checkpoints=jobs,protocol_sha256=sha(Q/'TERRAIN_PROTOCOL_LOCK.json')))
 records=[];input_audit=[];started=time.time();files=sorted((TEST/'degraded_blocks_b50_v3_medium/test').glob('*.npz'));assert len(files)==1100
 for i,p in enumerate(files):
  d=block(p);cp=TEST/'full_blocks_b50/test'/p.name.replace('_deg_v3_medium','');c=block(cp);alignment(c,d)
  x0,y0=float(c['x0']),float(c['y0']);ref,refn=median_grid(c['points'][c['labels']==0],x0,y0);sl=slope_grid(ref)
  heights=[];counts=[]
  for ji,j in enumerate(jobs):
   with np.load(EVAL/j['id']/p.name) as z:pred=z['prediction'];saved=z['confusion']
   assert len(pred)==len(d['labels']);cm=np.bincount(d['labels']*8+pred,minlength=64).reshape(8,8)
   assert np.array_equal(cm,saved) and np.array_equal(cm,maps[ji][p.name]['confusion'])
   z,n=median_grid(d['points'][pred==0],x0,y0);heights.append(z);counts.append(n)
  z,n=median_grid(d['points'][d['labels']==0],x0,y0);heights.append(z);counts.append(n)
  heights=np.stack(heights);counts=np.stack(counts);common=np.isfinite(ref)&np.isfinite(heights).all(0)
  density_group=np.where(refn>0,np.digitize(refn,lock['density_tertiles']),-1).astype(np.int8)
  slope_group=np.where(np.isfinite(sl),np.digitize(sl,lock['slope_tertiles_deg']),-1).astype(np.int8)
  tile=str(c['tile']);np.savez_compressed(out/'cells'/p.name,reference=ref,reference_count=refn,reference_slope=sl,predicted=heights,predicted_count=counts,common=common,density_group=density_group,slope_group=slope_group,x0=x0,y0=y0,tile=tile)
  for ji,jid in enumerate(ids):
   for scope,mask in [('own',None),('common_all',common)]:
    r=error_stats(ref,heights[ji],mask);r.update(file=p.name,tile=tile,model=jid,method=names[ji],seed=seeds[ji],scope=scope,all_reference_cells=int(np.isfinite(ref).sum()))
    records.append(r)
  input_audit.append(dict(file=p.name,tile=tile,clean_sha256=sha(cp),degraded_sha256=sha(p),retained_points=len(d['labels']),reference_cells=int(np.isfinite(ref).sum()),common_cells=int(common.sum())))
  if i%20==0:write(Q/'terrain_status.json',dict(phase='test_terrain',blocks=i+1,total=1100,seconds=time.time()-started));print('terrain',i+1,flush=True)
 with (out/'per_block.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 assert len(set(r['tile'] for r in input_audit))==11
 write(out/'input_audit.json',input_audit)
 for j in jobs:assert sha(j['checkpoint'])==j['sha256']
 write(Q/'terrain_status.json',dict(phase='complete',blocks=1100,models=14,oracle=True,parent_tiles=11,seconds=time.time()-started))
 print('TERRAIN COMPLETE',flush=True)

if __name__=='__main__':
 try:
  a=argparse.ArgumentParser();a.add_argument('phase',choices=['preflight','evaluate']);x=a.parse_args();preflight() if x.phase=='preflight' else evaluate()
 except BaseException as e:write(Q/'terrain_failure.json',dict(error=repr(e)));raise
