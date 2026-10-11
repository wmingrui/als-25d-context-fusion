"""Independent post-run check tying terrain inputs to the frozen E1 record order."""
from pathlib import Path
import hashlib,json,re,time
import numpy as np
Q=Path(__file__).resolve().parent;E=Path('/root/autodl-tmp/cg_evaluation_20260914_01/full_retained');D=Path('/root/autodl-tmp/als-transfer/data')
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
lock=json.loads((Q/'TERRAIN_PROTOCOL_LOCK.json').read_text())
for path,h in lock['source_hashes'].items():assert sha(path)==h,(path,'changed since terrain lock')
inputs=json.loads((Q/'terrain/input_audit.json').read_text())
results=[json.loads(p.read_text()) for p in E.glob('*/results.json')];results=[r for r in results if r['job']['dataset']=='DALES']
combos={(r['job']['variant'],int(re.findall(r'(?:^|_)s?(43|44)(?:$|_)',r['job']['id'])[0])) for r in results}
assert combos=={(m,s) for m in ['Base','O_point','O_sampled','O_full','F_pre','F_rnd','PointNeXt_S'] for s in [43,44]}
maps=[{b['file']:b for b in r['blocks']} for r in results];rows=[];start=time.time()
for item in inputs:
 p=D/'degraded_blocks_b50_v3_medium/test'/item['file'];assert sha(p)==item['degraded_sha256']
 cp=D/'full_blocks_b50/test'/item['file'].replace('_deg_v3_medium','');assert sha(cp)==item['clean_sha256']
 with np.load(p,allow_pickle=True) as z:d={k:z[k] for k in ['points','points_local','features','labels']}
 arrs=dict(xyz=d['points'].astype(np.float32),local=d['points_local'].astype(np.float32),scalar=d['features'].astype(np.float32),labels=d['labels'])
 h=hashlib.sha256()
 for key in ['xyz','local','scalar','labels']:
  a=np.ascontiguousarray(arrs[key]);h.update(key.encode());h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
 with np.load(D/'maps_25d_b50_r050_degraded_v3_medium_shared_v3/test'/(p.stem+'_25d.npz')) as z:h.update(np.ascontiguousarray(z['maps_norm']).tobytes())
 actual=h.hexdigest()
 for m in maps:assert actual==m[p.name]['input_sha256'],p.name
 rows.append(dict(file=p.name,input_sha256=actual))
report=dict(status='passed',blocks=len(rows),model_input_hash_comparisons=len(rows)*len(results),exact_method_seed_grid=True,locked_code_hashes_match=True,
 existing_terrain_inputs_unchanged=True,recipe='xyz/local/scalar/labels dtype+shape+bytes, then normalized full raster bytes, matching original E1 evaluator',seconds=time.time()-start,rows=rows)
(Q/'terrain/provenance_audit.json').write_text(json.dumps(report,indent=2));print('PROVENANCE PASSED',len(rows),len(rows)*len(results),flush=True)
