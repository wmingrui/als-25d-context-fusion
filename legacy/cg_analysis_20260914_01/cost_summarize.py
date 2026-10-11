import json,csv,hashlib
from pathlib import Path
Q=Path(__file__).parent
jobs=json.loads((Q/'cost_jobs.json').read_text());rows=[];audit=[]
for j in jobs:
    p=Q/'cost_results'/j['id'];r=json.loads((p/'results.json').read_text());assert r['checkpoint_unchanged']
    validation=json.loads((Q/'cost_validation'/j['id']/'validation.json').read_text());assert validation['status']=='passed'
    storage=json.loads((p/'storage_and_preparation.json').read_text())
    with (p/'timings.csv').open() as f:raw=list(csv.DictReader(f))
    for s in r['summary']:
        peak=next(x for x in r['peaks'] if x['mode']==s['mode'] and x['route']==s['route'])
        rr=[x for x in raw if x['mode']==s['mode'] and x['route']==s['route']]
        repeats=10 if s['mode']=='sampled' else 1
        assert len(rr)==s['blocks']*repeats==s['observations']
        assert set(int(x['round']) for x in rr)==set(range(repeats))
        s.update(trainable_parameters=r['trainable_parameters'],total_parameters=r['total_parameters'],frozen_parameters=r['total_parameters']-r['trainable_parameters'],gpu_allocated_MiB=peak['allocated_bytes']/2**20,gpu_reserved_MiB=peak['reserved_bytes']/2**20,cpu_process_peak_RSS_MiB=peak['cpu_peak_rss_bytes']/2**20,setup_seconds=r['setup_seconds'],point_input_bytes=storage['storage']['point_input_bytes'],normalized_map_bytes=storage['storage']['normalized_map_bytes'],latent_cache_bytes=r['cache_bytes'])
        rows.append(s)
    audit.append(dict(id=j['id'],validation=validation['status'],checkpoint_unchanged=True,observations=len(raw),source_sha256=r['source_sha256'],timings_sha256=hashlib.sha256((p/'timings.csv').read_bytes()).hexdigest()))
assert len(audit)==19
with (Q/'cost_SUMMARY.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(Q/'cost_SUMMARY.json').write_text(json.dumps(rows,indent=2))
(Q/'cost_COMPLETION_AUDIT.json').write_text(json.dumps(dict(status='complete',architectures=19,rows=len(rows),jobs=audit),indent=2))
print(json.dumps(dict(architectures=19,summary_rows=len(rows),total_observations=sum(x['observations'] for x in audit))))
