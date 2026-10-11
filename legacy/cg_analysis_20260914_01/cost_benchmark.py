"""Read-only E3 benchmark. Every invocation is an isolated architecture process."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import sys,json,time,inspect,resource,platform,traceback,csv
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,'/root/autodl-tmp/cg_evaluation_20260914_01')
import full_evaluation as E
import controls as C
Q=Path(__file__).parent
MASKS=['valid_mask','structural_hard_mask','low_density_mask','high_gradient_mask','high_roughness_mask']

s=inspect.getsource(C.R.build_one_map)
s=s.replace(s.splitlines()[0],'def raster(d,resolution=.5):').replace('    d = np.load(npz_path, allow_pickle=True)\n','').replace('    labels = d["labels"].astype(np.int64)\n','')
s=s[:s.index('    # Label map:')]+s[s.index('    # Preliminary structural masks,'):]
s=s[:s.index('    # Per-block normalized maps')]+"    return maps_raw,np.stack([valid,structural_hard_mask,low_density_mask,high_gradient_mask,high_roughness_mask]).astype(np.float32)\n"
scope=dict(C.R.__dict__);exec(s,scope);raster=scope['raster']
def timer(row,key,fn):
    torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();row[key]=row.get(key,0)+(time.perf_counter()-t)*1000;return v
def raw_input(e,d,ids=None):
    if e['job']['dataset']=='DALES':
        r={k:d[k] for k in ['points','intensity','x0','y0','block_size']}
    else:r=dict(points=d['xyz'],intensity=d['intensity'].reshape(-1)*np.float32(65535),x0=d['x0'],y0=d['y0'],block_size=50.)
    if ids is not None:
        ids=np.unique(ids);r=dict(r,points=r['points'][ids],intensity=r['intensity'][ids])
    return r
def maps(e,d,ids,row):
    raw,mask=timer(row,'raster_ms',lambda:raster(raw_input(e,d,ids)))
    norm=timer(row,'normalization_ms',lambda:C.normalize(raw,C.STATS if e['job']['dataset']=='DALES' else e['stats']))
    return norm,mask
def read(e,i,route):
    p=e['ds'].files[i];v=e['job']['variant'];dset=e['job']['dataset']
    if dset=='DALES':
        with np.load(p,allow_pickle=True) as z:d=dict(z)
        d.update(xyz=d['points'].astype(np.float32),local=d['points_local'].astype(np.float32),scalar=d['features'].astype(np.float32))
        d['map']=dict(x0=d['x0'],y0=d['y0'],resolution=.5)
    else:
        d={k:np.load(p/(k+'.npy')).copy() for k in ['xyz','local','intensity','labels']};d['scalar']=d['intensity'];d['x0'],d['y0']=e['ds'].maps[p.name][2:]
        d['map']=np.zeros((9,100,100),np.float32);d['raster']=np.zeros((14,100,100),np.float32)
    if route=='cached' and v not in ['Base','PointNeXt_S','O_point','O_sampled']:
        mp=e['ds'].deg_map_dir/e['split']/(p.stem+'_25d.npz') if dset=='DALES' else Path(e['module'].PROTOCOL['root'])/'data/maps_25d_b50_r050_degraded_v3_medium_shared_v3'/e['split']/(p.name+'_25d.npz')
        with np.load(mp,allow_pickle=True) as z:m=dict(z)
        if dset=='DALES':d['map']=m
        else:d['map']=m['maps_norm'];d['raster']=np.concatenate([m['maps_norm'],np.stack([m[k] for k in MASKS])])
        if v.startswith('F'):
            d['latent']=np.load(e['cache'][p.name]);d['latent_meta']=dict(x0=float(d.get('x0',m['x0'])),y0=float(d.get('y0',m['y0'])),resolution=.5)
    return d
def install_map(e,d,norm,mask):
    if e['job']['dataset']=='DALES':d['map'].update(maps_norm=norm)
    else:d['map']=norm;d['raster']=np.concatenate([norm,mask])
def feature(e,d,norm,mask,row):
    x=timer(row,'h2d_ms',lambda:torch.from_numpy(np.concatenate([norm,mask])[None]).cuda())
    if e['job']['dataset']=='DALES':return timer(row,'upstream_ms',lambda:e['extract'](e['upstream'],x))
    timer(row,'upstream_ms',lambda:e['model'].unet(x));return e['model'].latent
def batch(e,d,ids,row):
    orig=E.T.build_randlanet_hierarchy
    def h(*a,**kw):return timer(row,'hierarchy_ms',lambda:orig(*a,**kw))
    E.T.build_randlanet_hierarchy=h

    sp,ct=E.sampled_prior,E.context
    E.sampled_prior=lambda block,choice,xyz:(E.T.sample_maps(xyz,d['map']),len(np.unique(choice)))
    E.context=lambda *a:d['map']
    try:return e['collate']([E.sample(e,d,ids)])
    finally:E.T.build_randlanet_hierarchy=orig;E.sampled_prior=sp;E.context=ct
def gpu(e,b,row):
    def move():
        nonlocal b
        if e['job']['variant']=='PointNeXt_S':return {'coords':[b['coords'][0].cuda()],'base':b['base'].cuda()}
        b=E.T.move_batch_to_device(b,torch.device('cuda'))
        for k in ['raster','cell']:
            if k in b:b[k]=b[k].cuda()
        return b
    return timer(row,'h2d_ms',move)
def forward(e,b,latent,row):
    m=e['model'];v=e['job']['variant']
    if v.startswith('F') and e['job']['dataset']!='DALES':
        z=latent.flatten(2).transpose(1,2);b=dict(b);b['delta']=m.projection(z.gather(1,b['cell'].unsqueeze(-1).expand(-1,-1,48)));return m.point(b)
    return m(b['coords'][0],b['base']) if v=='PointNeXt_S' else m(b)
@torch.no_grad()
def run(e,i,route,mode,verify=False):
    row={k:0. for k in ['read_ms','raster_ms','normalization_ms','hierarchy_ms','h2d_ms','upstream_ms','segmentation_ms']};t=time.perf_counter();d=timer(row,'read_ms',lambda:read(e,i,route));p=e['ds'].files[i];v=e['job']['variant'];n=len(d['labels'])
    groups=list(E.groups(n,p.name)) if mode=='full_retained' else [(np.random.default_rng(E.T.stable_seed(42,p.name)).choice(n,8192,replace=n<8192),8192)]
    latent=None
    if route=='online' and v in ['O_full','F_pre','F_rnd']:
        norm,mask=maps(e,d,None,row);install_map(e,d,norm,mask)
        if v.startswith('F'):latent=feature(e,d,norm,mask,row)
    elif v.startswith('F'):latent=timer(row,'h2d_ms',lambda:torch.from_numpy(d['latent'][None]).cuda())
    logits=[]
    for ids,valid in groups:
        if v=='O_sampled':norm,mask=maps(e,d,ids,row);install_map(e,d,norm,mask)
        if v.startswith('F') and e['job']['dataset']=='DALES':

            d['latent']=np.zeros((48,100,100),np.float32);d['latent_meta']=dict(x0=float(d['x0']),y0=float(d['y0']),resolution=.5)
        b=timer(row,'packing_including_hierarchy_ms',lambda:batch(e,d,ids,row));b=gpu(e,b,row)
        if v.startswith('F') and e['job']['dataset']=='DALES':
            xyz=d['xyz'][ids];ix=np.clip(np.floor((xyz[:,0]-d['x0'])/.5).astype(np.int64),0,99);iy=np.clip(np.floor((xyz[:,1]-d['y0'])/.5).astype(np.int64),0,99)
            cell=timer(row,'h2d_ms',lambda:torch.from_numpy((iy*100+ix)[None]).cuda());b['restored']=latent.flatten(2).transpose(1,2).gather(1,cell[:,:,None].expand(-1,-1,48))
        z=timer(row,'segmentation_ms',lambda:forward(e,b,latent,row))
        if verify:logits.append(z.cpu().numpy())
    row['packing_ms']=row.pop('packing_including_hierarchy_ms')-row['hierarchy_ms'];row.update(total_ms=(time.perf_counter()-t)*1000,points=n if mode=='full_retained' else 8192,groups=len(groups),file=p.name,route=route,mode=mode)
    return row,logits
def prepare(e,indices):
    v=e['job']['variant'];e['cache']={};rows=[]
    if not v.startswith('F'):return rows
    if e['job']['dataset']=='DALES':
        import frozen_unet_features as F
        manifest=json.loads((e['feature_root']/'feature_cache_manifest.json').read_text());ck=manifest['checkpoint'];assert E.sha(ck['path'])==ck['sha256'];state=torch.load(ck['path'],map_location='cpu',weights_only=False)
        e['upstream']=F.load_unet_module().UNet2D(14,9,48,float(state.get('args',{}).get('dropout',0))).cuda().eval().requires_grad_(False);e['upstream'].load_state_dict(state['model']);e['extract']=F.extract_decoder_features
        for i in indices:
            p=e['ds'].files[i];e['cache'][p.name]=e['feature_root']/e['features'][p.stem+'_25d.npz']['feature_path']
    else:
        folder=Q/'cost_cache'/e['job']['id']/e['split'];folder.mkdir(parents=True,exist_ok=True)
        for i in indices:
            d=E.load_block(e,i);p=e['ds'].files[i];r={};t=time.perf_counter();latent=feature(e,d,d['map'],d['raster'][9:],r);arr=timer(r,'d2h_ms',lambda:latent.cpu().numpy()[0]);fp=folder/(p.name+'.npy');timer(r,'cache_write_ms',lambda:np.save(fp,arr));r.update(file=p.name,total_ms=(time.perf_counter()-t)*1000,bytes=fp.stat().st_size);rows.append(r);e['cache'][p.name]=fp
    return rows
def main():
    job=next(j for j in E.all_jobs() if j['id']==sys.argv[1]);smoke='--validate' in sys.argv;out=Q/('cost_validation' if smoke else 'cost_results')/job['id'];out.mkdir(parents=True,exist_ok=True)
    try:
        t=time.perf_counter();e=E.setup(job,'val' if smoke else 'test');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        indices=sorted(range(len(e['ds'].files)),key=lambda i:E.sha_text(e['ds'].files[i].name)) if hasattr(E,'sha_text') else sorted(range(len(e['ds'].files)),key=lambda i:__import__('hashlib').sha256(e['ds'].files[i].name.encode()).hexdigest())
        indices=indices[:1 if smoke else min(33,len(indices))];preparation=prepare(e,indices);setup_seconds=time.perf_counter()-t
        storage=dict(point_input_bytes=0,normalized_map_bytes=0,latent_cache_bytes=sum(Path(p).stat().st_size for p in e['cache'].values()))
        for i in indices:
            p=e['ds'].files[i]
            storage['point_input_bytes']+=p.stat().st_size if p.is_file() else sum(x.stat().st_size for x in p.glob('*.npy'))
            mp=e['ds'].deg_map_dir/e['split']/(p.stem+'_25d.npz') if job['dataset']=='DALES' else Path(e['module'].PROTOCOL['root'])/'data/maps_25d_b50_r050_degraded_v3_medium_shared_v3'/e['split']/(p.name+'_25d.npz')
            storage['normalized_map_bytes']+=mp.stat().st_size
        if job['dataset']=='DALES' and job['variant'].startswith('F'):
            manifest=json.loads((e['feature_root']/'feature_cache_manifest.json').read_text());storage['historical_feature_export_seconds']=manifest.get('walltime_seconds');storage['historical_feature_export_counts']=manifest.get('counts');storage['manifest_sha256']=E.sha(e['feature_root']/'feature_cache_manifest.json')
        E.write(out/'storage_and_preparation.json',dict(storage=storage,cache_preparation=preparation,setup_seconds=setup_seconds))
        routes=['online'] if job['variant'] in ['Base','O_point','O_sampled','PointNeXt_S'] else ['online','cached']
        if smoke:
            i=indices[0];d=E.load_block(e,i);rr={};norm,mask=maps(e,d,None,rr);cached=d['map']['maps_norm'] if job['dataset']=='DALES' else d['map'];err=float(np.max(np.abs(norm-cached)));assert np.allclose(norm,cached,atol=2e-6,rtol=2e-6),(err,job)
            if job['variant'].startswith('F'):
                r1,z1=run(e,i,'online','sampled',True);r2,z2=run(e,i,'cached','sampled',True);error=float(np.max(np.abs(z1[0]-z2[0])));assert np.allclose(z1,z2,atol=2e-4,rtol=2e-4),error
            else:error=None

            ids=np.random.default_rng(E.T.stable_seed(42,e['ds'].files[i].name)).choice(len(d['labels']),8192,replace=len(d['labels'])<8192);b=e['collate']([E.sample(e,d,ids)]);b=gpu(e,b,{});m=e['model'];zref=m(b['coords'][0],b['base']) if job['variant']=='PointNeXt_S' else m(b)
            _,za=run(e,i,'online','sampled',True);referr=float(np.max(np.abs(zref.detach().cpu().numpy()-za[0])));assert np.allclose(zref.detach().cpu().numpy(),za[0],atol=2e-4,rtol=2e-4),referr
            E.write(out/'validation.json',dict(job=job,map_max_abs=err,online_cached_logits_max_abs=error,reference_logits_max_abs=referr,status='passed'));return
        assert (Q/'cost_validation'/job['id']/'validation.json').exists()
        rows=[];peaks=[]
        for mode in ['sampled','full_retained']:
            for route in routes:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                for rnd in (range(15) if mode=='sampled' else [5]):
                    for i in indices:
                        r,_=run(e,i,route,mode);r['round']=rnd-5
                        if rnd>=5:rows.append(r)
                    E.write(out/'status.json',dict(mode=mode,route=route,round=rnd,rows=len(rows)))
                    if rows:
                        with (out/'timings.partial.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
                peaks.append(dict(mode=mode,route=route,allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        with (out/'timings.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        summary=[]
        for mode in ['sampled','full_retained']:
            for route in routes:
                rr=[r for r in rows if r['mode']==mode and r['route']==route];r=dict(dataset=job['dataset'],variant=job['variant'],mode=mode,route=route,blocks=len(indices),observations=len(rr),blocks_per_second=1000*len(rr)/sum(x['total_ms'] for x in rr),points_per_second=1000*sum(x['points'] for x in rr)/sum(x['total_ms'] for x in rr))
                for k in rows[0]:
                    if k.endswith('_ms'):r[k+'_median']=float(np.median([x[k] for x in rr]));r[k+'_p95']=float(np.percentile([x[k] for x in rr],95))
                summary.append(r)
        E.write(out/'results.json',dict(job=job,summary=summary,peaks=peaks,setup_seconds=setup_seconds,cache_preparation=preparation,cache_bytes=sum(Path(p).stat().st_size for p in e['cache'].values()),trainable_parameters=sum(p.numel() for p in e['model'].parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in e['model'].parameters())+sum(p.numel() for p in e.get('upstream',torch.nn.Identity()).parameters()),hardware=dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,python=platform.python_version(),cpu=platform.processor(),threads=torch.get_num_threads(),tf32=False,precision='FP32'),files=[e['ds'].files[i].name for i in indices],source_sha256=E.sha(__file__),checkpoint_unchanged=E.sha(job['checkpoint'])==job['sha256']))
        E.write(out/'status.json',dict(status='complete'))
    except BaseException as exc:E.write(out/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    with torch.no_grad():main()


