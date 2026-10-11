"""Fixed disjoint retained-point groups; no interpolation or label propagation."""
import hashlib,inspect,json,sys,time
from pathlib import Path
import numpy as np,torch
from evaluation import Q,TRAIN,T,DATA,jobs,identity,construct,sha,write
from controls import ROOT,make_dataset,sampled_prior,point_prior
from external_controls import ROOTS,load_external,context
RESULTS=ROOT/'PROJECT_NAV_20260909/05_new_work/results'
def all_jobs():
    out=[]
    for j in jobs():
        d,v=identity(j);out.append(dict(id=j['path'].replace('/','__'),dataset=d,variant=v,new=j,checkpoint=str(TRAIN/j['path']/'best.pt'),sha256=j['checkpoints']['best.pt']['sha256']))
    old=json.loads((RESULTS/'medium_seeds43_44_test_20260910_01/EXPERIMENT_LOCK.json').read_text())['checkpoints']
    for j in old:
        if j['method'] in ['base','feature']:
            out.append(dict(id='old_DALES_'+j['name'],dataset='DALES',variant='Base' if j['method']=='base' else 'F_pre',checkpoint=j['checkpoint'],sha256=j['checkpoint_sha256'],config=j['config']))
    for s in [43,44]:
        for v,folder,name in [('O_full','observation_seeds43_44_20260910_01','observation'),('F_rnd','random_unet_control_medium_20260910_01','random_feature')]:
            p=RESULTS/folder/'train'/f's{s}_{name}'
            out.append(dict(id=f'old_DALES_{name}_{s}',dataset='DALES',variant=v,checkpoint=str(p/'best.pt'),sha256=sha(p/'best.pt'),config=str(p/'config.json')))
    for d in ['WHU','H3D']:
        for s in [43,44]:
            for method,v in [('base','Base'),('observation','O_full'),('feature','F_pre'),('random_feature','F_rnd')]:
                p=ROOTS[d]/'train'/f's{s}_{method}'
                out.append(dict(id=f'old_{d}_{method}_{s}',dataset=d,variant=v,method=method,checkpoint=str(p/'best.pt'),sha256=sha(p/'best.pt'),config=str(p/'config.json')))
    assert len(out)==38
    return sorted(out,key=lambda j:0 if j['variant'] in ['Base','O_full'] else 1 if 'new' in j else 2)
def setup(j,split):
    env=dict(job=j,split=split)
    if 'new' in j:
        ds,collate,model,nc,batch=construct(j['new'],split,'medium')
        env.update(ds=ds,collate=collate,model=model,nc=nc)
        if j['dataset']!='DALES':env['module'],_,env['stats']=load_external(j['dataset'])
        return env
    cfg=json.loads(Path(j['config']).read_text());d=j['dataset'];v=j['variant']
    if d=='DALES':
        root=Path('/root/autodl-tmp/als-transfer/data') if split=='test' else DATA
        trainer=T;cls=T._RESTORATION_CONTROL_ORIGINAL_DATASET
        modelcls=T.RandLASegmentationModel
        if v=='O_full':cls=make_dataset('full')
        if v in ['F_pre','F_rnd']:
            import train_feature_control as fa
            trainer=fa.load_trainer();seedfn=trainer.stable_seed;trainer.stable_seed=lambda s,n:seedfn(42,n)
            manifest=Path(cfg['feature_cache'])
            if split=='test':manifest=ROOT/'outputs_uq'/('frozen_test_20260909_feature_cache_medium_01/feature_cache_manifest.json' if v=='F_pre' else 'random_unet_control_medium_20260910_01/test/feature_cache_manifest.json')
            cls=fa.install_feature_dataset(trainer,manifest);modelcls=fa.FeatureFusionRandLASegmentationModel
            contents=json.loads(manifest.read_text());env['features']={e['degraded_filename']:e for e in contents['entries'] if e['split']==split};env['feature_root']=manifest.parent
        ds=cls(block_dir=root/'degraded_blocks_b50_v3_medium',clean_map_dir=Q/'FORBIDDEN',deg_map_dir=root/'maps_25d_b50_r050_degraded_v3_medium_shared_v3',restored_map_dir=Q/'FORBIDDEN',split=split,mode='base' if v=='Base' else 'restored_rgdi',num_points=8192,num_layers=4,num_neighbors=16,sub_sampling_ratio=[4]*4,degradation_tag='deg_v3_medium',seed=42)
        kw={k:cfg[k] for k in inspect.signature(T.RandLASegmentationModel.__init__).parameters if k in cfg}
        for k in ['sub_sampling_ratio','dim_output']:
            if isinstance(kw.get(k),str):kw[k]=[int(x) for x in kw[k].split(',')]
        kw.update(use_explicit_quality_modulation=not cfg['disable_explicit_quality_modulation'],use_reliability_descriptors=not cfg['disable_reliability_descriptors'],device=torch.device('cuda'))
        model=modelcls(**kw);collate=T.randla_collate;nc=8
    else:
        m,original,stats=load_external(d);ds=original(split,42,j['method'],'medium');model=m.Model(j['method']);collate=m.collate;nc=8 if d=='WHU' else 11;env.update(module=m,stats=stats)
    assert sha(j['checkpoint'])==j['sha256'];model.load_state_dict(torch.load(j['checkpoint'],map_location='cpu',weights_only=False)['model']);model.cuda().eval()
    env.update(ds=ds,model=model,collate=collate,nc=nc);return env
def load_block(env,i):
    ds=env['ds'];p=ds.files[i];j=env['job']
    if j['dataset']=='DALES':
        with np.load(p,allow_pickle=True) as z:d=dict(z)
        d['xyz']=d['points'].astype(np.float32);d['local']=d['points_local'].astype(np.float32);d['scalar']=d['features'].astype(np.float32)
        with np.load(ds.deg_map_dir/env['split']/(p.stem+'_25d.npz'),allow_pickle=True) as z:d['map']=dict(z)
        if j['variant'] in ['F_pre','F_rnd']:
            e=env['features'][p.stem+'_25d.npz'];fp=env['feature_root']/e['feature_path'];assert sha(fp)==e['feature_sha256'];d['latent']=np.load(fp);d['latent_meta']=e
    else:
        x=env['module'].points(p);d=dict(x);d['scalar']=x['intensity'];d['map'],d['raster'],d['x0'],d['y0']=ds.maps[p.name]
    return d
def sample(env,d,ids):
    j=env['job'];v=j['variant'];xyz=d['xyz'][ids].astype(np.float32);coords=d['local'][ids].astype(np.float32);f=coords.copy();f[:,:2]/=25;f[:,2]=np.clip(f[:,2]/20,-2,2);base=np.concatenate([f,d['scalar'][ids]],1).astype(np.float32)
    prior=np.zeros((8192,9),np.float32);restored=np.zeros_like(prior);delta=np.zeros_like(prior)
    if j['dataset']=='DALES':
        hard=d.get('point_hard_mask',np.zeros(len(d['labels']),bool))[ids].astype(np.float32)
        if v=='O_sampled':prior,_=sampled_prior(d,ids,xyz)
        elif v=='O_point':prior=point_prior(base)
        elif v not in ['Base','PointNeXt_S']:prior=T.sample_maps(xyz,d['map'])
        if v in ['F_pre','F_rnd']:
            e=d['latent_meta'];restored=T.sample_maps(xyz,dict(maps_norm=d['latent'],x0=e['x0'],y0=e['y0'],resolution=e['resolution']))
        elif v in ['O_sampled','O_point','O_full']:delta=prior.copy()
    else:
        ix=np.clip(np.floor((xyz[:,0]-d['x0'])/.5).astype(np.int64),0,99);iy=np.clip(np.floor((xyz[:,1]-d['y0'])/.5).astype(np.int64),0,99)
        raw=context(d,ids,d['x0'],d['y0'],env['stats']) if v=='O_sampled' else d['map']
        prior=raw[:,iy,ix].T.copy();delta=prior.copy();hard=d['raster'][10,iy,ix].astype(np.float32)
    hier=T.build_randlanet_hierarchy(coords,4,16,[4]*4)
    b=dict(base=base,degraded=prior,delta=delta,restored=restored,rel=np.zeros((8192,6),np.float32),labels=np.asarray(d['labels'][ids],np.int64),hard=hard,**hier,name='full_chunk')
    if j['dataset']!='DALES':b.update(raster=d['raster'],cell=iy*100+ix)
    return b
def groups(n,name):
    assert n>0
    order=np.random.default_rng(T.stable_seed(42,name)).permutation(n)
    for start in range(0,n,8192):
        ids=order[start:start+8192];yield np.resize(ids,8192),len(ids)
@torch.no_grad()
def evaluate_full(j):
    lock=json.loads((Q/'FULL_PROTOCOL_LOCK.json').read_text());assert lock['checkpoints'][j['id']]==j['sha256']
    out=Q/'full_retained'/j['id'];out.mkdir(parents=True,exist_ok=False)
    env=setup(j,'test');cm=np.zeros((env['nc'],env['nc']),np.int64);rows=[];started=time.time()
    for i,p in enumerate(env['ds'].files):
        d=load_block(env,i);n=len(d['labels']);pred=np.full(n,255,np.uint8);seen=np.zeros(n,bool)
        for ids,valid in groups(n,p.name):
            batch=env['collate']([sample(env,d,ids)])
            if j['variant']=='PointNeXt_S':z=env['model'](batch['coords'][0].cuda(),batch['base'].cuda())
            else:
                batch=T.move_batch_to_device(batch,torch.device('cuda'))
                if 'raster' in batch:batch['raster']=batch['raster'].cuda();batch['cell']=batch['cell'].cuda()
                z=env['model'](batch)
            assert torch.isfinite(z).all()
            keep=ids[:valid];assert not seen[keep].any();pred[keep]=z.argmax(-1)[0,:valid].cpu().numpy().astype(np.uint8);seen[keep]=True
        assert seen.all() and (pred<env['nc']).all()
        c=np.bincount((d['labels'].astype(np.int64)*env['nc']+pred).reshape(-1),minlength=env['nc']**2).reshape(env['nc'],env['nc']);assert c.sum()==n;cm+=c
        np.savez_compressed(out/(p.stem+'.npz'),prediction=pred,confusion=c)
        datahash=hashlib.sha256()
        for key in ['xyz','local','scalar','labels']:
            a=np.ascontiguousarray(d[key]);datahash.update(key.encode());datahash.update(str((a.dtype,a.shape)).encode());datahash.update(a.tobytes())
        mapdata=d['map']['maps_norm'] if j['dataset']=='DALES' else d['raster']
        datahash.update(np.ascontiguousarray(mapdata).tobytes())
        rows.append(dict(file=p.name,points=n,input_sha256=datahash.hexdigest(),confusion=c.tolist()))
        write(out/'status.json',dict(status='running',blocks=i+1,total_blocks=len(env['ds']),points=int(cm.sum()),seconds=time.time()-started))
    assert sha(j['checkpoint'])==j['sha256']
    write(out/'results.json',dict(job=j,metrics=T.metrics_from_confusion(cm),confusion=cm.tolist(),blocks=rows,points=int(cm.sum()),seconds=time.time()-started,protocol='all retained records counted once; fixed disjoint 8192 groups, padding discarded; predictions indexed in original record order; no smoothing'))
    write(out/'status.json',dict(status='complete',blocks=len(rows),points=int(cm.sum()),seconds=time.time()-started))
if __name__=='__main__':
    try:evaluate_full(next(j for j in all_jobs() if j['id']==sys.argv[1]))
    except BaseException as e:
        write(Q/'full_retained'/sys.argv[1]/'failure.json',dict(error=repr(e)));raise
