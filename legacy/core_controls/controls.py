"""Isolated DALES controls; reuse the frozen raster mathematics and trainer."""
import inspect, json, sys, textwrap, importlib.util
from pathlib import Path
from functools import lru_cache
import numpy as np

ROOT=Path('/root/autodl-tmp/ALS-DiffBridge')
DATA=ROOT/'data/processed/dales_objects'
sys.path.insert(0,str(ROOT/'scripts/uq'))
import train_restoration_control as adapter
adapter.configure_runtime()
T=adapter.load_trainer()
original_seed=T.stable_seed
T.stable_seed=lambda seed,name: original_seed(42,name)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

R=module('cg_original_raster',ROOT/'scripts/preprocess/05_build_25d_maps_from_full_blocks.py')
N=module('cg_original_normalization',ROOT/'scripts/preprocess/14_apply_shared_25d_normalization_v3.py')
STATS=json.loads((ROOT/'outputs_v3/preprocessing/shared_25d_norm_v3.json').read_text())
N.validate_stats(STATS)
assert STATS['stats_sha256']==adapter.EXPECTED_NORMALIZATION_SHA256
normalize=N.normalize_raw_maps

src=inspect.getsource(R.build_one_map)
src=src.replace(src.splitlines()[0], 'def raw_map(d, resolution=0.5):')
src=src.replace('    d = np.load(npz_path, allow_pickle=True)\n','')
src=src.replace('    labels = d["labels"].astype(np.int64)\n','')
start=src.index('    # Label map:')
end=src.index('    # Raw physical maps.')
src=src[:start]+src[end:]
src=src[:src.index('    # Per-block normalized maps')]+ '    return maps_raw\n'
scope=dict(R.__dict__); exec(compile(src,'<cg_exact_raw_raster>','exec'),scope)
raw_map=scope['raw_map']

def point_prior(base):
    return np.concatenate([base,base,np.zeros((len(base),1),np.float32)],axis=1)

@lru_cache(maxsize=32)
def load_block(fp):
    with np.load(fp,allow_pickle=True) as z: return dict(z)

def sampled_prior(block, choice, pts):
    unique=np.unique(choice)
    d={k:block[k] for k in ['x0','y0','block_size']}
    d['points']=block['points'][unique]
    d['intensity']=block['intensity'][unique]
    maps=normalize(raw_map(d),STATS)
    metadata=dict(d,maps_norm=maps,resolution=np.float32(.5))
    return T.sample_maps(pts,metadata),len(unique)

def make_dataset(variant):
    if variant=='full': return adapter.install_control(T,'observation')
    if variant not in ['point','sampled']: raise ValueError(variant)
    original=T._RESTORATION_CONTROL_ORIGINAL_DATASET
    source=textwrap.dedent(inspect.getsource(original.__getitem__))
    source=source.replace('block = np.load(fp, allow_pickle=True)','block = load_block(fp)')
    begin=source.index('    if self.mode != "base":')
    end=source.index('    hierarchy =',begin)
    replacement='''    if self.variant == 'point':
        prior_degraded = point_prior(base)
    else:
        prior_degraded, self.last_support_count = sampled_prior(block, choice, pts)
    delta = prior_degraded.copy()

'''
    source=source[:begin]+replacement+source[end:]
    scope=dict(T.__dict__); scope.update(load_block=load_block,point_prior=point_prior,sampled_prior=sampled_prior)
    exec(compile(source,'<cg_control_dataset>','exec'),scope)
    cls=type('CG_'+variant,(original,),{'variant':variant,'__getitem__':scope['__getitem__']})
    return cls
