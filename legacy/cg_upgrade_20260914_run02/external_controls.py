import inspect,json,textwrap
from pathlib import Path
import numpy as np
from controls import module,raw_map,normalize,N
ROOTS={'WHU':Path('/root/autodl-tmp/whu25d/completion_v2'),'H3D':Path('/root/autodl-tmp/h3d25d/run_v2')}
def context(d,idx,x0,y0,stats):
    selected=np.unique(idx)
    raw=raw_map(dict(points=np.asarray(d['xyz'][selected],np.float32),intensity=np.asarray(d['intensity'][selected],np.float32).reshape(-1)*np.float32(65535),x0=x0,y0=y0,block_size=50.0))
    return normalize(raw,stats)
def load_external(name):
    root=ROOTS[name];m=module('cg_external_'+name,root/'train_complete.py')
    stats=json.loads((root/'normalization.json').read_text());N.validate_stats(stats)
    original=m.Points
    source=textwrap.dedent(inspect.getsource(original.__getitem__))
    needle='prior=raw[:,iy,ix].T.copy();hier='
    assert source.count(needle)==1
    source=source.replace(needle,'raw=context(d,idx,x0,y0,stats);prior=raw[:,iy,ix].T.copy();hier=')
    scope=dict(m.__dict__);scope.update(context=context,stats=stats)
    exec(compile(source,'<cg_external_sampled>','exec'),scope)
    m.Points=type('CGSampled_'+name,(original,),{'__getitem__':scope['__getitem__']})
    return m,original,stats
