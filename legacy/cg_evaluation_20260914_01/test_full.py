import numpy as np
from full_evaluation import setup,load_block,sample,groups,all_jobs
from evaluation import T
for j in all_jobs():
    env=setup(j,'val')
    ds=env['ds'];d=load_block(env,0)
    idx=np.random.default_rng(T.stable_seed(42,ds.files[0].name)).choice(len(d['labels']),8192,replace=len(d['labels'])<8192)
    a=ds[0];b=sample(env,d,idx)
    for k in ['base','degraded','delta','restored','labels','hard','rel']:
        np.testing.assert_allclose(a[k],b[k],atol=1e-6,rtol=1e-6,err_msg=str((j['id'],k)))
    for k in ['coords','neighbor_indices','sub_idx','interp_idx']:
        for x,y in zip(a[k],b[k]):np.testing.assert_array_equal(x,y)
    chunks=list(groups(len(d['labels']),ds.files[0].name));ids=np.concatenate([x[:n] for x,n in chunks])
    np.testing.assert_array_equal(np.sort(ids),np.arange(len(d['labels'])))
    assert all(len(x)==8192 for x,n in chunks)
    print('PASS',j['id'],flush=True)
print('All 38 full-retained adapters passed validation-input and coverage checks.')
