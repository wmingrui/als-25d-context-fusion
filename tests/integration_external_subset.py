"""Check two validation blocks per external dataset without preloading every split."""
import argparse
from pathlib import Path
import sys
import tempfile
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.workspace/'cg_upgrade_20260914_run02'))
    from external_controls import load_external,context
    for name in ['WHU','H3D']:
        m,original,stats=load_external(name)
        with tempfile.TemporaryDirectory(prefix='als_external_check_') as temp:
            original_data=m.D
            subset=Path(temp)
            cache=subset/'point_cache/medium/val';cache.mkdir(parents=True)
            selected=sorted((original_data/'point_cache/medium/val').iterdir())[:2]
            assert len(selected)==2, name+' requires at least two validation blocks'
            for p in selected:(cache/p.name).symlink_to(p,target_is_directory=True)
            m.D=subset
            try:
                old=original('val',43,'observation')
                new=m.Points('val',43,'observation')
                for i in range(2):
                    p=old.files[i];d=m.points(p)
                    raw,_,x0,y0=old.maps[p.name]
                    np.testing.assert_allclose(context(d,np.arange(len(d['xyz'])),x0,y0,stats),raw,atol=2e-6,rtol=2e-6)
                    a,b=old[i],new[i]
                    for key in ['base','labels','hard']:np.testing.assert_array_equal(a[key],b[key])
                    for key in ['coords','neighbor_indices','sub_idx','interp_idx']:
                        for x,y in zip(a[key],b[key]):np.testing.assert_array_equal(x,y)
                    assert not np.array_equal(a['degraded'],b['degraded'])
                    assert np.isfinite(b['degraded']).all()
                print(name+': passed two validation blocks',flush=True)
            finally:
                m.D=original_data


if __name__=='__main__':
    main()
