import unittest,numpy as np
from external_controls import load_external, context
class ExternalTest(unittest.TestCase):
    def test_exact_maps_and_sampling(self):
        for name in ['WHU','H3D']:
            m,original,stats=load_external(name)
            for split in ['train','val']:
                old=original(split,43,'observation');new=m.Points(split,43,'observation')
                for i in [0,1]:
                    p=old.files[i];d=m.points(p);raw,ras,x0,y0=old.maps[p.name]
                    got=context(d,np.arange(len(d['xyz'])),x0,y0,stats)
                    np.testing.assert_allclose(got,raw,atol=2e-6,rtol=2e-6,err_msg=name)
                    a=old[i];b=new[i]
                    for key in ['base','labels','hard']:
                        np.testing.assert_array_equal(a[key],b[key])
                    for key in ['coords','neighbor_indices','sub_idx','interp_idx']:
                        for x,y in zip(a[key],b[key]):np.testing.assert_array_equal(x,y)
                    self.assertFalse(np.array_equal(a['degraded'],b['degraded']))
                    self.assertTrue(np.isfinite(b['degraded']).all())
if __name__=='__main__':unittest.main()
