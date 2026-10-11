import unittest
import numpy as np
from controls import raw_map, normalize, point_prior, sampled_prior, make_dataset, T, DATA, STATS

class ControlsTest(unittest.TestCase):
    def test_full_raster_regression(self):
        for split in ['train', 'val']:
            for fp in sorted((DATA/'degraded_blocks_b50_v3_medium'/split).glob('*.npz'))[:3]:
                with np.load(fp, allow_pickle=True) as z:
                    d = dict(z)
                with np.load(DATA/'maps_25d_b50_r050_degraded_v3_medium_shared_v3'/split/(fp.stem+'_25d.npz')) as z:
                    expected = z['maps_norm']
                got = normalize(raw_map(d), STATS)
                np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)

    def test_point_mapping(self):
        x = np.arange(12, dtype=np.float32).reshape(3,4)
        y = point_prior(x)
        np.testing.assert_array_equal(y[:,:4], x)
        np.testing.assert_array_equal(y[:,4:8], x)
        np.testing.assert_array_equal(y[:,8], 0)

    def test_no_unselected_support_or_duplicate_density(self):
        fp=next((DATA/'degraded_blocks_b50_v3_medium'/'val').glob('*.npz'))
        with np.load(fp,allow_pickle=True) as z: d=dict(z)
        choice=np.arange(1000);pts=d['points'][choice]
        before,n=sampled_prior(d,choice,pts)
        duplicated,n2=sampled_prior(d,np.tile(choice,2),pts)
        np.testing.assert_array_equal(before,duplicated)
        self.assertEqual(n,n2)
        d['points']=d['points'].copy();d['intensity']=d['intensity'].copy()
        d['points'][1000:,2]+=100;d['intensity'][1000:]=0
        after,_=sampled_prior(d,choice,pts)
        np.testing.assert_array_equal(before,after)

    def test_dataset_support_and_sampling(self):
        kwargs=dict(block_dir=DATA/'degraded_blocks_b50_v3_medium',clean_map_dir=DATA/'FORBIDDEN',deg_map_dir=DATA/'maps_25d_b50_r050_degraded_v3_medium_shared_v3',restored_map_dir=DATA/'FORBIDDEN',split='val',mode='restored_rgdi',num_points=8192,num_layers=4,num_neighbors=16,sub_sampling_ratio=[4]*4,degradation_tag='deg_v3_medium',max_files=1,seed=43)
        old=make_dataset('full')(**kwargs)[0]
        for variant in ['sampled','point']:
            ds=make_dataset(variant)(**kwargs); new=ds[0]
            for k in ['base','labels','hard']:
                np.testing.assert_array_equal(old[k],new[k])
            for k in ['coords','neighbor_indices','sub_idx','interp_idx']:
                for a,b in zip(old[k],new[k]): np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(new['degraded'],new['delta'])
            self.assertTrue(np.isfinite(new['degraded']).all())
            self.assertEqual(new['rel'].sum(),0)
            if variant=='sampled':
                self.assertLessEqual(ds.last_support_count,8192)
                self.assertFalse(np.array_equal(old['degraded'],new['degraded']))

if __name__=='__main__': unittest.main()
