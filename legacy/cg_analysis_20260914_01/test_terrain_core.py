import unittest
import numpy as np
from terrain_core import median_grid, error_stats, slope_grid

class TerrainTests(unittest.TestCase):
 def test_median_and_empty_cells(self):
  xyz=np.array([[.1,.2,1],[.9,.7,3],[1.2,.1,9],[1.5,.5,11],[1.9,.9,13]])
  z,n=median_grid(xyz,0,0,size=2)
  np.testing.assert_equal(n,[2,3,0,0]);np.testing.assert_allclose(z[:2],[2,11]);self.assertTrue(np.isnan(z[2:]).all())
 def test_empty_grid(self):
  z,n=median_grid(np.empty((0,3)),0,0,size=2);self.assertTrue(np.isnan(z).all());self.assertEqual(n.sum(),0)
 def test_coverage_not_hidden_by_common_cells(self):
  ref=np.array([0.,0.,0.,np.nan]);pred=np.array([0.,2.,np.nan,9.])
  r=error_stats(ref,pred)
  self.assertEqual(r['reference_cells'],3);self.assertEqual(r['valid_cells'],2);self.assertAlmostEqual(r['coverage'],2/3)
  self.assertEqual(r['false_positive_cells'],1);self.assertEqual(r['mae_m'],1);self.assertAlmostEqual(r['rmse_m'],2**.5)
 def test_slope_of_plane_and_invalid_neighbors(self):
  y,x=np.mgrid[:5,:5];z=(2*x+y).astype(float);s=slope_grid(z.flatten(),size=5).reshape(5,5)
  self.assertAlmostEqual(s[2,2],np.degrees(np.arctan(np.sqrt(5))))
  z[2,1]=np.nan;s=slope_grid(z.flatten(),size=5).reshape(5,5);self.assertTrue(np.isnan(s[2,2]))
if __name__=='__main__':unittest.main()
