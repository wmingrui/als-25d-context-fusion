"""Ground-cell statistics. Empty cells stay NaN; no interpolation or smoothing."""
import numpy as np

def median_grid(xyz,x0,y0,size=50,resolution=1.):
 xyz=np.asarray(xyz,dtype=np.float64);z=np.full(size*size,np.nan);n=np.zeros(size*size,dtype=np.int64)
 if not len(xyz):return z,n
 assert xyz.shape[1]==3 and np.isfinite(xyz).all()
 ij=np.floor((xyz[:,:2]-[x0,y0])/resolution).astype(np.int64)

 assert np.all(xyz[:,:2]>=[x0-1e-4,y0-1e-4]) and np.all(xyz[:,:2]<=[x0+size*resolution+1e-4,y0+size*resolution+1e-4])
 ij=np.clip(ij,0,size-1);flat=ij[:,1]*size+ij[:,0]
 order=np.lexsort((xyz[:,2],flat));height=xyz[order,2]
 n=np.bincount(flat,minlength=size*size);end=np.cumsum(n);start=end-n;valid=n>0
 z[valid]=(height[start[valid]+(n[valid]-1)//2]+height[start[valid]+n[valid]//2])/2
 return z,n

def slope_grid(z,size=50,resolution=1.):
 a=np.asarray(z).reshape(size,size);out=np.full_like(a,np.nan,dtype=float)
 dx=(a[1:-1,2:]-a[1:-1,:-2])/(2*resolution);dy=(a[2:,1:-1]-a[:-2,1:-1])/(2*resolution)
 v=np.isfinite(a[1:-1,1:-1])&np.isfinite(dx)&np.isfinite(dy)
 b=np.degrees(np.arctan(np.hypot(dx,dy)));out[1:-1,1:-1]=np.where(v,b,np.nan)
 return out.ravel()

def error_stats(reference,prediction,mask=None):
 ref=np.asarray(reference);pr=np.asarray(prediction);base=np.ones(ref.shape,bool) if mask is None else np.asarray(mask,bool)
 rv=np.isfinite(ref)&base;pv=np.isfinite(pr)&base;valid=rv&pv;err=pr[valid]-ref[valid];abs_err=np.abs(err)
 fp=pv&~np.isfinite(ref);nonref=base&~np.isfinite(ref)
 return dict(reference_cells=int(rv.sum()),valid_cells=int(valid.sum()),coverage=float(valid.sum()/rv.sum()) if rv.any() else None,
  false_positive_cells=int(fp.sum()),nonreference_cells=int(nonref.sum()),false_positive_occupancy=float(fp.sum()/nonref.sum()) if nonref.any() else None,
  mae_m=float(abs_err.mean()) if len(err) else None,rmse_m=float(np.sqrt(np.mean(err**2))) if len(err) else None,
  median_abs_error_m=float(np.median(abs_err)) if len(err) else None,p95_abs_error_m=float(np.percentile(abs_err,95)) if len(err) else None,
  bias_m=float(err.mean()) if len(err) else None)
