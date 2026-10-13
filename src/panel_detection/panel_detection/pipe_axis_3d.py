"""Lift an image-space pipe axis to a robust local 3D surface-axis estimate.

Requires depth registered to the color image and matching optical intrinsics.
The line follows the visible pipe surface; its point is not a cylinder center.
"""
import numpy as np
from .depth_utils import deproject_pixels_to_points


def estimate_pipe_axis_3d(result, depth_image, intrin, depth_scale=0.001, cfg=None):
    cfg = cfg or {}
    def invalid(reason, **quality):
        return dict(valid=False, reason=reason, point_xyz_m=None,
                    direction_xyz=None, endpoints_xyz_m=None, **quality)
    if result is None:
        return invalid('no_2d_axis')
    if depth_image is None or depth_image.ndim != 2 or not depth_image.size:
        return invalid('invalid_depth')
    if not np.isfinite([intrin.fx, intrin.fy, intrin.cx, intrin.cy, depth_scale]).all() or min(intrin.fx, intrin.fy, depth_scale) <= 0:
        return invalid('invalid_intrinsics_or_scale')
    h, w = depth_image.shape
    if intrin.width != w or intrin.height != h:
        return invalid('intrinsics_size_mismatch')
    axis = np.asarray(result.vector_xy, dtype=float)
    if axis.shape != (2,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        return invalid('invalid_2d_axis')
    axis /= np.linalg.norm(axis)
    perpendicular = np.array([-axis[1], axis[0]])
    origin = np.asarray(result.leak_point, dtype=float)
    half_length = max(1, int(cfg.get('half_length_px', 100)))
    stride = max(1, int(cfg.get('sample_stride_px', 2)))
    radius = max(0, int(cfg.get('strip_radius_px', 2)))
    minimum = max(3, int(cfg.get('min_points', 12)))
    lo, hi = float(cfg.get('min_depth_m', .10)), float(cfg.get('max_depth_m', 5.0))
    samples, depths, offsets = [], [], []
    rx1, ry1, rx2, ry2 = result.roi
    for t in range(-half_length, half_length + 1, stride):
        center = origin + t * axis
        if not (max(0,rx1) <= center[0] < min(w,rx2) and max(0,ry1) <= center[1] < min(h,ry2)):
            continue
        pixels = np.round(center + np.arange(-radius, radius+1)[:,None] * perpendicular).astype(int)
        valid = (pixels[:,0]>=max(0,rx1)) & (pixels[:,0]<min(w,rx2)) & (pixels[:,1]>=max(0,ry1)) & (pixels[:,1]<min(h,ry2))
        pixels = pixels[valid]
        z = depth_image[pixels[:,1], pixels[:,0]].astype(float) * depth_scale
        z = z[np.isfinite(z) & (z>=lo) & (z<=hi)]
        if len(z):
            samples.append(center);depths.append(float(np.median(z)));offsets.append(t)
    if len(depths) < minimum:
        return invalid('insufficient_depth', point_count=len(depths))
    pixels, z, t = np.asarray(samples), np.asarray(depths), np.asarray(offsets)
    near = np.abs(t) <= float(cfg.get('seed_radius_px', 12))
    if not np.any(near):
        return invalid('missing_depth_near_leak')
    seed = np.median(z[near])
    keep = np.abs(z-seed) <= float(cfg.get('depth_gate_m', .30))
    pixels, z, t = pixels[keep], z[keep], t[keep]
    if len(z) < minimum:
        return invalid('insufficient_local_depth', point_count=len(z))
    points = deproject_pixels_to_points(intrin, pixels, z)
    threshold = float(cfg.get('inlier_threshold_m', .008))
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(100):
        i,j = rng.choice(len(points),2,replace=False)
        if abs(t[i]-t[j]) < 20:
            continue
        direction = points[j]-points[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction /= norm
        residual = np.linalg.norm(np.cross(points-points[i],direction),axis=1)
        mask = residual <= threshold
        if mask.sum() > best.sum():best = mask
    ratio = float(best.mean())
    if best.sum() < minimum or ratio < float(cfg.get('min_inlier_ratio', .65)):
        return invalid('poor_line_support', point_count=len(points), inlier_ratio=ratio)
    cloud = points[best]
    centroid = cloud.mean(axis=0)
    _, singular, vh = np.linalg.svd(cloud-centroid,full_matrices=False)
    direction = vh[0]
    along = (cloud-centroid) @ direction
    span = float(np.ptp(along))
    rms = float(np.sqrt(np.mean(np.linalg.norm(np.cross(cloud-centroid,direction),axis=1)**2)))
    # Consistent sign: follows increasing image-axis parameter (no physical flow direction).
    trend = ((t[best]-np.mean(t[best]))[:,None] * (cloud-centroid)).sum(axis=0)
    if np.dot(direction,trend)<0:direction=-direction;along=-along
    if span < float(cfg.get('min_span_m', .04)) or np.ptp(t[best]) < 30:
        return invalid('short_axis_support', span_m=span)
    if rms > threshold or singular[0] < 3*max(singular[1],1e-9):
        return invalid('nonlinear_depth', rms_error_m=rms)
    # Require inliers near the leak, not a disconnected background line.
    if not np.any(np.abs(t[best])<=float(cfg.get('seed_radius_px',12))):
        return invalid('line_not_supported_near_leak')
    endpoints = centroid + np.array([along.min(),along.max()])[:,None]*direction
    return dict(valid=True, reason='ok', point_xyz_m=centroid.tolist(),
                direction_xyz=direction.tolist(), endpoints_xyz_m=endpoints.tolist(),
                point_count=len(points), inlier_count=int(best.sum()),
                inlier_ratio=ratio, rms_error_m=rms, span_m=span,
                source='registered_depth_line_fit', axis_is_undirected=True)
