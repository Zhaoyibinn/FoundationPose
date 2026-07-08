# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F


def _parse_ply_header(f):
  header = []
  while True:
    line = f.readline()
    if line == b'':
      raise ValueError('Unexpected EOF while reading PLY header')
    text = line.decode('ascii').strip()
    header.append(text)
    if text == 'end_header':
      break

  fmt = None
  vertex_count = None
  vertex_props = []
  in_vertex = False
  for line in header:
    parts = line.split()
    if not parts:
      continue
    if parts[0] == 'format':
      fmt = parts[1]
    elif parts[:2] == ['element', 'vertex']:
      vertex_count = int(parts[2])
      in_vertex = True
    elif parts[0] == 'element':
      in_vertex = False
    elif in_vertex and parts[0] == 'property':
      if len(parts) != 3 or parts[1] not in ('float', 'float32'):
        raise ValueError('Only scalar float vertex properties are supported for Gaussian PLY')
      vertex_props.append(parts[2])

  if fmt not in ('binary_little_endian', 'ascii'):
    raise ValueError('Unsupported PLY format: %s' % fmt)
  if vertex_count is None:
    raise ValueError('PLY header does not define element vertex')
  return fmt, vertex_count, vertex_props


def _read_gaussian_ply(path):
  with open(path, 'rb') as f:
    fmt, vertex_count, props = _parse_ply_header(f)
    if fmt == 'binary_little_endian':
      dtype = np.dtype([(name, '<f4') for name in props])
      data = np.fromfile(f, dtype=dtype, count=vertex_count)
    else:
      rows = []
      for _ in range(vertex_count):
        rows.append([float(v) for v in f.readline().decode('ascii').split()])
      data = np.asarray(rows, dtype=np.float32)
      dtype = np.dtype([(name, '<f4') for name in props])
      structured = np.empty(vertex_count, dtype=dtype)
      for i, name in enumerate(props):
        structured[name] = data[:, i]
      data = structured

  prop_set = set(props)
  for name in ('x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity'):
    if name not in prop_set:
      raise ValueError('Missing required Gaussian PLY property: %s' % name)

  xyz = np.stack([data[name] for name in ('x', 'y', 'z')], axis=1).astype(np.float32)
  features_dc = np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=1).astype(np.float32)
  features_dc = features_dc[:, None, :]
  opacity = data['opacity'].astype(np.float32)[:, None]

  rest_names = sorted([name for name in props if name.startswith('f_rest_')], key=lambda x: int(x.split('_')[-1]))
  if rest_names:
    if len(rest_names) % 3 != 0:
      raise ValueError('Invalid f_rest channel count: %d' % len(rest_names))
    sh_coeffs = len(rest_names) // 3
    sh_degree = int(math.sqrt(sh_coeffs + 1) - 1)
    if 3 * ((sh_degree + 1) ** 2 - 1) != len(rest_names):
      raise ValueError('f_rest channel count does not match a SH degree: %d' % len(rest_names))
    flat = np.stack([data[name] for name in rest_names], axis=1).astype(np.float32)
    features_rest = flat.reshape(vertex_count, 3, sh_coeffs).transpose(0, 2, 1).copy()
  else:
    sh_degree = 0
    features_rest = np.zeros((vertex_count, 0, 3), dtype=np.float32)

  scale_names = sorted([name for name in props if name.startswith('scale_')], key=lambda x: int(x.split('_')[-1]))
  rot_names = sorted([name for name in props if name.startswith('rot_')], key=lambda x: int(x.split('_')[-1]))
  if not scale_names or not rot_names:
    raise ValueError('Gaussian PLY must contain scale_* and rot_* properties')
  scales = np.stack([data[name] for name in scale_names], axis=1).astype(np.float32)
  rotations = np.stack([data[name] for name in rot_names], axis=1).astype(np.float32)

  if 'filter_3D' in prop_set:
    filter_3d = data['filter_3D'].astype(np.float32)[:, None]
  else:
    filter_3d = np.zeros((vertex_count, 1), dtype=np.float32)

  return {
    'xyz': xyz,
    'features_dc': features_dc,
    'features_rest': features_rest,
    'opacity': opacity,
    'scales': scales,
    'rotations': rotations,
    'filter_3d': filter_3d,
    'sh_degree': sh_degree,
  }


class GaussianPlyModel:
  def __init__(self, ply_path, scale=1.0, center=None, device='cuda'):
    data = _read_gaussian_ply(os.path.expanduser(ply_path))
    xyz = data['xyz'] * float(scale)
    if center is not None:
      xyz = xyz - np.asarray(center, dtype=np.float32).reshape(1, 3)

    self.device = torch.device(device)
    self.max_sh_degree = data['sh_degree']
    self.active_sh_degree = data['sh_degree']
    self.max_sg_degree = 0
    self.active_sg_degree = 0
    self.scale = float(scale)

    self._xyz = self._tensor(xyz)
    self._features_dc = self._tensor(data['features_dc'])
    self._features_rest = self._tensor(data['features_rest'])
    self._opacity = self._tensor(data['opacity'])
    self._scaling = self._tensor(data['scales'])
    self._rotation = self._tensor(data['rotations'])
    self.filter_3D = self._tensor(data['filter_3d'] * float(scale))

    n = xyz.shape[0]
    self._sg_axis = torch.empty((n, 0, 3), dtype=torch.float32, device=self.device)
    self._sg_sharpness = torch.empty((n, 0), dtype=torch.float32, device=self.device)
    self._sg_color = torch.empty((n, 0, 3), dtype=torch.float32, device=self.device)

  def _tensor(self, array):
    return torch.as_tensor(array, dtype=torch.float32, device=self.device)

  @property
  def get_xyz(self):
    return self._xyz

  @property
  def get_scaling(self):
    return torch.exp(self._scaling) * self.scale

  @property
  def get_rotation(self):
    return F.normalize(self._rotation, dim=1)

  @property
  def get_features(self):
    return torch.cat((self._features_dc, self._features_rest), dim=1)

  @property
  def get_sg_axis(self):
    return self._sg_axis

  @property
  def get_sg_sharpness(self):
    return self._sg_sharpness

  @property
  def get_sg_color(self):
    return self._sg_color

  @property
  def get_scaling_n_opacity_with_3D_filter(self):
    opacity = torch.sigmoid(self._opacity)
    scales = self.get_scaling
    scales_square = torch.square(scales)
    det1 = scales_square.prod(dim=1)
    scales_after_square = scales_square + torch.square(self.filter_3D)
    det2 = scales_after_square.prod(dim=1)
    coef = det1.sqrt() * det2.rsqrt()
    return scales_after_square.sqrt(), opacity * coef[..., None]


def _projection_from_intrinsics(K, height, width, znear, zfar):
  fx = K[0, 0]
  fy = K[1, 1]
  cx = K[0, 2]
  cy = K[1, 2]
  P = torch.zeros((4, 4), dtype=torch.float32, device=K.device)
  P[0, 0] = 2.0 * fx / width
  P[1, 1] = 2.0 * fy / height
  P[0, 2] = (2.0 * cx - width) / width
  P[1, 2] = (2.0 * cy - height) / height
  P[2, 2] = zfar / (zfar - znear)
  P[2, 3] = -(zfar * znear) / (zfar - znear)
  P[3, 2] = 1.0
  return P


class _MiniCamera:
  def __init__(self, width, height, K, world_to_camera, znear, zfar):
    self.image_width = int(width)
    self.image_height = int(height)
    self.znear = znear
    self.zfar = zfar
    self.Fx = float(K[0, 0])
    self.Fy = float(K[1, 1])
    self.Cx = float(K[0, 2])
    self.Cy = float(K[1, 2])
    self.FoVx = 2.0 * math.atan(float(width) / (2.0 * self.Fx))
    self.FoVy = 2.0 * math.atan(float(height) / (2.0 * self.Fy))
    self.world_view_transform = world_to_camera.transpose(0, 1).contiguous()
    projection = _projection_from_intrinsics(K, float(height), float(width), znear, zfar)
    self.projection_matrix = projection.transpose(0, 1).contiguous()
    self.full_proj_transform = (
      self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    self.camera_center = torch.inverse(self.world_view_transform)[3, :3]


class GaussianSplatRenderer:
  def __init__(self, ply_path, scale=1.0, center=None, kernel_size=0.0, znear=0.001, zfar=100.0, bg_color=(0, 0, 0), save_dir=None, save_depth=True):
    from gggs_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    self.raster_settings_cls = GaussianRasterizationSettings
    self.rasterizer_cls = GaussianRasterizer
    self.gaussians = GaussianPlyModel(ply_path, scale=scale, center=center)
    self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    self.kernel_size = float(kernel_size)
    self.znear = float(znear)
    self.zfar = float(zfar)
    self.save_dir = save_dir
    self.save_depth = bool(save_depth)
    self.call_index = 0
    if self.save_dir is not None:
      os.makedirs(os.path.join(self.save_dir, 'rgb'), exist_ok=True)
      if self.save_depth:
        os.makedirs(os.path.join(self.save_dir, 'depth_mm'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'depth_npy'), exist_ok=True)
      os.makedirs(os.path.join(self.save_dir, 'meta'), exist_ok=True)

  def _render_K_for_bbox(self, K, H, W, bbox, output_size):
    K_render = K.clone()
    out_h = int(output_size[0])
    out_w = int(output_size[1])
    if bbox is None:
      K_render[0, :] *= float(out_w) / float(W)
      K_render[1, :] *= float(out_h) / float(H)
      return K_render, out_h, out_w

    left, top, right, bottom = bbox
    crop_w = torch.clamp(right - left, min=1.0)
    crop_h = torch.clamp(bottom - top, min=1.0)
    K_render[0, 0] = K[0, 0] * float(out_w) / crop_w
    K_render[1, 1] = K[1, 1] * float(out_h) / crop_h
    K_render[0, 2] = (K[0, 2] - left) * float(out_w) / crop_w
    K_render[1, 2] = (K[1, 2] - top) * float(out_h) / crop_h
    return K_render, out_h, out_w

  @staticmethod
  def _xyz_from_depth(depth, K):
    h, w = depth.shape
    ys, xs = torch.meshgrid(
      torch.arange(h, dtype=torch.float32, device=depth.device),
      torch.arange(w, dtype=torch.float32, device=depth.device),
      indexing='ij',
    )
    z = depth
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    return torch.stack((x, y, z), dim=-1)

  def _save_view(self, call_index, view_index, color, depth, K, pose):
    if self.save_dir is None:
      return
    stem = f'call_{call_index:06d}_view_{view_index:06d}'
    rgb = (color.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(self.save_dir, 'rgb', f'{stem}.png'), rgb)
    if self.save_depth:
      depth_np = depth.detach().cpu().numpy().astype(np.float32)
      np.save(os.path.join(self.save_dir, 'depth_npy', f'{stem}.npy'), depth_np)
      depth_mm = np.nan_to_num(depth_np * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
      depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
      imageio.imwrite(os.path.join(self.save_dir, 'depth_mm', f'{stem}.png'), depth_mm)
    np.savez_compressed(
      os.path.join(self.save_dir, 'meta', f'{stem}.npz'),
      K=K.detach().cpu().numpy().astype(np.float32),
      ob_in_cam=pose.detach().cpu().numpy().astype(np.float32),
    )

  def _render_gaussians(self, camera, require_depth=True):
    pc = self.gaussians
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(
      pc.get_xyz,
      dtype=pc.get_xyz.dtype,
      requires_grad=True,
      device='cuda',
    )

    raster_settings = self.raster_settings_cls(
      image_height=int(camera.image_height),
      image_width=int(camera.image_width),
      tanfovx=tanfovx,
      tanfovy=tanfovy,
      kernel_size=self.kernel_size,
      bg=self.background,
      scale_modifier=1.0,
      viewmatrix=camera.world_view_transform,
      projmatrix=camera.full_proj_transform,
      sh_degree=pc.active_sh_degree,
      sg_degree=pc.active_sg_degree,
      campos=camera.camera_center,
      prefiltered=False,
      require_depth=require_depth,
      debug=False,
    )
    rasterizer = self.rasterizer_cls(raster_settings=raster_settings)
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rendered_image, radii, median_depth, alpha, normal = rasterizer(
      means3D=pc.get_xyz,
      means2D=screenspace_points,
      shs=pc.get_features,
      sg_axis=pc.get_sg_axis,
      sg_sharpness=pc.get_sg_sharpness,
      sg_color=pc.get_sg_color,
      colors_precomp=None,
      opacities=opacity,
      scales=scales,
      rotations=pc.get_rotation,
      cov3Ds_precomp=None,
    )
    return {
      'render': rendered_image,
      'mask': alpha,
      'median_depth': median_depth,
      'visibility_filter': radii > 0,
      'radii': radii,
      'normal': normal,
    }

  @torch.no_grad()
  def render(self, K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None, output_size=None, use_light=False, light_color=None, light_dir=np.array([0,0,1]), light_pos=np.array([0,0,0]), w_ambient=0.8, w_diffuse=0.5, extra={}):
    del glctx, context, mesh_tensors, mesh, projection_mat, use_light, light_color, light_dir, light_pos, w_ambient, w_diffuse

    if output_size is None:
      output_size = (H, W)
    K_t = torch.as_tensor(K, dtype=torch.float32, device='cuda')
    poses = torch.as_tensor(ob_in_cams, dtype=torch.float32, device='cuda')
    if bbox2d is not None:
      bbox2d_t = torch.as_tensor(bbox2d, dtype=torch.float32, device='cuda')
    else:
      bbox2d_t = None

    call_index = self.call_index
    self.call_index += 1

    colors = []
    depths = []
    normals = []
    xyz_maps = []
    for i in range(len(poses)):
      bbox = bbox2d_t[i] if bbox2d_t is not None else None
      K_render, out_h, out_w = self._render_K_for_bbox(K_t, H, W, bbox, output_size)
      camera = _MiniCamera(
        width=out_w,
        height=out_h,
        K=K_render,
        world_to_camera=poses[i],
        znear=self.znear,
        zfar=self.zfar,
      )
      pkg = self._render_gaussians(camera, require_depth=True)
      color = pkg['render'].permute(1, 2, 0).contiguous().clamp(0.0, 1.0)
      depth = pkg['median_depth']
      if depth.ndim == 3:
        depth = depth[0]
      alpha = pkg.get('mask', None)
      if alpha is not None:
        if alpha.ndim == 3:
          alpha = alpha[0]
        depth = torch.where(alpha > 1e-4, depth, torch.zeros_like(depth))
      colors.append(color)
      depths.append(depth)
      xyz_maps.append(self._xyz_from_depth(depth, K_render))
      self._save_view(call_index, i, color, depth, K_render, poses[i])
      if get_normal:
        normal = pkg.get('normal', None)
        if normal is None:
          normal = torch.zeros((3, out_h, out_w), dtype=torch.float32, device='cuda')
        normals.append(normal.permute(1, 2, 0).contiguous())

    extra['xyz_map'] = torch.stack(xyz_maps, dim=0)
    normal_batch = torch.stack(normals, dim=0) if get_normal else None
    return torch.stack(colors, dim=0), torch.stack(depths, dim=0), normal_batch
