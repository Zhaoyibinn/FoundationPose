# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import copy,json,uuid,joblib,os,sys
import scipy.spatial as spatial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
from datareader import *
from estimater import *
from gaussian_splat_renderer import GaussianSplatRenderer
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml


def as_trimesh(mesh_or_scene):
  if isinstance(mesh_or_scene, trimesh.Scene):
    meshes = [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if len(meshes)==0:
      raise RuntimeError('Loaded scene does not contain any Trimesh geometry')
    mesh = trimesh.util.concatenate(meshes)
  else:
    mesh = mesh_or_scene
  if not isinstance(mesh, trimesh.Trimesh):
    raise RuntimeError(f'Unsupported mesh type: {type(mesh)}')
  return mesh


def load_mesh_file(mesh_file, mesh_scale=1.0):
  mesh = as_trimesh(trimesh.load(mesh_file, process=False))
  if mesh_scale != 1.0:
    mesh.apply_scale(mesh_scale)
  logging.info(f'loaded mesh {mesh_file}, vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, extents={mesh.extents}')
  return mesh


def get_default_gggs_recon_dir(ob_id):
  if opt.gggs_recon_dir is not None:
    return opt.gggs_recon_dir
  return f'{opt.ref_view_dir}/{ob_id:06d}_sampled/sparse/pred_da3_pose/gggs_recon'


def get_gaussian_ply_file(ob_id):
  if opt.gaussian_ply_file is not None:
    return opt.gaussian_ply_file
  return f'{get_default_gggs_recon_dir(ob_id)}/points3D_masked.ply'


def get_gaussian_mesh_file(ob_id):
  if opt.gaussian_mesh_file is not None:
    return opt.gaussian_mesh_file
  return f'{get_default_gggs_recon_dir(ob_id)}/mesh_tsdf_masked.ply'


def configure_gaussian_render_backend(est, ob_id):
  if opt.render_backend != 'gaussian':
    return None

  import learning.training.predict_pose_refine as predict_pose_refine
  import learning.training.predict_score as predict_score

  ply_file = get_gaussian_ply_file(ob_id)
  if not os.path.exists(ply_file):
    raise FileNotFoundError(ply_file)
  renderer = GaussianSplatRenderer(
    ply_path=ply_file,
    scale=opt.gaussian_scale,
    center=est.model_center,
    kernel_size=opt.gaussian_kernel_size,
    znear=opt.gaussian_znear,
    zfar=opt.gaussian_zfar,
    save_dir=opt.gaussian_render_save_dir if opt.save_gaussian_renders else None,
    save_depth=opt.save_gaussian_depth,
  )
  predict_pose_refine.nvdiffrast_render = renderer.render
  predict_score.nvdiffrast_render = renderer.render
  logging.info(f'using Gaussian render backend: {ply_file}')
  return renderer


def load_object_mesh(reader, ob_id, use_reconstructed_mesh):
  if opt.render_backend == 'gaussian':
    mesh_file = get_gaussian_mesh_file(ob_id)
    if not os.path.exists(mesh_file):
      raise FileNotFoundError(mesh_file)
    return load_mesh_file(mesh_file, mesh_scale=opt.gaussian_scale)
  if opt.mesh_file is not None:
    return load_mesh_file(opt.mesh_file, mesh_scale=opt.mesh_scale)
  if use_reconstructed_mesh:
    mesh = reader.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
    if opt.mesh_scale != 1.0:
      mesh.apply_scale(opt.mesh_scale)
    return mesh
  mesh = reader.get_gt_mesh(ob_id)
  if opt.mesh_scale != 1.0:
    mesh.apply_scale(opt.mesh_scale)
  return mesh



def write_results(res):
  os.makedirs(opt.debug_dir, exist_ok=True)
  with open(f'{opt.debug_dir}/linemod_res.yml','w') as ff:
    yaml.safe_dump(make_yaml_dumpable(copy.deepcopy(res)), ff)


def ensure_runtime_output_dirs(debug_dir):
  os.makedirs(debug_dir, exist_ok=True)
  os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
  os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
  if opt.save_frame_inputs:
    os.makedirs(f'{debug_dir}/rgb', exist_ok=True)
    os.makedirs(f'{debug_dir}/mask', exist_ok=True)


def save_frame_outputs(reader, i_frame, ob_id, color, ob_mask, pose, est):
  debug_dir = est.debug_dir
  id_str = reader.id_strs[i_frame]
  ensure_runtime_output_dirs(debug_dir)

  np.savetxt(f'{debug_dir}/ob_in_cam/{id_str}.txt', pose.reshape(4,4))

  if opt.save_frame_inputs:
    imageio.imwrite(f'{debug_dir}/rgb/{id_str}.png', color)
    cv2.imwrite(f'{debug_dir}/mask/{id_str}.png', (ob_mask.astype(np.uint8)*255))

  if not opt.save_visualization:
    return

  to_origin = getattr(est, 'vis_to_origin', np.eye(4))
  bbox = getattr(est, 'vis_bbox', None)
  if bbox is None:
    extents = est.mesh_ori.extents
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  center_pose = pose @ np.linalg.inv(to_origin)
  vis = draw_posed_3d_box(reader.K, img=color.copy(), ob_in_cam=center_pose, bbox=bbox)
  axis_scale = opt.viz_axis_scale
  if axis_scale <= 0:
    axis_scale = max(float(np.max(bbox[1]-bbox[0]))*0.75, 0.03)
  vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=axis_scale, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
  imageio.imwrite(f'{debug_dir}/track_vis/{id_str}.png', vis)
  logging.info(f"saved frame outputs for {id_str} to {debug_dir}")


def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id)
    if mask is None:
      return None
    valid = mask>0
  elif detect_type=='detected':
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cosypose'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError
  return valid



def run_pose_estimation_worker(reader, i_frames, est:FoundationPose=None, debug=0, ob_id=None, device='cuda:0'):
  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()

  for i, i_frame in enumerate(i_frames):
    logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
    video_id = reader.get_video_id()
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    id_str = reader.id_strs[i_frame]
    H,W = color.shape[:2]

    debug_dir =est.debug_dir

    ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
    if ob_mask is None:
      logging.info("ob_mask not found, skip")
      result[video_id][id_str][ob_id] = np.eye(4)
      return result

    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)

    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
    logging.info(f"pose:\n{pose}")
    save_frame_outputs(reader=reader, i_frame=i_frame, ob_id=ob_id, color=color, ob_mask=ob_mask, pose=pose, est=est)

    if debug>=3:
      m = est.mesh_ori.copy()
      tmp = m.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')

    result[video_id][id_str][ob_id] = pose

  return result


def get_linemod_video_dir(ob_id):
  if opt.linemod_video_dir is not None:
    return opt.linemod_video_dir
  return f'{opt.linemod_dir}/lm_test_all/test/{ob_id:06d}'


def run_pose_estimation():
  wp.force_load(device='cuda')
  tmp_ob_id = opt.ob_id if opt.ob_id is not None else 2
  reader_tmp = LinemodReader(get_linemod_video_dir(tmp_ob_id), split=None)

  debug = opt.debug
  use_reconstructed_mesh = opt.use_reconstructed_mesh
  debug_dir = opt.debug_dir
  ensure_runtime_output_dirs(debug_dir)

  res = NestDict()
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)

  ob_ids = [opt.ob_id] if opt.ob_id is not None else reader_tmp.ob_ids
  for ob_id in ob_ids:
    ob_id = int(ob_id)
    mesh = load_object_mesh(reader_tmp, ob_id, use_reconstructed_mesh)
    symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]

    args = []

    video_dir = get_linemod_video_dir(ob_id)
    reader = LinemodReader(video_dir, split=None)
    video_id = reader.get_video_id()
    est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)
    configure_gaussian_render_backend(est, ob_id)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    est.vis_to_origin = to_origin
    est.vis_bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    for i in range(len(reader.color_files)):
      args.append((reader, [i], est, debug, ob_id, "cuda:0"))

    outs = []
    for arg in args:
      out = run_pose_estimation_worker(*arg)
      outs.append(out)
      for out_video_id in out:
        for id_str in out[out_video_id]:
          for out_ob_id in out[out_video_id][id_str]:
            res[out_video_id][id_str][out_ob_id] = out[out_video_id][id_str][out_ob_id]
      write_results(res)

    for out in outs:
      for out_video_id in out:
        for id_str in out[out_video_id]:
          for out_ob_id in out[out_video_id][id_str]:
            res[out_video_id][id_str][out_ob_id] = out[out_video_id][id_str][out_ob_id]

  write_results(res)


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--linemod_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD", help="linemod root dir")
  parser.add_argument('--linemod_video_dir', type=str, default=None, help="Direct path to one LINEMOD video folder, e.g. linemod/all/000001")
  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
  parser.add_argument('--ob_id', type=int, default=None, help="Only run one LINEMOD object id, e.g. 1 for ob_0000001")
  parser.add_argument('--render_backend', type=str, default='mesh', choices=['mesh', 'gaussian'], help="Render synthetic candidate views with mesh nvdiffrast or Gaussian splatting")
  parser.add_argument('--mesh_file', type=str, default=None, help="Optional explicit mesh file for FoundationPose geometry")
  parser.add_argument('--mesh_scale', type=float, default=1.0, help="Scale mesh vertices on load for mesh backend or explicit mesh_file")
  parser.add_argument('--gggs_recon_dir', type=str, default=None, help="GGG/Splatting reconstruction dir containing points3D_masked.ply and mesh_tsdf_masked.ply")
  parser.add_argument('--gaussian_ply_file', type=str, default=None, help="Gaussian PLY file. Defaults to <gggs_recon_dir>/points3D_masked.ply")
  parser.add_argument('--gaussian_mesh_file', type=str, default=None, help="Mesh used for FoundationPose geometry. Defaults to <gggs_recon_dir>/mesh_tsdf_masked.ply")
  parser.add_argument('--gaussian_scale', type=float, default=1.0, help="Scale Gaussian points and GGG mesh vertices on load")
  parser.add_argument('--gaussian_kernel_size', type=float, default=0.0, help="2D Gaussian filter kernel size passed to GGG renderer")
  parser.add_argument('--gaussian_znear', type=float, default=0.001)
  parser.add_argument('--gaussian_zfar', type=float, default=100.0)
  parser.add_argument('--save_gaussian_renders', type=int, default=0, help="Save every Gaussian-rendered candidate view to disk")
  parser.add_argument('--gaussian_render_save_dir', type=str, default=None, help="Output dir for saved Gaussian renders. Defaults to <debug_dir>/gaussian_renders")
  parser.add_argument('--save_gaussian_depth', type=int, default=1, help="Also save Gaussian-rendered depth PNG/NPY when saving renders")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--save_visualization', type=int, default=1, help="Save per-frame 3D bbox/axis overlay while running")
  parser.add_argument('--save_frame_inputs', type=int, default=1, help="Save per-frame rgb and mask while running")
  parser.add_argument('--viz_axis_scale', type=float, default=0.0, help="Axis length in meters for visualization; <=0 uses mesh size")
  opt = parser.parse_args()
  if opt.gaussian_render_save_dir is None:
    opt.gaussian_render_save_dir = os.path.join(opt.debug_dir, 'gaussian_renders')
  set_seed(0)

  detect_type = 'mask'   # mask / box / detected

  run_pose_estimation()
