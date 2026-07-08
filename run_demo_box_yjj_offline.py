# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
import os
import shutil

import cv2
import imageio
import numpy as np
import trimesh


def as_trimesh(mesh_or_scene):
  if isinstance(mesh_or_scene, trimesh.Scene):
    geometries = [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not geometries:
      raise RuntimeError('No mesh geometry found in scene')
    return trimesh.util.concatenate(geometries)
  if isinstance(mesh_or_scene, trimesh.Trimesh):
    return mesh_or_scene
  raise TypeError(f'Unsupported mesh type: {type(mesh_or_scene)}')


def simplify_mesh_for_inference(mesh, max_faces, aggression):
  if max_faces <= 0 or len(mesh.faces) <= max_faces:
    return mesh

  print(f'Simplifying mesh for inference: faces {len(mesh.faces)} -> {max_faces}')
  try:
    simplified = mesh.simplify_quadric_decimation(face_count=max_faces, aggression=aggression)
  except Exception as exc:
    print(f'trimesh simplify failed ({exc}); falling back to open3d')
    import open3d as o3d

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) == len(mesh.vertices):
      o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64) / 255.0)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=max_faces)
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_non_manifold_edges()
    o3d_mesh.compute_vertex_normals()

    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    simplified = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if o3d_mesh.has_vertex_colors():
      colors = (np.asarray(o3d_mesh.vertex_colors) * 255).clip(0, 255).astype(np.uint8)
      alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
      simplified.visual.vertex_colors = np.concatenate([colors, alpha], axis=1)

  simplified.remove_unreferenced_vertices()
  if len(simplified.faces) == 0:
    raise RuntimeError('Mesh simplification produced an empty mesh')

  print(f'Inference mesh: vertices={len(simplified.vertices)}, faces={len(simplified.faces)}')
  return simplified


def load_mesh_for_inference(mesh_file, mesh_scale, max_faces, aggression, output_dir):
  mesh = as_trimesh(trimesh.load(mesh_file, process=False))
  print(f'Loaded mesh: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}')
  if mesh_scale != 1.0:
    mesh.apply_scale(mesh_scale)
    print(f'Applied mesh scale: {mesh_scale}')
  mesh = simplify_mesh_for_inference(mesh, max_faces=max_faces, aggression=aggression)

  os.makedirs(output_dir, exist_ok=True)
  inference_mesh_file = os.path.join(output_dir, 'inference_mesh_simplified.ply')
  mesh.export(inference_mesh_file)
  print(f'Saved inference mesh: {inference_mesh_file}')
  return mesh


def labelme_json_to_mask(json_file, mask_file):
  with open(json_file, 'r') as f:
    data = json.load(f)

  h = int(data['imageHeight'])
  w = int(data['imageWidth'])
  mask = np.zeros((h, w), dtype=np.uint8)

  for shape in data.get('shapes', []):
    points = shape.get('points', [])
    if len(points) < 2:
      continue

    shape_type = shape.get('shape_type', 'polygon')
    pts = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)

    if shape_type == 'rectangle' and len(pts) >= 2:
      x0, y0 = pts[0]
      x1, y1 = pts[1]
      cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
    elif len(pts) >= 3:
      cv2.fillPoly(mask, [pts], 255)

  os.makedirs(os.path.dirname(mask_file), exist_ok=True)
  if not cv2.imwrite(mask_file, mask):
    raise RuntimeError(f'Failed to write mask: {mask_file}')

  return mask_file


def ensure_masks_from_labelme(scene_dir):
  rgb_dir = os.path.join(scene_dir, 'rgb')
  mask_dir = os.path.join(scene_dir, 'masks')
  generated = []

  for json_file in sorted(glob_png_stem_files(rgb_dir, '.json')):
    stem = os.path.splitext(os.path.basename(json_file))[0]
    mask_file = os.path.join(mask_dir, f'{stem}.png')
    if not os.path.exists(mask_file):
      labelme_json_to_mask(json_file, mask_file)
      generated.append(mask_file)

  return generated


def glob_png_stem_files(folder, suffix):
  return [
    os.path.join(folder, name)
    for name in os.listdir(folder)
    if name.endswith(suffix)
  ]


def prepare_output_dir(output_dir, overwrite):
  if overwrite and os.path.exists(output_dir):
    shutil.rmtree(output_dir)

  os.makedirs(os.path.join(output_dir, 'ob_in_cam'), exist_ok=True)
  os.makedirs(os.path.join(output_dir, 'track_vis'), exist_ok=True)
  os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)


def save_input_snapshot(output_dir, frame_id, color, depth, mask):
  imageio.imwrite(os.path.join(output_dir, 'masks', f'{frame_id}.png'), (mask > 0).astype(np.uint8) * 255)
  cv2.imwrite(os.path.join(output_dir, 'depth_mm_' + frame_id + '.png'), (depth * 1000).astype(np.uint16))
  imageio.imwrite(os.path.join(output_dir, 'rgb_' + frame_id + '.png'), color)


def save_render_batch(render_dir, call_name, rgb_tensor, depth_tensor):
  batch_dir = os.path.join(render_dir, call_name)
  rgb_dir = os.path.join(batch_dir, 'rgb')
  depth_dir = os.path.join(batch_dir, 'depth_mm')
  depth_npy_dir = os.path.join(batch_dir, 'depth_npy')
  os.makedirs(rgb_dir, exist_ok=True)
  os.makedirs(depth_dir, exist_ok=True)
  os.makedirs(depth_npy_dir, exist_ok=True)

  rgb = rgb_tensor.detach().float().cpu().numpy()
  depth = depth_tensor.detach().float().cpu().numpy()

  if rgb.max() <= 1.5:
    rgb = rgb * 255.0
  rgb = np.clip(rgb, 0, 255).astype(np.uint8)

  for i in range(rgb.shape[0]):
    imageio.imwrite(os.path.join(rgb_dir, f'{i:06d}.png'), rgb[i])
    depth_i = depth[i]
    np.save(os.path.join(depth_npy_dir, f'{i:06d}.npy'), depth_i)
    cv2.imwrite(os.path.join(depth_dir, f'{i:06d}.png'), np.clip(depth_i * 1000.0, 0, 65535).astype(np.uint16))


def install_render_saver(render_dir):
  import learning.training.predict_pose_refine as predict_pose_refine
  import learning.training.predict_score as predict_score

  os.makedirs(render_dir, exist_ok=True)
  counters = {'refine': 0, 'score': 0}

  def wrap(module, name):
    original = module.nvdiffrast_render

    def wrapped(*args, **kwargs):
      color, depth, normal = original(*args, **kwargs)
      call_name = f'{name}_{counters[name]:04d}'
      counters[name] += 1
      save_render_batch(render_dir, call_name, color, depth)
      return color, depth, normal

    module.nvdiffrast_render = wrapped

  wrap(predict_pose_refine, 'refine')
  wrap(predict_score, 'score')
  print(f'Saving rendered RGB/depth batches to: {render_dir}')


def parse_args():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  scene_dir = os.path.join(code_dir, 'demo_data/demo_box_yjj_long')

  parser = argparse.ArgumentParser(
    description='Run FoundationPose on demo_data/demo_box_yjj over SSH without realtime visualization.'
  )
  parser.add_argument('--mesh_file', type=str, default=os.path.join(scene_dir, 'mesh/fuse_mesh_rgb_cut_align.ply'))
  parser.add_argument('--test_scene_dir', type=str, default=scene_dir)
  parser.add_argument('--output_dir', type=str, default=os.path.join(code_dir, 'output/demo_box_yjj_long_ori'))
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1, help='Use 2 or higher to save FoundationPose internal debug files.')
  parser.add_argument('--mesh_scale', type=float, default=0.001, help='Scale mesh vertices on load. demo_box_yjj PLY is in millimeters, while depth is in meters.')
  parser.add_argument('--mesh_max_faces', type=int, default=50000, help='Downsample mesh to this many faces when loading. Set <=0 to disable.')
  parser.add_argument('--mesh_simplify_aggression', type=int, default=7, help='Quadric simplification aggression, 0 is slower/better and 10 is faster/coarser.')
  parser.add_argument('--save_rendered', action='store_true', help='Save every rendered candidate RGB and depth image under output_dir/rendered.')
  parser.add_argument('--zfar', type=float, default=np.inf)
  parser.add_argument('--overwrite', action='store_true', help='Clear output_dir before running.')
  parser.add_argument('--no_auto_mask', action='store_true', help='Do not create masks from LabelMe json files.')
  return parser.parse_args()


def main():
  args = parse_args()

  from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
  from estimater import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed
  from datareader import YcbineoatReader
  import nvdiffrast.torch as dr

  set_logging_format()
  set_seed(0)

  if args.save_rendered:
    install_render_saver(os.path.join(args.output_dir, 'rendered'))

  if not args.no_auto_mask:
    generated = ensure_masks_from_labelme(args.test_scene_dir)
    if generated:
      print(f'Generated {len(generated)} mask(s):')
      for mask_file in generated:
        print(f'  {mask_file}')

  prepare_output_dir(args.output_dir, args.overwrite)

  mesh = load_mesh_for_inference(
    mesh_file=args.mesh_file,
    mesh_scale=args.mesh_scale,
    max_faces=args.mesh_max_faces,
    aggression=args.mesh_simplify_aggression,
    output_dir=args.output_dir,
  )
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=args.output_dir,
    debug=args.debug,
    glctx=glctx,
  )

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=args.zfar)
  if len(reader.color_files) == 0:
    raise RuntimeError(f'No RGB png files found under {args.test_scene_dir}/rgb')

  for i in range(len(reader.color_files)):
    frame_id = reader.id_strs[i]
    color = reader.get_color(i)
    depth = reader.get_depth(i)

    if i == 0:
      mask = reader.get_mask(i).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    np.savetxt(os.path.join(args.output_dir, 'ob_in_cam', f'{frame_id}.txt'), pose.reshape(4, 4))
    np.save(os.path.join(args.output_dir, 'ob_in_cam', f'{frame_id}.npy'), pose.reshape(4, 4))

    center_pose = pose @ np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
    imageio.imwrite(os.path.join(args.output_dir, 'track_vis', f'{frame_id}.png'), vis)

    if i == 0:
      save_input_snapshot(args.output_dir, frame_id, color, depth, mask)

    print(f'[{i + 1}/{len(reader.color_files)}] saved pose and visualization for frame {frame_id}')

  print(f'Done. Results saved to: {args.output_dir}')


if __name__ == '__main__':
  main()
