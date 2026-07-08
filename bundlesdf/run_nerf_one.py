#!/usr/bin/env python

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent
sys.path.append(str(REPO_DIR))


def parse_ob_id(value):
  value = str(value)
  if value.startswith("ob_"):
    return int(value.split("_", 1)[1])
  return int(value)


def ob_dir_name(ob_id):
  return f"ob_{ob_id:07d}"


def resolve_mask_mode(base_dir, mask_mode):
  if mask_mode == "refined":
    return True
  if mask_mode == "raw":
    return False

  refined_dir = base_dir / "mask_refined"
  raw_dir = base_dir / "mask"
  if refined_dir.is_dir() and list(refined_dir.glob("*.png")):
    return True
  if raw_dir.is_dir() and list(raw_dir.glob("*.png")):
    return False
  raise FileNotFoundError(f"No mask_refined/*.png or mask/*.png found under {base_dir}")


def validate_ref_dir(base_dir, use_refined_mask):
  required_files = [
    base_dir / "K.txt",
    base_dir / "select_frames.yml",
  ]
  required_dirs = [
    base_dir / "rgb",
    base_dir / "depth_enhanced",
    base_dir / ("mask_refined" if use_refined_mask else "mask"),
    base_dir / "cam_in_ob",
  ]

  missing = [str(p) for p in required_files if not p.is_file()]
  missing += [str(p) for p in required_dirs if not p.is_dir()]
  if missing:
    raise FileNotFoundError("Missing required input paths:\n  " + "\n  ".join(missing))

  rgb_files = sorted((base_dir / "rgb").glob("*.png"))
  if not rgb_files:
    raise FileNotFoundError(f"No rgb/*.png files found under {base_dir}")

  missing_pairs = []
  mask_dir_name = "mask_refined" if use_refined_mask else "mask"
  for rgb_file in rgb_files:
    rel = rgb_file.relative_to(base_dir / "rgb")
    for subdir, suffix in [
      ("depth_enhanced", ".png"),
      (mask_dir_name, ".png"),
      ("cam_in_ob", ".txt"),
    ]:
      pair = base_dir / subdir / rel.with_suffix(suffix)
      if not pair.is_file():
        missing_pairs.append(str(pair))

  if missing_pairs:
    preview = "\n  ".join(missing_pairs[:20])
    extra = "" if len(missing_pairs) <= 20 else f"\n  ... and {len(missing_pairs) - 20} more"
    raise FileNotFoundError(f"Missing paired files for rgb frames:\n  {preview}{extra}")

  return len(rgb_files)


def load_config(dataset, config):
  if config is None:
    config = CODE_DIR / f"config_{dataset}.yml"
  else:
    config = Path(config)
  with open(config, "r") as f:
    return yaml.safe_load(f), config


def main():
  parser = argparse.ArgumentParser(
    description="Train BundleSDF Neural Object Field and export one reconstructed object mesh."
  )
  parser.add_argument("--ref_view_dir", required=True, help="Parent folder containing ob_XXXXXXX reference folders.")
  parser.add_argument("--ob_id", default="1", help="Object id, e.g. 1 or ob_0000001.")
  parser.add_argument("--dataset", choices=["linemod", "ycbv"], default="linemod")
  parser.add_argument("--config", default=None, help="Optional BundleSDF config YAML. Defaults to config_<dataset>.yml.")
  parser.add_argument("--mask_mode", choices=["auto", "refined", "raw"], default="auto")
  parser.add_argument("--skip_texture", action="store_true", help="Export geometry-only mesh and skip pyrender offscreen texture baking.")
  parser.add_argument("--no_hard_exit", action="store_true", help="Use normal Python teardown after saving instead of os._exit(0).")
  parser.add_argument("--dry_run", action="store_true", help="Only validate inputs and print planned output.")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  ob_id = parse_ob_id(args.ob_id)
  ref_view_dir = Path(args.ref_view_dir).expanduser().resolve()
  base_dir = ref_view_dir / ob_dir_name(ob_id)
  if not base_dir.is_dir():
    raise FileNotFoundError(f"Object reference directory not found: {base_dir}")

  cfg, config_path = load_config(args.dataset, args.config)
  use_refined_mask = resolve_mask_mode(base_dir, args.mask_mode)
  n_frames = validate_ref_dir(base_dir, use_refined_mask)

  out_file = base_dir / "model" / "model.obj"
  logging.info("ref_view_dir: %s", ref_view_dir)
  logging.info("object dir: %s", base_dir)
  logging.info("dataset/config: %s / %s", args.dataset, config_path)
  logging.info("frames: %d", n_frames)
  logging.info("mask folder: %s", "mask_refined" if use_refined_mask else "mask")
  logging.info("texture baking: %s", "disabled" if args.skip_texture else "enabled")
  logging.info("output mesh: %s", out_file)
  logging.info("BundleSDF scratch folder will be refreshed: %s", base_dir / "nerf")

  if args.dry_run:
    logging.info("dry run finished")
    return

  from run_nerf import run_one_ob

  mesh = run_one_ob(base_dir=str(base_dir), cfg=cfg, use_refined_mask=use_refined_mask, texture_mesh=not args.skip_texture)
  out_file.parent.mkdir(parents=True, exist_ok=True)
  mesh.export(out_file)
  logging.info("saved to %s", out_file)
  if not args.no_hard_exit:
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
  main()
