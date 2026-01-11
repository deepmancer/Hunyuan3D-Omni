# -*- coding: utf-8 -*-
"""
Script for 3D generation with point cloud control using Hunyuan3D-Omni.

This script loads an image and a mesh, samples points from the mesh surface,
and uses them as conditioning input for Hunyuan3D-Omni.
"""

# Load libgcc_s library to prevent runtime issues
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings('ignore')

import argparse
import random
import time
import torch
import trimesh
import shutil
import numpy as np
import pandas as pd
from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer


def save_ply_points(filename: str, points: np.ndarray) -> None:
    """Save 3D points to a PLY format file."""
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % len(points))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for point in points:
            f.write('%f %f %f\n' % (point[0], point[1], point[2]))


def normalize_mesh(mesh: trimesh.Trimesh, scale: float = 0.9999) -> trimesh.Trimesh:
    """
    Normalize a 3D mesh to fit within a centered cube [-scale, scale].
    """
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale_ = (bbox[1] - bbox[0]).max()
    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale_ * 2 * scale)
    return mesh


def postprocess_and_save(mesh, save_dir, max_facenum=200000):
    """Post-process the generated mesh and save outputs."""
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=max_facenum)
    mesh.export(os.path.join(save_dir, 'shape_mesh.glb'))


def compute_normalized_bbox(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute the normalized bounding box aspect ratios from a mesh.
    
    The bounding box is represented as 3 values (width, height, depth) normalized
    so that the maximum dimension equals 1.0. This matches the format expected
    by Hunyuan3D-Omni's bbox conditioning.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh
        
    Returns:
        np.ndarray: Normalized bbox with shape [3], values in range (0, 1]
    """
    # Get bounding box: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    bounds = mesh.bounds
    
    # Compute extent (size) in each dimension
    extent = bounds[1] - bounds[0]  # [width, height, depth]
    
    # Normalize so the maximum dimension is 1.0
    max_extent = extent.max()
    if max_extent > 0:
        normalized_bbox = extent / max_extent
    else:
        normalized_bbox = np.array([1.0, 1.0, 1.0])
    
    return normalized_bbox


def infer_bbox_from_mesh(
    pipeline,
    data_dir: str,
    sample_id: str,
    seed: int = 1234,
    max_facenum: int = 200000,
) -> None:
    """
    Perform 3D generation with bounding box control derived from input mesh.
    
    This function computes the bounding box aspect ratios from a reference mesh
    and uses them as conditioning input for Hunyuan3D-Omni.
    
    Args:
        pipeline: Hunyuan3D-Omni pipeline instance
        data_dir (str): Base directory containing the data
        sample_id (str): Sample identifier
        seed (int): Random seed for reproducibility (default: 1234)
        max_facenum (int): Maximum number of faces for mesh reduction (default: 200000)
    
    Input paths:
        - Image: {data_dir}/matted_image_centered/{sample_id}.png
        - Mesh: {data_dir}/hi3dgen/{sample_id}/shape_mesh.glb
    
    Outputs:
        - {data_dir}/hunyuan_omni_bbox/{sample_id}/shape_mesh.glb: Generated 3D mesh
    """
    # Construct input paths
    image_file = os.path.join(data_dir, "matted_image_centered", f"{sample_id}.png")
    mesh_file = os.path.join(data_dir, "hi3dgen", sample_id, "shape_mesh.glb")
    
    # Construct output directory (different from point conditioning)
    save_dir = os.path.join(data_dir, "hunyuan_omni_bbox", sample_id)
    
    print(f"Image path: {image_file}")
    print(f"Mesh path: {mesh_file}")
    print(f"Output dir: {save_dir}")
    
    # Skip if output already exists
    output_mesh_path = os.path.join(save_dir, "shape_mesh.glb")
    if os.path.exists(output_mesh_path):
        print(f"Output already exists, skipping: {output_mesh_path}")
        return
    
    # Validate input files exist
    if not os.path.exists(image_file):
        print(f"Warning: Image file not found, skipping: {image_file}")
        return
    if not os.path.exists(mesh_file):
        print(f"Warning: Mesh file not found, skipping: {mesh_file}")
        return
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load mesh to compute bounding box
    print(f"Loading mesh from: {mesh_file}")
    mesh = trimesh.load(mesh_file, force='mesh')
    
    # Compute normalized bounding box aspect ratios
    normalized_bbox = compute_normalized_bbox(mesh)
    print(f"Mesh bounds: {mesh.bounds}")
    print(f"Normalized bbox (aspect ratios): {normalized_bbox}")
    
    # Convert to tensor with shape [1, 1, 3] as expected by the pipeline
    bbox_tensor = torch.FloatTensor(normalized_bbox).unsqueeze(0).unsqueeze(0)
    bbox_tensor = bbox_tensor.to(pipeline.device).to(pipeline.dtype)
    print(f"Bbox tensor shape: {bbox_tensor.shape}")
    
    # Run inference with bounding box conditioning
    print("Running Hunyuan3D-Omni inference with bbox conditioning...")
    result = pipeline(
        image=image_file,
        bbox=bbox_tensor,
        num_inference_steps=50,
        octree_resolution=512,
        mc_level=0,
        guidance_scale=4.5,
        generator=torch.Generator('cuda').manual_seed(seed),
    )
    
    # Extract results
    generated_mesh = result['shapes'][0][0]
    
    # Save outputs
    print(f"Saving outputs to: {save_dir}")
    postprocess_and_save(generated_mesh, save_dir, max_facenum=max_facenum)
    print(f"Saved: shape_mesh.glb")


def infer_point_from_mesh(
    pipeline,
    data_dir: str,
    sample_id: str,
    num_points: int = 1024,
    seed: int = 1234,
    max_facenum: int = 200000,
) -> None:
    """
    Perform 3D generation with point cloud control.
    
    Args:
        pipeline: Hunyuan3D-Omni pipeline instance
        data_dir (str): Base directory containing the data
        sample_id (str): Sample identifier
        num_points (int): Number of points to sample from mesh surface (default: 1024)
        seed (int): Random seed for reproducibility (default: 1234)
        max_facenum (int): Maximum number of faces for mesh reduction (default: 200000)
    
    Input paths:
        - Image: {data_dir}/matted_image_outpainted/{sample_id}.png
        - Mesh: {data_dir}/hi3dgen/{sample_id}/shape_mesh.glb
    
    Outputs:
        - {data_dir}/hunyuan_omni/{sample_id}/shape_mesh.glb: Generated 3D mesh
        - {data_dir}/hunyuan_omni/{sample_id}/shape_mesh.ply: Point cloud representation
        - {data_dir}/hunyuan_omni/{sample_id}/input.png: Copy of input image
    """
    # Construct input paths
    image_file = os.path.join(data_dir, "matted_image_centered", f"{sample_id}.png")
    mesh_file = os.path.join(data_dir, "hi3dgen", sample_id, "shape_mesh.glb")
    
    # Construct output directory
    save_dir = os.path.join(data_dir, "hunyuan_omni", sample_id)
    
    print(f"Image path: {image_file}")
    print(f"Mesh path: {mesh_file}")
    print(f"Output dir: {save_dir}")
    
    # Skip if output already exists
    output_mesh_path = os.path.join(save_dir, "shape_mesh.glb")
    if os.path.exists(output_mesh_path):
        print(f"Output already exists, skipping: {output_mesh_path}")
        return
    
    # Validate input files exist
    if not os.path.exists(image_file):
        print(f"Warning: Image file not found, skipping: {image_file}")
        return
    if not os.path.exists(mesh_file):
        print(f"Warning: Mesh file not found, skipping: {mesh_file}")
        return
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and process mesh
    print(f"Loading mesh from: {mesh_file}")
    mesh = trimesh.load(mesh_file, force='mesh')
    
    # Normalize mesh to [-0.98, 0.98] range
    mesh = normalize_mesh(mesh, scale=0.98)
    
    # Sample points uniformly from the mesh surface
    print(f"Sampling {num_points} points from mesh surface...")
    sampled_points, _ = trimesh.sample.sample_surface(mesh, num_points)
    print(f"Sampled point cloud shape: {sampled_points.shape}")
    
    # Convert to tensor with shape [1, num_points, 3]
    surface = torch.FloatTensor(sampled_points).unsqueeze(0)
    surface = surface.to(pipeline.device).to(pipeline.dtype)
    print(f"Point cloud tensor shape: {surface.shape}")
    
    # Run inference with point cloud conditioning
    print("Running Hunyuan3D-Omni inference...")
    result = pipeline(
        image=image_file,
        point=surface,
        num_inference_steps=50,
        octree_resolution=512,
        mc_level=0,
        guidance_scale=4.5,
        generator=torch.Generator('cuda').manual_seed(seed),
    )
    
    # Extract results
    generated_mesh = result['shapes'][0][0]
    
    # Save outputs
    print(f"Saving outputs to: {save_dir}")
    postprocess_and_save(generated_mesh, save_dir, max_facenum=max_facenum)
    print(f"Saved: shape_mesh.glb")


def get_all_sample_ids(data_dir: str) -> list:
    """
    Get all sample IDs by listing directories in hi3dgen folder.
    Optionally filters by target_id column in pairs.csv if it exists.
    
    Args:
        data_dir (str): Base directory containing hi3dgen/ subdirectory
        
    Returns:
        list: Sorted list of sample IDs
    """
    hi3dgen_dir = os.path.join(data_dir, "hi3dgen")
    if not os.path.exists(hi3dgen_dir):
        raise FileNotFoundError(f"hi3dgen directory not found: {hi3dgen_dir}")
    
    # Check for pairs.csv to filter by target_id
    pairs_csv_path = os.path.join(data_dir, 'pairs.csv')
    target_ids = None
    if os.path.exists(pairs_csv_path):
        print(f"Found pairs.csv at {pairs_csv_path}")
        try:
            pairs_df = pd.read_csv(pairs_csv_path)
            if 'target_id' in pairs_df.columns:
                target_ids = set(pairs_df['target_id'].astype(str).unique())
                print(f"Filtering by {len(target_ids)} unique target_ids from pairs.csv")
            else:
                print("Warning: pairs.csv found but 'target_id' column not present")
        except Exception as e:
            print(f"Warning: Failed to read pairs.csv: {e}")
    
    sample_ids = []
    for name in os.listdir(hi3dgen_dir):
        sample_path = os.path.join(hi3dgen_dir, name)
        if os.path.isdir(sample_path):
            # Check if shape_mesh.glb exists
            mesh_file = os.path.join(sample_path, "shape_mesh.glb")
            if os.path.exists(mesh_file):
                # Filter by target_ids if pairs.csv was found
                if target_ids is not None:
                    if name in target_ids:
                        sample_ids.append(name)
                else:
                    sample_ids.append(name)
    
    return sorted(sample_ids)


def get_args():
    parser = argparse.ArgumentParser(
        description='Hunyuan3D-Omni Point Cloud Conditioning from Mesh',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, default="/workspace/celeba_reduced",
                        help='Base directory containing matted_image_outpainted/ and hi3dgen/ subdirectories')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Number of points to sample from mesh surface')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for reproducibility')
    parser.add_argument('--max_facenum', type=int, default=100_000,
                        help='Maximum number of faces for mesh reduction')
    parser.add_argument('--repo_id', type=str, default="tencent/Hunyuan3D-Omni",
                        help='Model ID on HuggingFace')
    parser.add_argument('--flashvdm', action='store_true', default=True,
                        help='Use FlashVDM for faster decoding')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    print("=" * 80)
    print("HUNYUAN3D-OMNI POINT CLOUD INFERENCE FROM MESH")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Number of points: {args.num_points}")
    print(f"Max face count: {args.max_facenum}")
    print("=" * 80)
    
    # Get all sample IDs (filtered by pairs.csv if exists)
    sample_ids = get_all_sample_ids(args.data_dir)
    print(f"Found {len(sample_ids)} samples matching criteria")
    
    # Filter out already-processed samples
    samples_to_process = []
    for sample_id in sample_ids:
        output_mesh_path = os.path.join(args.data_dir, "hunyuan_omni", sample_id, "shape_mesh.glb")
        if not os.path.exists(output_mesh_path):
            samples_to_process.append(sample_id)
    
    already_processed = len(sample_ids) - len(samples_to_process)
    print(f"Already processed: {already_processed}")
    print(f"Remaining to process: {len(samples_to_process)}")
    
    if len(samples_to_process) == 0:
        print("All samples have already been processed!")
        exit(0)
    
    # Randomize processing order using current timestamp as seed
    random.seed(int(time.time()))
    random.shuffle(samples_to_process)
    print(f"Randomized processing order (seed: {int(time.time())})")
    
    # Initialize pipeline
    print(f"Loading model from: {args.repo_id}")
    pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
        args.repo_id,
        fast_decode=args.flashvdm
    )
    
    # Run inference on all samples
    success_count = 0
    for idx, sample_id in enumerate(samples_to_process):
        print("\n" + "=" * 80)
        print(f"Processing sample {idx + 1}/{len(samples_to_process)}: {sample_id}")
        print("=" * 80)
        
        try:
            infer_bbox_from_mesh(
                pipeline=pipeline,
                data_dir=args.data_dir,
                sample_id=sample_id,
                seed=args.seed,
                max_facenum=args.max_facenum,
            )
            success_count += 1
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"INFERENCE COMPLETED: {success_count}/{len(samples_to_process)} successful")
    print("=" * 80)
