#!/usr/bin/env python3
import argparse
import struct
import h5py
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

# Constants 
DESCRIPTOR_DIM = 256  # (256 - superpoint, 128 - others)
LINE_DESCRIPTOR_DIM = 0 

# Binary writing helpers
def u32(x: int) -> bytes: 
    return struct.pack("<I", int(x))

def f32(x: float) -> bytes: 
    return struct.pack("<f", float(x))

def write_optional_u32(f, val: Optional[int]) -> None:
    """Write an optional uint32 in Polycam format"""
    has_value = 1 if val is not None else 0
    f.write(struct.pack("<B", has_value))
    if has_value:
        f.write(u32(val))

def write_optional_f32(f, val: Optional[float]) -> None:
    """Write an optional float32 in Polycam format"""
    has_value = 1 if val is not None else 0
    f.write(struct.pack("<B", has_value))
    if has_value:
        f.write(f32(val))

def write_optional_matrix3(f, val: Optional[np.ndarray]) -> None:
    """Write an optional 3×3 matrix"""
    has_value = 1 if val is not None else 0
    f.write(struct.pack("<B", has_value))
    if has_value:
        if val.shape != (3, 3):
            raise ValueError(f"Matrix3 must be 3x3, got {val.shape}")
        f.write(val.astype("float32").tobytes())

def write_optional_pose3(f, val: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Write an optional Pose3 (R: 3×3, C: 3)"""
    has_value = 1 if val is not None else 0
    f.write(struct.pack("<B", has_value))
    if has_value:
        R, C = val
        if R.shape != (3, 3) or C.shape != (3,):
            raise ValueError(f"Pose3 must have R(3x3) and C(3), got R{R.shape} C{C.shape}")
        f.write(R.astype("float32").tobytes())
        f.write(C.astype("float32").tobytes())

def write_features_bin(dst: Path, wh: Tuple[int, int], kpts: np.ndarray, desc: np.ndarray) -> None:
    """
    Writes the features file in the exact Polycam format.

    Args:
        dst: destination path
        wh: image (width, height)
        kpts: array (N, 2) float32 keypoints
        desc: array (N, DESCRIPTOR_DIM) float32 descriptors
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    w, h = wh
    N = kpts.shape[0]
    
    # Validations
    if desc.shape[1] != DESCRIPTOR_DIM:
        raise ValueError(f"Descriptor dimension must be {DESCRIPTOR_DIM}, got {desc.shape[1]}")
    
    if kpts.shape[1] != 2:
        raise ValueError(f"Keypoints must have shape (N,2), got {kpts.shape}")
    
    with open(dst, "wb") as f:
        # Header
        f.write(u32(N)) # num_keypoints
        f.write(u32(w)) # width
        f.write(u32(h)) # height
        
        # Keypoints - cv::Point2f (x, y)
        kpts_float32 = kpts.astype("float32", copy=False)
        f.write(kpts_float32.tobytes())
        
        # Descriptors - N × DESCRIPTOR_DIM floats
        desc_float32 = desc.astype("float32", copy=False)
        f.write(desc_float32.tobytes())
        
        # Lines
        f.write(u32(0))  # num_lines = 0
        # Do not write any line data when num_lines == 0
        
        # Planes — when there are no planes, ExportPlanes writes only numPlanes
        f.write(u32(0))  # num_planes = 0

def write_matches_bin(dst: Path, matches: np.ndarray, num_inliers: Optional[int] = None) -> None:
    """
    Writes the matches file in the exact Polycam format.

    Args:
        dst: destination path
        matches: array (M, 2) uint32 — indices of the matched features
        num_inliers: optional number of inliers
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    M = matches.shape[0] if matches is not None else 0
    
    with open(dst, "wb") as f:
        # Header
        f.write(u32(M))  # num_matches
        
        # Optional numInliers
        write_optional_u32(f, num_inliers)
        
        # If num_inliers is present and equals 0, STOP
        if num_inliers == 0:
            return
        # Write matches only if we actually have matches
        if M > 0:
            matches_uint32 = matches.astype("<u4", copy=False)
            f.write(matches_uint32.tobytes()) 
        # 10 optional fields — all absent
        # H, F, E, pose, focalLength, sampsonMsacScore, inlierRatio, homographyScore, angle, poseScore
        for _ in range(10):
            f.write(struct.pack("<B", 0))  # hasValue = 0
        
        # Line matches — none
        f.write(u32(0))  # num_line_matches = 0
        write_optional_u32(f, None)  # numLineInliers absent

# HLOC readers 
def load_hloc_features(h5_path: Path, image_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features from the HLOC H5 file.
    
    Returns:
        kpts: (N, 2) float32 keypoints
        desc: (N, DESCRIPTOR_DIM) float32 descriptors
    """
    with h5py.File(h5_path, "r") as h5:
        # Try the exact name first, then the basename
        if image_name in h5:
            grp = h5[image_name]
        elif Path(image_name).name in h5:
            grp = h5[Path(image_name).name]
        else:
            raise KeyError(f"Image {image_name} not found in {h5_path}")
        
        kpts = grp["keypoints"][()].astype("float32")  # (N, 2)
        desc = grp["descriptors"][()]  # can be (D, N) or (N, D)
        
        # Normalize to shape (N, D)
        if desc.ndim != 2:
            raise ValueError(f"Descriptors must be 2D, got {desc.shape}")
        
        if desc.shape[0] != kpts.shape[0] and desc.shape[1] == kpts.shape[0]:
            desc = desc.T
        
        if desc.shape[0] != kpts.shape[0]:
            raise ValueError(f"Descriptor count {desc.shape[0]} != keypoint count {kpts.shape[0]}")
        
        # Resize descriptors if necessary
        if desc.shape[1] != DESCRIPTOR_DIM:
            print(f"Warning: Resizing descriptors from {desc.shape[1]} to {DESCRIPTOR_DIM} for {image_name}")
            # Interpolation or padding — choose the strategy you want
            if desc.shape[1] < DESCRIPTOR_DIM:
                # Zero-padding
                desc_padded = np.zeros((desc.shape[0], DESCRIPTOR_DIM), dtype=desc.dtype)
                desc_padded[:, :desc.shape[1]] = desc
                desc = desc_padded
            else:
                # Truncate
                desc = desc[:, :DESCRIPTOR_DIM]
        
        return kpts, desc.astype("float32")

def load_image_wh(images_root: Path, image_name: str, h5_path: Optional[Path] = None) -> Tuple[int, int]:
    """
    Loads the image dimensions from the H5 file or from the image file.
    """
    # Try H5 first
    if h5_path is not None:
        try:
            with h5py.File(h5_path, "r") as h5:
                key = image_name if image_name in h5 else Path(image_name).name
                if key in h5 and "image_size" in h5[key]:
                    H, W = h5[key]["image_size"][()]  # HLOC stores (height, width)
                    return (int(W), int(H))
        except Exception as e:
            print(f"Warning: Could not read image size from H5 for {image_name}: {e}")
    
    # Fallback to PIL
    if Image is None:
        raise RuntimeError("Pillow not available and H5 has no image size")
    
    image_path = images_root / image_name
    if not image_path.exists():
        # Search by basename
        candidates = list(images_root.rglob(Path(image_name).name))
        if candidates:
            image_path = candidates[0]
        else:
            raise FileNotFoundError(f"Image {image_name} not found in {images_root}")
    
    with Image.open(image_path) as img:
        return (img.width, img.height)

def iter_hloc_pairs(h5_matches: Path):
    import numpy as np, h5py

    def to_pairs(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[1] == 2:
            return a.astype("<u4", copy=False)
        if a.ndim == 2 and a.shape[0] == 2:
            return a.T.astype("<u4", copy=False)
        if a.ndim == 1:
            idx = a.astype(np.int64, copy=False)
            valid = np.where(idx >= 0)[0]
            if valid.size == 0:
                return np.zeros((0, 2), dtype="<u4")
            return np.stack([valid.astype(np.uint32), idx[valid].astype(np.uint32)], axis=1)
        # unsupported
        return np.zeros((0, 2), dtype="<u4")

    with h5py.File(h5_matches, "r") as h5:
        root = h5["matches"] if "matches" in h5 else h5
        out = []

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            parts = name.strip("/").split("/")
            ds_name = parts[-1]

            # Case 1: /img1/img2/{matches, matches0, matches1}
            if ds_name in {"matches", "matches0", "matches1"} and len(parts) >= 3:
                img1, img2 = parts[-3], parts[-2]
                pairs = to_pairs(obj[()])
                out.append((img1, img2, pairs))
                return

            # Case 2: dataset is named "<img1, img2>" or "<img1|img2>" or "<img1 img2>"
            for sep in [",", "|", " "]:
                if sep in ds_name:
                    img1, img2 = ds_name.split(sep, 1)
                    pairs = to_pairs(obj[()])
                    out.append((img1.strip(), img2.strip(), pairs))
                    return

            # Case 3: parent group is "<img1, img2>" etc., dataset is "matches*"
            if len(parts) >= 2:
                parent = parts[-2]
                for sep in [",", "|", " "]:
                    if sep in parent:
                        img1, img2 = parent.split(sep, 1)
                        pairs = to_pairs(obj[()])
                        out.append((img1.strip(), img2.strip(), pairs))
                        return

        root.visititems(visit)
        for rec in out:
            yield rec


def sanitize_for_filename(name: str) -> str:
    """
    Sanitize the name for filesystem use.
    """
    import re
    # Replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unknown_image"
    return sanitized

def export_polycam_bins(
    sparse_dir: Path,
    images_dir: Path, 
    feats_h5: Path,
    matches_h5: Path,
    dst_dir: Path,
    num_inliers_strategy: str = "all"  # "all", "none", or "adaptive"
) -> None:
    """
    Exports features and matches in the Polycam format.

    Args:
        sparse_dir: COLMAP sparse directory (unused, reserved)
        images_dir: Directory with the original images
        feats_h5: HLOC features H5 file
        matches_h5: HLOC matches H5 file
        dst_dir: Output (destination) directory
        num_inliers_strategy: How to set numInliers in the matches:
            - "all": numInliers = num_matches for each pair
            - "none": numInliers is omitted for all pairs
            - "adaptive": numInliers = min(num_matches, threshold)
    """
    dst_features = dst_dir / "features"
    dst_matches = dst_dir / "matches"
    dst_features.mkdir(parents=True, exist_ok=True)
    dst_matches.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting features to {dst_features}")
    print(f"Exporting matches to {dst_matches}")
    print(f"Using descriptor dimension: {DESCRIPTOR_DIM}")
    
    # Export features
    with h5py.File(feats_h5, "r") as hf:
        image_keys = list(hf.keys())
    
    print(f"Processing {len(image_keys)} images...")
    
    for i, image_key in enumerate(image_keys):
        try:
            kpts, desc = load_hloc_features(feats_h5, image_key)
            w, h = load_image_wh(images_dir, image_key, feats_h5)
            out_name = f"{sanitize_for_filename(image_key)}.bin"
            out_path = dst_features / out_name
            
            write_features_bin(out_path, (w, h), kpts, desc)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_keys)} images")
                
        except Exception as e:
            print(f"Error processing image {image_key}: {e}")
            continue
    
    # Export matches
    print("Processing matches...")
    match_count = 0
    
    for img1, img2, matches in iter_hloc_pairs(matches_h5):
        try:
            # Determine num_inliers according to the chosen strategy
            num_inliers = None
            if num_inliers_strategy == "all":
                num_inliers = matches.shape[0]
            elif num_inliers_strategy == "adaptive":
                # You can adjust this logic
                num_inliers = min(matches.shape[0], matches.shape[0] // 2)
            # For "none", num_inliers stays None
            
            out_name = f"{sanitize_for_filename(img1)}_{sanitize_for_filename(img2)}.bin"
            out_path = dst_matches / out_name
            
            write_matches_bin(out_path, matches, num_inliers)
            match_count += 1
            
            if match_count % 100 == 0:
                print(f"Processed {match_count} match pairs")
                
        except Exception as e:
            print(f"Error processing match pair {img1}-{img2}: {e}")
            continue
    
    print(f"Export completed: {len(image_keys)} features, {match_count} match pairs")

def main():
    global DESCRIPTOR_DIM
    parser = argparse.ArgumentParser(
        description="Export HLOC/COLMAP outputs to Polycam binary format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--sparse", type=Path, required=True, 
                       help="COLMAP sparse directory (reserved for future use)")
    parser.add_argument("--features-h5", type=Path, required=True,
                       help="HLOC features H5 file")
    parser.add_argument("--matches-h5", type=Path, required=True,
                       help="HLOC matches H5 file") 
    parser.add_argument("--images", type=Path, required=True,
                       help="Images directory")
    parser.add_argument("--dst", type=Path, required=True,
                       help="Destination directory for Polycam binaries")
    parser.add_argument("--descriptor-dim", type=int, default=DESCRIPTOR_DIM,
                       help="Descriptor dimension (should match C++ sizeDescriptor)")
    parser.add_argument("--num-inliers", choices=["all", "none", "adaptive"], 
                       default="all", help="Strategy for numInliers in matches")
    
    args = parser.parse_args()
    
    # Set the global descriptor dimension
    DESCRIPTOR_DIM = args.descriptor_dim
    
    print(f"Using descriptor dimension: {DESCRIPTOR_DIM}")
    print(f"Using num_inliers strategy: {args.num_inliers}")
    
    export_polycam_bins(
        args.sparse,
        args.images, 
        args.features_h5,
        args.matches_h5,
        args.dst,
        args.num_inliers
    )

if __name__ == "__main__":
    main()