"""Script to crop and prepare real lunar image pairs for the registration demo."""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None

def parse_crop_arg(crop_str):
    """Parse 'x,y,w,h' string into a tuple of integers."""
    if not crop_str:
        return None
    try:
        parts = [int(p.strip()) for p in crop_str.split(",")]
        if len(parts) != 4:
            raise ValueError("Crop string must contain exactly 4 comma-separated integers.")
        return tuple(parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid crop argument '{crop_str}': {e}")

def load_image(filepath):
    """Load image safely, preferring tifffile for TIFFs, with fallback to Pillow."""
    filepath_str = str(filepath)
    img_array = None
    
    # Try tifffile first if it's a TIF and the library is available
    if filepath_str.lower().endswith(('.tif', '.tiff')) and tifffile is not None:
        try:
            img_array = tifffile.imread(filepath_str)
        except Exception as e:
            print(f"Warning: tifffile failed to load {filepath_str} ({e}), falling back to Pillow.")
    
    # Fallback to Pillow
    if img_array is None:
        try:
            img = Image.open(filepath_str).convert('L')
            img_array = np.array(img)
        except Exception as e:
            raise ValueError(f"Failed to load {filepath_str} with Pillow: {e}")
            
    # Ensure 2D (if it loaded as 3D for some reason, e.g. an RGB TIFF)
    if img_array.ndim > 2:
        # Simple luminance conversion if 3+ channels
        if img_array.shape[2] >= 3:
            img_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            img_array = img_array[:, :, 0]
            
    return img_array

def process_image(img_array, crop):
    """Apply cropping and convert to an 8-bit grayscale array."""
    # 1. Apply Crop
    if crop:
        x, y, w, h = crop
        
        # Ensure crop is within image bounds to prevent slice errors
        height, width = img_array.shape
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(width, x + w)
        y_end = min(height, y + h)
        
        if x_start >= x_end or y_start >= y_end:
             raise ValueError(f"Crop region {crop} is completely outside image bounds {width}x{height}")
             
        img_array = img_array[y_start:y_end, x_start:x_end]
        
    # 2. Convert to 8-bit grayscale
    if img_array.dtype != np.uint8:
        if np.issubdtype(img_array.dtype, np.floating):
            img_min, img_max = np.nanmin(img_array), np.nanmax(img_array)
            if img_max > img_min:
                img_array = ((img_array - img_min) / (img_max - img_min) * 255.0)
            img_array = np.clip(np.nan_to_num(img_array), 0, 255).astype(np.uint8)
        else:
            # For integer types like uint16
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                 img_array = ((img_array - img_min) / (img_max - img_min) * 255.0)
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
    return img_array

def main():
    parser = argparse.ArgumentParser(
        description="Crop and convert large source/reference images to 8-bit grayscale PNGs for the demo.",
        epilog="""
Examples:
  # Prepare a real pair with cropping
  python scripts/crop_and_prepare_real_pair.py \\
      --source data/demo_pairs/ohrc_nac_crater_x/source_ohrc.tif \\
      --reference data/demo_pairs/ohrc_nac_crater_x/reference_nac.tif \\
      --crop-source 1000,2000,500,500 \\
      --crop-reference 2500,3000,800,800 \\
      --output-name my_demo_pair
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--source", required=True, type=str,
                        help="Path to the original source image (e.g., OHRC TIFF file).")
    parser.add_argument("--reference", required=True, type=str,
                        help="Path to the original reference image (e.g., NAC TIFF file).")
    parser.add_argument("--crop-source", type=parse_crop_arg, default=None,
                        help="Crop region for the source image as 'x,y,width,height'.")
    parser.add_argument("--crop-reference", type=parse_crop_arg, default=None,
                        help="Crop region for the reference image as 'x,y,width,height'.")
    parser.add_argument("--output-name", required=True, type=str,
                        help="Name of the new subfolder to create in data/demo_pairs/.")
                        
    args = parser.parse_args()
    
    source_path = Path(args.source)
    reference_path = Path(args.reference)
    
    if not source_path.exists():
        parser.error(f"Source file not found: {source_path}")
    if not reference_path.exists():
        parser.error(f"Reference file not found: {reference_path}")
        
    # Setup output paths
    project_root = Path(__file__).resolve().parent.parent
    demo_pairs_dir = project_root / "data" / "demo_pairs"
    output_dir = demo_pairs_dir / args.output_name
    
    if not demo_pairs_dir.exists():
        demo_pairs_dir.mkdir(parents=True, exist_ok=True)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_source_path = output_dir / "source.png"
    out_reference_path = output_dir / "reference.png"
    
    # 1. Load and process SOURCE
    print(f"Loading source image: {source_path}...")
    source_img = load_image(source_path)
    print(f"Source image loaded. Shape: {source_img.shape}, Type: {source_img.dtype}")
    source_processed = process_image(source_img, args.crop_source)
    
    # 2. Load and process REFERENCE
    print(f"\nLoading reference image: {reference_path}...")
    reference_img = load_image(reference_path)
    print(f"Reference image loaded. Shape: {reference_img.shape}, Type: {reference_img.dtype}")
    reference_processed = process_image(reference_img, args.crop_reference)
    
    # 3. Save as PNG
    print(f"\nSaving processed source to: {out_source_path}")
    Image.fromarray(source_processed).save(out_source_path, format="PNG")
    
    print(f"Saving processed reference to: {out_reference_path}")
    Image.fromarray(reference_processed).save(out_reference_path, format="PNG")
    
    # 4. Print Summary
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Source image dimensions: {source_processed.shape[1]}x{source_processed.shape[0]}")
    print(f"Source file size:      {out_source_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Reference dimensions:    {reference_processed.shape[1]}x{reference_processed.shape[0]}")
    print(f"Reference file size:   {out_reference_path.stat().st_size / (1024*1024):.2f} MB")
    print("="*50)
    print(f"Done! Your new demo pair is ready in: {output_dir}")

if __name__ == "__main__":
    main()
