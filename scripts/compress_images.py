#!/usr/bin/env python3
"""
Image Compression Script

Compresses images by converting PNG to JPEG and reducing file sizes.
Handles transparency, resizes large images, and shows compression statistics.

Usage Examples:
    # Compress a single image
    uv run scripts/compress_images.py posts/static_blog_imgs/large_image.png
    
    # Compress multiple images
    uv run scripts/compress_images.py posts/static_blog_imgs/img1.png posts/static_blog_imgs/img2.png
    
    # Compress all PNG files in a directory
    uv run scripts/compress_images.py posts/static_blog_imgs/*.png
    
    # Custom quality (lower = smaller file, default: 85)
    uv run scripts/compress_images.py -q 70 posts/static_blog_imgs/image.png
    
    # Custom max width (default: 1200px)
    uv run scripts/compress_images.py -w 800 posts/static_blog_imgs/image.png
    
    # Combine options
    uv run scripts/compress_images.py -q 75 -w 1000 posts/static_blog_imgs/*.png

Features:
- Converts PNG to JPEG (handles transparency with white background)
- Resizes images larger than max_width while maintaining aspect ratio
- Shows original vs compressed file sizes and compression percentage
- Deletes original PNG files after successful compression
- Preserves filename but changes extension to .jpg
"""

import argparse
import os
import sys
from PIL import Image
from pathlib import Path

def compress_image(input_path, quality=85, max_width=1200):
    """
    Compress an image by converting to JPEG and reducing size/quality
    
    Args:
        input_path: Path to input image
        quality: JPEG quality (1-100, lower = smaller file)
        max_width: Maximum width to resize to (maintains aspect ratio)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        return False
    
    try:
        # Open and process image
        with Image.open(input_path) as img:
            # Convert RGBA to RGB if needed (for PNG transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            
            # Resize if too large
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                print(f"Resized from {input_path.name} to {max_width}x{new_height}")
            
            # Generate output filename (replace extension with .jpg)
            output_path = input_path.with_suffix('.jpg')
            
            # Get original file size
            original_size = input_path.stat().st_size
            
            # Save as JPEG
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            # Get new file size
            new_size = output_path.stat().st_size
            compression_ratio = (1 - new_size/original_size) * 100
            
            print(f"Compressed: {input_path.name}")
            print(f"  Original: {original_size/1024/1024:.1f}MB")
            print(f"  New: {new_size/1024/1024:.1f}MB")
            print(f"  Savings: {compression_ratio:.1f}%")
            
            # Delete original file if it's different from output
            if input_path != output_path:
                input_path.unlink()
                print(f"  Deleted original: {input_path.name}")
            
            return True
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Compress images by converting to JPEG')
    parser.add_argument('files', nargs='+', help='Image files to compress')
    parser.add_argument('-q', '--quality', type=int, default=85, 
                       help='JPEG quality 1-100 (default: 85)')
    parser.add_argument('-w', '--width', type=int, default=1200,
                       help='Maximum width in pixels (default: 1200)')
    
    args = parser.parse_args()
    
    # Validate quality
    if not 1 <= args.quality <= 100:
        print("Error: Quality must be between 1-100")
        sys.exit(1)
    
    success_count = 0
    total_count = len(args.files)
    
    print(f"Compressing {total_count} files with quality={args.quality}, max_width={args.width}")
    print("-" * 50)
    
    for file_path in args.files:
        if compress_image(file_path, args.quality, args.width):
            success_count += 1
        print()
    
    print(f"Successfully compressed {success_count}/{total_count} files")

if __name__ == '__main__':
    main()