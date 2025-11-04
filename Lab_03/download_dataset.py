"""
Download Cats vs Dogs dataset from Kaggle.

Usage:
    python download_dataset.py --output_dir ./data
"""
import os
import argparse
import shutil


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download Cats vs Dogs dataset')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for dataset')
    return parser.parse_args()


def download_dataset(output_dir):
    """
    Download the Cats vs Dogs dataset from Kaggle.
    
    Args:
        output_dir (str): Directory to save the dataset
    """
    try:
        import kagglehub
    except ImportError:
        print("❌ Error: kagglehub not installed")
        print("Install it with: pip install kagglehub")
        return False
    
    print("\n" + "="*60)
    print("DOWNLOADING CATS VS DOGS DATASET FROM KAGGLE")
    print("="*60 + "\n")
    
    # Download dataset
    print("Downloading dataset (this may take a while)...")
    path = kagglehub.dataset_download("tongpython/cat-and-dog")
    
    print(f"\n✓ Dataset downloaded to: {path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy files to output directory (use copy instead of move for read-only filesystems)
    print(f"\n✓ Copying files to {output_dir}...")
    
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(output_dir, item)
        
        if os.path.exists(dst):
            print(f"  - Skipping {item} (already exists)")
        else:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
                print(f"  - Copied {item}/")
            else:
                shutil.copy2(src, dst)
                print(f"  - Copied {item}")
    
    print("\n" + "="*60)
    print("DATASET READY!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"{output_dir}/")
    print(f"├── training_set/")
    print(f"│   └── training_set/")
    print(f"│       ├── cats/")
    print(f"│       └── dogs/")
    print(f"└── test_set/")
    print(f"    └── test_set/")
    print(f"        ├── cats/")
    print(f"        └── dogs/")
    print()
    
    return True


def main():
    """Main function."""
    args = parse_args()
    success = download_dataset(args.output_dir)
    
    if success:
        print("You can now run training with:")
        print(f"  python train.py --data_dir {args.output_dir}")
    else:
        print("\n❌ Dataset download failed")


if __name__ == '__main__':
    main()
