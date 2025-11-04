"""
Evaluation script for trained models on test set.
Based on MLVM Lab3 warmup notebook - structured version.

Usage:
    python eval.py --checkpoint ./checkpoints/best_model.pth --data_dir <path>
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import custom modules
from dataset import CustomImageDataset, create_annotations_csv
from models import create_vgg16_model
from utils import get_val_test_transforms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of output classes')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to dataset directory')
    parser.add_argument('--test_dir', type=str, default='test_set/test_set',
                       help='Relative path to test set')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    
    return parser.parse_args()


def test(model, device, test_loader):
    """
    Test the model on test set, following Lab02 implementation.
    
    Args:
        model: PyTorch model to test
        device: Device to run on (cpu or cuda)
        test_loader: DataLoader for test set
    
    Returns:
        tuple: (test_loss, test_accuracy, correct, total)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    # ITERATE DATALOADER
    with torch.no_grad():
        for data, target in test_loader:
            batch_size = data.shape[0]
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # Sanity check
            pred = pred.view(batch_size)  # [bs,]
            target = target.view(batch_size)  # [bs,]
            
            # Compute prediction ok
            batch_pred_ok = pred.eq(target).sum().item()
            correct += batch_pred_ok
            total += batch_size
    
    test_loss /= total
    test_accuracy = correct / total
    
    return test_loss, test_accuracy, correct, total


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    # Prepare dataset path
    test_path = os.path.join(args.data_dir, args.test_dir)
    
    # Create annotation file if it doesn't exist
    if not os.path.exists('test_annotations.csv'):
        print("\n✓ Creating test annotations...")
        create_annotations_csv(test_path, 'test_annotations.csv')
    
    # Get transforms (no augmentation for test!)
    test_transform = get_val_test_transforms()
    
    # Create test dataset
    print("\n✓ Loading test dataset...")
    test_dataset = CustomImageDataset(
        annotations_file='test_annotations.csv',
        img_dir=test_path,
        transform=test_transform
    )
    
    # Create test dataloader
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Test batches: {len(testloader)}")
    
    # Create model
    print("\n✓ Loading model...")
    model = create_vgg16_model(num_classes=args.num_classes, pretrained=False, freeze_base=False)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"  - Loaded checkpoint from: {args.checkpoint}")
    if 'epoch' in checkpoint:
        print(f"  - Checkpoint epoch: {checkpoint['epoch']}")
    if 'valid_accuracy' in checkpoint:
        print(f"  - Validation accuracy at checkpoint: {checkpoint['valid_accuracy']:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    test_loss, test_accuracy, correct, total = test(model, device, testloader)
    
    print(f"\nTest Results:")
    print(f"  - Average loss: {test_loss:.4f}")
    print(f"  - Accuracy: {correct}/{total} ({100. * test_accuracy:.2f}%)")
    print(f"  - Error rate: {100. * (1 - test_accuracy):.2f}%")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60 + "\n")
    
    return test_accuracy


if __name__ == '__main__':
    main()
