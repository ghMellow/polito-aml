"""
Example usage of the structured project.
This file shows how to use the modules programmatically (not via CLI).
"""
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# Import custom modules
from dataset import CustomImageDataset, create_annotations_csv
from models import create_vgg16_model, count_trainable_parameters
from utils import get_train_transforms, get_val_test_transforms
from utils import visualize_batch, calculate_dataset_stats

# Import config
import config


def example_1_create_dataset():
    """Example: Create and explore dataset."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Create and Explore Dataset")
    print("="*60 + "\n")
    
    # Create annotations (if needed)
    # create_annotations_csv(
    #     data_dir='./data/training_set/training_set',
    #     output_csv='train_annotations.csv'
    # )
    
    # Create dataset with transforms
    train_transform = get_train_transforms()
    dataset = CustomImageDataset(
        annotations_file='train_annotations.csv',
        img_dir='./data/training_set/training_set',
        transform=train_transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get statistics
    calculate_dataset_stats(dataset, class_names=config.CLASS_NAMES)
    
    # Get a sample
    image, label = dataset[0]
    print(f"\nSample image shape: {image.shape}")
    print(f"Sample label: {label} ({config.CLASS_NAMES[label]})")


def example_2_create_model():
    """Example: Create and inspect model."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Create and Inspect Model")
    print("="*60 + "\n")
    
    # Create model with frozen base (Feature Extraction)
    model = create_vgg16_model(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_base=True
    )
    
    # Count parameters
    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print(f"Percentage trainable: {100 * trainable / total:.2f}%")
    
    # Show model structure
    print("\nModel classifier layers:")
    for i, layer in enumerate(model.classifier):
        print(f"  [{i}] {layer}")


def example_3_dataloaders():
    """Example: Create dataloaders with train/val split."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Create DataLoaders")
    print("="*60 + "\n")
    
    # Create datasets
    train_transform = get_train_transforms()
    val_transform = get_val_test_transforms()
    
    train_dataset = CustomImageDataset(
        annotations_file='train_annotations.csv',
        img_dir='./data/training_set/training_set',
        transform=train_transform
    )
    
    valid_dataset = CustomImageDataset(
        annotations_file='train_annotations.csv',
        img_dir='./data/training_set/training_set',
        transform=val_transform
    )
    
    # Create train/validation split
    indices = list(range(len(train_dataset)))
    split = int(np.floor(config.DEFAULT_VAL_SPLIT * len(train_dataset)))
    train_sample = SubsetRandomSampler(indices[split:])
    valid_sample = SubsetRandomSampler(indices[:split])
    
    # Create dataloaders
    trainloader = DataLoader(
        train_dataset,
        sampler=train_sample,
        batch_size=config.DEFAULT_BATCH_SIZE
    )
    
    validloader = DataLoader(
        valid_dataset,
        sampler=valid_sample,
        batch_size=config.DEFAULT_BATCH_SIZE
    )
    
    print(f"Training samples: {len(indices[split:])}")
    print(f"Validation samples: {len(indices[:split])}")
    print(f"Training batches: {len(trainloader)}")
    print(f"Validation batches: {len(validloader)}")
    
    # Visualize a batch
    images, labels = next(iter(trainloader))
    print(f"\nBatch shape: {images.shape}")
    
    # Visualize (uncomment to show images)
    # visualize_batch(images, labels, class_names=config.CLASS_NAMES)


def example_4_training_setup():
    """Example: Complete training setup."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Training Setup")
    print("="*60 + "\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Model
    model = create_vgg16_model(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_base=True
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    parameters_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
        parameters_to_optimize,
        lr=config.DEFAULT_LR,
        momentum=config.DEFAULT_MOMENTUM
    )
    
    print(f"Loss function: {criterion}")
    print(f"Optimizer: SGD (lr={config.DEFAULT_LR}, momentum={config.DEFAULT_MOMENTUM})")
    
    print("\n✓ Ready for training!")
    print("Run: python train.py --data_dir ./data --epochs 10")


def example_5_load_checkpoint():
    """Example: Load and evaluate a trained model."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Load Checkpoint")
    print("="*60 + "\n")
    
    checkpoint_path = './checkpoints/best_model.pth'
    
    # Create model
    model = create_vgg16_model(num_classes=config.NUM_CLASSES, pretrained=False)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
        print(f"  - Valid Loss: {checkpoint.get('valid_loss', 'N/A'):.4f}")
        print(f"  - Valid Accuracy: {checkpoint.get('valid_accuracy', 'N/A'):.4f}")
        
        print("\n✓ Model ready for evaluation!")
        print("Run: python eval.py --checkpoint ./checkpoints/best_model.pth")
        
    except FileNotFoundError:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Train a model first: python train.py")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "PROJECT USAGE EXAMPLES")
    print("="*70)
    
    # Uncomment examples you want to run
    # Note: Some examples require the dataset to be downloaded
    
    # example_1_create_dataset()
    # example_2_create_model()
    # example_3_dataloaders()
    example_4_training_setup()
    example_5_load_checkpoint()
    
    print("\n" + "="*70)
    print("For more examples, check the README.md and NOTEBOOK_TO_PROJECT.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
