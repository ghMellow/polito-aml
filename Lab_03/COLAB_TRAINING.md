# VGG16 Fine-tuning - Google Colab Training Notebook

Questo notebook permette di eseguire il training del progetto strutturato direttamente da Google Colab con integrazione Wandb.

## 🚀 Setup e Training su Google Colab

### 1. Clone del Repository e Setup

```python
# Cell 1: Clone repository
!git clone https://github.com/ghMellow/polito-aml-project_skeleton.git
%cd polito-aml-project_skeleton
```

### 2. Install Dependencies

```python
# Cell 2: Install requirements
!pip install -q torch torchvision numpy pandas Pillow torchsummary wandb kagglehub matplotlib
```

### 3. Wandb Authentication

```python
# Cell 3: Login to Wandb
import wandb
wandb.login()
# Inserisci la tua API key quando richiesto
```

### 4. Download Dataset

```python
# Cell 4: Download Cats vs Dogs dataset
import kagglehub

# Download dataset
path = kagglehub.dataset_download("tongpython/cat-and-dog")
print(f"Dataset downloaded to: {path}")

# Move to data folder
import shutil
import os

os.makedirs('./data', exist_ok=True)
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join('./data', item)
    if not os.path.exists(dst):
        shutil.move(src, dst)
        
print("✓ Dataset ready!")
```

### 5. Training con Wandb (Metodo CLI)

```python
# Cell 5: Training usando train.py
!python train.py \
    --data_dir ./data \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.0001 \
    --use_wandb \
    --wandb_project "vgg16-cats-vs-dogs"
```

### 6. Evaluation

```python
# Cell 6: Evaluate best model
!python eval.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data
```

---

## 🎯 Training Programmatico (Stile Slide Professore)

Se preferisci controllare tutto dal notebook seguendo lo stile dello slide:

### Cell 1: Imports

```python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Import custom modules
from dataset import CustomImageDataset, create_annotations_csv
from models import create_vgg16_model, count_trainable_parameters
from utils import get_train_transforms, get_val_test_transforms, plot_training_history

import wandb
```

### Cell 2: Wandb Setup (Seguendo Slide Professore)

```python
# 1. Start a W&B run (Slide professore)
wandb.init(
    project='vgg16-cats-vs-dogs',
    name='feature-extraction-run',
    tags=['vgg16', 'transfer-learning', 'cats-vs-dogs']
)

# 2. Save model inputs and hyperparameters (Slide professore)
config = wandb.config
config.learning_rate = 0.0001
config.batch_size = 128
config.epochs = 10
config.momentum = 0.9
config.val_split = 0.2
config.architecture = 'VGG16'
config.mode = 'feature_extraction'

print("✓ Wandb initialized")
print(f"  - Project: {wandb.run.project}")
print(f"  - Run name: {wandb.run.name}")
print(f"  - Config: {dict(config)}")
```

### Cell 3: Setup Device e Dataset

```python
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

# Paths
data_dir = './data'
train_path = os.path.join(data_dir, 'training_set/training_set')
test_path = os.path.join(data_dir, 'test_set/test_set')

# Create annotations if needed
if not os.path.exists('train_annotations.csv'):
    create_annotations_csv(train_path, 'train_annotations.csv')
if not os.path.exists('test_annotations.csv'):
    create_annotations_csv(test_path, 'test_annotations.csv')

# Transforms
train_transform = get_train_transforms()
val_transform = get_val_test_transforms()

# Datasets
train_dataset = CustomImageDataset('train_annotations.csv', train_path, transform=train_transform)
valid_dataset = CustomImageDataset('train_annotations.csv', train_path, transform=val_transform)

# Train/Val split
indices = list(range(len(train_dataset)))
split = int(np.floor(config.val_split * len(train_dataset)))
train_sample = SubsetRandomSampler(indices[split:])
valid_sample = SubsetRandomSampler(indices[:split])

# Dataloaders
trainloader = DataLoader(train_dataset, sampler=train_sample, batch_size=config.batch_size)
validloader = DataLoader(valid_dataset, sampler=valid_sample, batch_size=config.batch_size)

print(f"✓ Datasets ready")
print(f"  - Training samples: {len(indices[split:])}")
print(f"  - Validation samples: {len(indices[:split])}")
```

### Cell 4: Create Model

```python
# Create VGG16 model
model = create_vgg16_model(num_classes=2, pretrained=True, freeze_base=True)
model = model.to(device)

trainable, total = count_trainable_parameters(model)
print(f"✓ Model created")
print(f"  - Trainable parameters: {trainable:,} / {total:,}")

# Log model architecture to wandb
wandb.watch(model, log='all', log_freq=100)
```

### Cell 5: Training Setup

```python
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
parameters_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.SGD(parameters_to_optimize, lr=config.learning_rate, momentum=config.momentum)

print("✓ Training setup complete")
print(f"  - Loss: CrossEntropyLoss")
print(f"  - Optimizer: SGD (lr={config.learning_rate}, momentum={config.momentum})")
```

### Cell 6: Training Loop (Con Wandb - Stile Slide)

```python
# Training loop
best_val_loss = float('inf')

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

for epoch in range(config.epochs):
    # TRAINING
    model.train()
    running_loss = 0.0
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(trainloader)
    
    # VALIDATION
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    valid_loss = running_loss / len(validloader)
    valid_acc = correct / total
    
    # 3. Log metrics over time to visualize performance (SLIDE PROFESSORE)
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_acc,
        "learning_rate": config.learning_rate
    })
    
    print(f"Epoch [{epoch+1}/{config.epochs}] "
          f"Train Loss: {train_loss:.4f} | "
          f"Valid Loss: {valid_loss:.4f} | "
          f"Valid Acc: {valid_acc:.4f}")
    
    # Save best model
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'valid_accuracy': valid_acc,
        }, './checkpoints/best_model.pth')
        print(f"  ✓ New best model saved!")

print("\n" + "="*60)
print("TRAINING COMPLETED")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print("="*60)

# Finish wandb run
wandb.finish()
```

### Cell 7: Evaluation on Test Set

```python
# Evaluate on test set
from eval import test

test_dataset = CustomImageDataset('test_annotations.csv', test_path, transform=val_transform)
testloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Load best model
checkpoint = torch.load('./checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Test
test_loss, test_accuracy, correct, total = test(model, device, testloader)

print(f"\n{'='*60}")
print("TEST SET RESULTS")
print(f"{'='*60}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {correct}/{total} ({100. * test_accuracy:.2f}%)")
print(f"{'='*60}\n")
```

---

## 📊 Visualizza Risultati su Wandb

Dopo il training, vai su [wandb.ai](https://wandb.ai) per vedere:
- Loss curves (train vs validation)
- Accuracy curve
- Model gradients
- System metrics (GPU, memory)

---

## 🎯 Versione Semplificata (Quick Start)

Se vuoi solo fare training veloce su Colab:

```python
# Setup completo in una cella
!git clone https://github.com/ghMellow/polito-aml-project_skeleton.git
%cd polito-aml-project_skeleton
!pip install -q torch torchvision wandb kagglehub matplotlib

# Download dataset
import kagglehub
path = kagglehub.dataset_download("tongpython/cat-and-dog")

# Train con wandb
import wandb
wandb.login()

!python train.py \
    --data_dir {path} \
    --epochs 10 \
    --batch_size 128 \
    --use_wandb \
    --wandb_project "vgg16-colab-run"
```

---

## 💡 Note per Google Colab

1. **GPU**: Assicurati di abilitare GPU in Runtime → Change runtime type → GPU
2. **Wandb Login**: Copia la API key da [wandb.ai/authorize](https://wandb.ai/authorize)
3. **Data Persistence**: I dati in Colab vengono persi quando la sessione termina. Salva i checkpoint su Google Drive se necessario:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copia checkpoint
!cp -r ./checkpoints /content/drive/MyDrive/vgg16_checkpoints
```

4. **Batch Size**: Se hai memory errors, riduci il batch size:
```python
!python train.py --batch_size 64
```
