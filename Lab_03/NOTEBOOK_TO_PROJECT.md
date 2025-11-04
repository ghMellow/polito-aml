# Dal Notebook al Codice Strutturato - Guida alla Trasformazione

Questo documento spiega come il codice del notebook `MLVM_lab3_warmup_lab02style.ipynb` è stato trasformato in un progetto Python strutturato.

---

## 📓 Notebook (Codice Non Strutturato)

### Struttura Originale
Il notebook conteneva **tutte le celle in sequenza**:
1. Import
2. Definizione modello
3. Download dataset
4. Preparazione dataset
5. Classe Dataset
6. Trasformazioni
7. Dataloader
8. Training setup
9. Funzione train()
10. Funzione test()
11. Esecuzione training
12. Valutazione

### ❌ Problemi del Notebook

1. **Non Riutilizzabile**: Difficile usare il codice in altri progetti
2. **Non Modulare**: Tutto in un unico file
3. **Difficile da Testare**: Non puoi testare singoli componenti
4. **Nessuna CLI**: Devi modificare celle per cambiare parametri
5. **Difficile da Versionare**: Git non gestisce bene i notebook
6. **Nessun Checkpoint Management**: Salvataggio manuale
7. **Nessuna Integrazione CI/CD**: Impossibile automatizzare

---

## 🏗️ Progetto Strutturato

### Nuova Organizzazione

```
Notebook Cell → File Strutturato
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Celle 1-2: Import + Model
    ↓
models/vgg_finetuning.py
    - create_vgg16_model()
    - unfreeze_layers()
    - count_trainable_parameters()

Celle 3-4: Download + Preparazione Dataset
    ↓
dataset/custom_dataset.py
    - CustomImageDataset class
    - create_annotations_csv()

Celle 5-6: Trasformazioni
    ↓
utils/transforms.py
    - get_train_transforms()
    - get_val_test_transforms()

Celle 7-9: Visualizzazione
    ↓
utils/visualization.py
    - denormalize()
    - visualize_batch()
    - plot_training_history()

Celle 10: Statistiche
    ↓
utils/metrics.py
    - calculate_dataset_stats()
    - plot_class_distribution()

Celle 11-13: Training Setup + Train Function
    ↓
train.py
    - parse_args()
    - train_one_epoch()
    - validate()
    - train()
    - main()

Celle 14-16: Test Function
    ↓
eval.py
    - test()
    - main()
```

---

## 🔄 Trasformazioni Principali

### 1. **Dataset Module**

**Prima (Notebook - Cella singola):**
```python
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        # ... resto del codice ...
```

**Dopo (dataset/custom_dataset.py):**
```python
"""
Custom Dataset implementation for image classification tasks.
Based on MLVM Lab3 warmup notebook.
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    """
    Custom Dataset for loading images from a CSV annotation file.
    
    Args:
        annotations_file (str): Path to CSV with annotations
        img_dir (str): Directory containing images
        transform (callable, optional): Transform to apply to images
        target_transform (callable, optional): Transform for labels
    """
    # ... implementazione ...
```

**✅ Vantaggi:**
- Docstring completi
- Type hints
- Import organizzati
- Riutilizzabile in altri progetti

---

### 2. **Model Module**

**Prima (Notebook - Celle sparse):**
```python
# Cella 1
model_ft = models.vgg16(weights='IMAGENET1K_V1')
num_classes = 2
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

# Cella 2
for param in model_ft.parameters():
    param.requires_grad = False
# ...
```

**Dopo (models/vgg_finetuning.py):**
```python
def create_vgg16_model(num_classes=2, pretrained=True, freeze_base=True):
    """
    Create a VGG16 model for fine-tuning.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use ImageNet weights
        freeze_base (bool): Freeze convolutional layers
    
    Returns:
        torch.nn.Module: The modified VGG16 model
    """
    # ... implementazione pulita ...
```

**✅ Vantaggi:**
- Funzione parametrica
- Gestione opzioni via parametri
- Facilmente testabile

---

### 3. **Training Logic**

**Prima (Notebook - Cella con parametri hard-coded):**
```python
lr = 0.0001
num_epochs = 10
batch_size = 128

def train(model, trainloader, validloader, device, optimizer, criterion, epochs=10):
    # ... codice training ...
    
# Esecuzione diretta
train_losses, valid_losses = train(model_ft, trainloader, validloader, ...)
```

**Dopo (train.py con argparse):**
```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    # ... altri parametri ...
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Usa args per configurare tutto
```

**Ora puoi fare:**
```bash
python train.py --epochs 20 --batch_size 64 --lr 0.0005
```

**✅ Vantaggi:**
- CLI professionale
- No modifica codice per cambiare parametri
- Scriptabile e automatizzabile

---

### 4. **Transforms (IMPORTANTE!)**

**Prima (Notebook - Trasformazioni corrette ma non riutilizzabili):**
```python
# Cella 1: Train transforms
train_transform = transforms.Compose([...])

# Cella 2: Val/Test transforms
val_test_transform = transforms.Compose([...])
```

**Dopo (utils/transforms.py):**
```python
def get_train_transforms(image_size=224):
    """Get transforms for TRAINING SET with data augmentation."""
    return transforms.Compose([...])

def get_val_test_transforms(image_size=224):
    """Get transforms for VALIDATION/TEST SET WITHOUT augmentation."""
    return transforms.Compose([...])
```

**✅ Vantaggi:**
- Riutilizzabile
- Documentato
- Parametrico (image_size)
- No duplicazione codice

---

### 5. **Checkpoint Management**

**Prima (Notebook - Salvataggio Manuale):**
```python
# Alla fine del training, devi manualmente salvare:
torch.save(model.state_dict(), 'model.pth')
```

**Dopo (train.py - Automatico):**
```python
# Durante il training:
if (epoch + 1) % args.save_every == 0 or valid_loss < best_val_loss:
    checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'valid_accuracy': valid_acc,
    }, checkpoint_path)
    
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(..., 'best_model.pth')  # Salva best model
```

**✅ Vantaggi:**
- Salvataggio automatico
- Best model tracking
- Metadati salvati (epoch, loss, accuracy)
- Riproducibile

---

### 6. **Evaluation Script Separato**

**Prima (Notebook - Test inline):**
```python
# Cella finale
def test(model, device, test_loader):
    # ... codice test ...

test_accuracy = test(model_ft, device, testloader)
print(f'Test Accuracy: {test_accuracy}')
```

**Dopo (eval.py):**
```python
# Script separato con CLI
python eval.py --checkpoint ./checkpoints/best_model.pth --data_dir ./data
```

**✅ Vantaggi:**
- Valutazione indipendente
- Testa modelli salvati
- Non serve rifare training
- Automatizzabile

---

## 📊 Comparazione Completa

| Aspetto | Notebook | Progetto Strutturato |
|---------|----------|---------------------|
| **File Organization** | 1 file | 10+ file modulari |
| **Riutilizzabilità** | ❌ Bassa | ✅ Alta |
| **Testabilità** | ❌ Difficile | ✅ Facile |
| **CLI Support** | ❌ No | ✅ Sì (argparse) |
| **Version Control** | ❌ Problematico | ✅ Pulito |
| **Checkpoint Management** | ❌ Manuale | ✅ Automatico |
| **Logging (Wandb)** | ❌ No | ✅ Integrato |
| **Documentation** | ❌ Minima | ✅ Completa |
| **CI/CD Ready** | ❌ No | ✅ Sì |
| **Reproducibility** | ⚠️ Media | ✅ Alta |
| **Team Collaboration** | ❌ Difficile | ✅ Facile |

---

## 🎯 Best Practices Aggiunte

### 1. **Docstring e Documentation**
Ogni funzione/classe ha docstring con:
- Descrizione
- Args
- Returns
- Example usage (dove appropriato)

### 2. **Type Hints** (opzionale ma consigliato)
```python
def train_one_epoch(model: nn.Module, 
                   trainloader: DataLoader, 
                   device: torch.device,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module) -> float:
    """..."""
```

### 3. **Separation of Concerns**
- `dataset/`: Solo logica dataset
- `models/`: Solo architetture
- `utils/`: Utilities riutilizzabili
- `train.py`: Solo training logic
- `eval.py`: Solo evaluation logic

### 4. **Configuration Management**
- `config.py`: Configurazione centralizzata
- `requirements.txt`: Dipendenze versionate

### 5. **Logging e Monitoring**
```python
if args.use_wandb:
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'valid_loss': valid_loss
    })
```

---

## 🚀 Workflow Consigliato

### Development Workflow

1. **Prototipazione in Notebook** (come Lab3)
   - Sperimenta velocemente
   - Visualizza risultati
   - Itera rapidamente

2. **Strutturazione del Codice** (questo progetto)
   - Identifica componenti riutilizzabili
   - Crea moduli separati
   - Aggiungi CLI

3. **Production Deployment**
   - CI/CD pipeline
   - Docker containers
   - Model serving

### Quando Usare Notebook vs Progetto Strutturato

**✅ Usa Notebook per:**
- Esplorazione dati
- Prototipazione rapida
- Visualizzazioni interattive
- Presentazioni/Report
- Teaching (come i Lab)

**✅ Usa Progetto Strutturato per:**
- Training modelli production
- Esperimenti riproducibili
- Team collaboration
- Deployment
- Progetti a lungo termine

---

## 📝 Summary

**Trasformazione Notebook → Progetto Strutturato:**

1. ✅ **Modularizzazione**: Dividi in file logici
2. ✅ **CLI Interface**: Usa argparse
3. ✅ **Checkpoint Management**: Salvataggio automatico
4. ✅ **Logging**: Integra Wandb
5. ✅ **Documentation**: README + docstring
6. ✅ **Reproducibility**: requirements.txt + config
7. ✅ **Separation**: train.py + eval.py
8. ✅ **Git-friendly**: Proper .gitignore

**Risultato:**
Codice professionale, riutilizzabile, testabile e production-ready! 🎉
