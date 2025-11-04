# Lab03: VGG16 Image Classification - Structured Project

Questo progetto implementa il fine-tuning di VGG16 per classificazione binaria (Cats vs Dogs) seguendo le best practices di strutturazione del codice.

## 📁 Struttura del Progetto

```
polito-aml-project_skeleton/
├── dataset/                  # Dataset classes e utilities
│   ├── __init__.py
│   └── custom_dataset.py    # CustomImageDataset + create_annotations_csv
├── models/                   # Architetture modelli
│   ├── __init__.py
│   └── vgg_finetuning.py    # VGG16 con fine-tuning
├── utils/                    # Utilities di supporto
│   ├── __init__.py
│   ├── transforms.py        # Trasformazioni train/val/test
│   ├── visualization.py     # Plot e visualizzazioni
│   └── metrics.py           # Metriche e statistiche
├── checkpoints/             # Modelli salvati durante training
├── data/                    # Dataset scaricato (non in git)
├── train.py                 # Script di training
├── eval.py                  # Script di evaluation
├── requirements.txt         # Dipendenze pip
└── README.md               # Questo file
```

## 🚀 Setup

### 1. Clona il repository
```bash
git clone <your-repo-url>
cd polito-aml-project_skeleton
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 3. Scarica il dataset
Il dataset Cats vs Dogs può essere scaricato da Kaggle:
```python
import kagglehub
path = kagglehub.dataset_download("tongpython/cat-and-dog")
```

Poi sposta i file nella cartella `data/`:
```bash
mv <downloaded_path>/* ./data/
```

## 🎯 Training

### Training Base (Feature Extraction)
Allena solo l'ultimo layer con i layer base congelati:

```bash
python train.py \
    --data_dir ./data \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.0001 \
    --freeze_base
```

### Training con Wandb Logging
```bash
python train.py \
    --data_dir ./data \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.0001 \
    --freeze_base \
    --use_wandb \
    --wandb_project "my-vgg16-project"
```

### Parametri Disponibili
- `--data_dir`: Path al dataset (default: `./data`)
- `--epochs`: Numero di epoche (default: `10`)
- `--batch_size`: Batch size (default: `128`)
- `--lr`: Learning rate (default: `0.0001`)
- `--momentum`: Momentum per SGD (default: `0.9`)
- `--val_split`: Percentuale validation set (default: `0.2` = 20%)
- `--freeze_base`: Congela i layer base (feature extraction)
- `--checkpoint_dir`: Directory per i checkpoint (default: `./checkpoints`)
- `--use_wandb`: Abilita logging su Wandb
- `--wandb_project`: Nome progetto Wandb

## 📊 Evaluation

Valuta un modello salvato sul test set:

```bash
python eval.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data \
    --batch_size 128
```

### Parametri Disponibili
- `--checkpoint`: Path al checkpoint del modello (richiesto)
- `--data_dir`: Path al dataset (default: `./data`)
- `--batch_size`: Batch size (default: `128`)
- `--num_classes`: Numero di classi (default: `2`)

## 📦 Moduli

### `dataset/`
- **`custom_dataset.py`**: 
  - `CustomImageDataset`: Dataset PyTorch per caricare immagini da CSV
  - `create_annotations_csv()`: Crea file CSV annotations da struttura cartelle

### `models/`
- **`vgg_finetuning.py`**:
  - `create_vgg16_model()`: Crea VGG16 con pre-trained weights
  - `unfreeze_layers()`: Sblocca N layer per fine-tuning progressivo
  - `count_trainable_parameters()`: Conta parametri trainabili

### `utils/`
- **`transforms.py`**:
  - `get_train_transforms()`: Transforms con data augmentation
  - `get_val_test_transforms()`: Transforms SENZA augmentation
  
- **`visualization.py`**:
  - `denormalize()`: Denormalizza tensor per visualizzazione
  - `visualize_batch()`: Visualizza batch di immagini
  - `plot_training_history()`: Plot loss curves
  
- **`metrics.py`**:
  - `calculate_dataset_stats()`: Statistiche dataset
  - `plot_class_distribution()`: Plot distribuzione classi

## 🎓 Dal Notebook al Codice Strutturato

Questo progetto è una versione strutturata del notebook `MLVM_lab3_warmup_lab02style.ipynb`.

### Differenze Principali:

**❌ Notebook (Non Strutturato)**:
- Tutto in un unico file
- Difficile da riutilizzare
- Difficile da testare
- Nessuna modularità

**✅ Progetto Strutturato**:
- Codice organizzato in moduli
- Facilmente riutilizzabile
- Command line interface
- Logging con Wandb
- Checkpoint management
- Separazione train/eval

## 📈 Output del Training

Durante il training vedrai:
```
✓ Using device: cuda

✓ Loading datasets...
  - Training samples: 6400
  - Validation samples: 1600
  - Batch size: 128

✓ Creating model...
  - Trainable parameters: 8,194 / 134,268,738
  - Mode: Feature Extraction

============================================================
STARTING TRAINING
============================================================
Epoch [1/10] Train Loss: 0.2345 | Valid Loss: 0.1987 | Valid Acc: 0.9234
  ✓ New best model saved: ./checkpoints/best_model.pth
...
```

I checkpoint vengono salvati in `./checkpoints/`:
- `checkpoint_epoch_X.pth`: Checkpoint ogni N epoche
- `best_model.pth`: Miglior modello su validation set

## 🔬 Best Practices Implementate

1. ✅ **Modularità**: Codice diviso in moduli riutilizzabili
2. ✅ **Argparse**: Command line interface professionale
3. ✅ **Data Augmentation**: Solo su training set (IMPORTANTE!)
4. ✅ **Checkpoint Management**: Salvataggio automatico best model
5. ✅ **Logging**: Integrazione Wandb opzionale
6. ✅ **Reproducibility**: requirements.txt + seed management
7. ✅ **Documentation**: Docstrings e README completo
8. ✅ **Lab02 Style**: Seguendo lo stile dei lab precedenti

## 🐛 Troubleshooting

### Import Errors
Assicurati di essere nella directory root del progetto:
```bash
cd /path/to/polito-aml-project_skeleton
python train.py ...
```

### CUDA Out of Memory
Riduci il batch size:
```bash
python train.py --batch_size 64
```

### Dataset Non Trovato
Verifica che la struttura sia:
```
data/
├── training_set/
│   └── training_set/
│       ├── cats/
│       └── dogs/
└── test_set/
    └── test_set/
        ├── cats/
        └── dogs/
```

## 📚 Riferimenti

- Lab02: Training loop e test function style
- Lab03: Transfer Learning e Fine-tuning VGG16
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
