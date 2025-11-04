# 🚀 Quick Start - Google Colab

## Caricamento Veloce del Progetto su Colab

### Opzione 1: Upload Notebook Diretto

1. **Vai su Google Colab**: https://colab.research.google.com
2. **Upload il notebook**:
   - Click su `File` → `Upload notebook`
   - Carica il file `colab_training.ipynb` da questo repository
3. **Abilita GPU**:
   - Click su `Runtime` → `Change runtime type`
   - Seleziona `T4 GPU` o `L4 GPU`
   - Click `Save`
4. **Esegui le celle** una alla volta partendo dall'inizio

---

### Opzione 2: Apri da GitHub (Raccomandato)

1. **Push il progetto su GitHub**:
```bash
cd /Users/nicolotermine/zMellow/GitHub-Poli/Polito/polito-aml-project_skeleton
git add .
git commit -m "Add Colab notebook with Wandb integration"
git push origin main
```

2. **Apri su Colab**:
   - Vai su: `https://colab.research.google.com/github/ghMellow/polito-aml-project_skeleton/blob/main/colab_training.ipynb`
   - Oppure da Colab: `File` → `Open notebook` → tab `GitHub` → cerca `ghMellow/polito-aml-project_skeleton`

3. **Abilita GPU** come sopra

---

### Opzione 3: Quick One-Liner (Super Veloce)

Crea un nuovo notebook su Colab e copia questa singola cella:

```python
# Setup completo in una cella - COPIA E INCOLLA SU COLAB
!git clone https://github.com/ghMellow/polito-aml-project_skeleton.git
%cd polito-aml-project_skeleton
!pip install -q -r requirements.txt

# Wandb login
import wandb
wandb.login()

# Download dataset
import kagglehub, shutil, os
path = kagglehub.dataset_download("tongpython/cat-and-dog")
os.makedirs('./data', exist_ok=True)
for item in os.listdir(path):
    src, dst = os.path.join(path, item), os.path.join('./data', item)
    if not os.path.exists(dst): shutil.move(src, dst)

# Train con Wandb
!python train.py \
    --data_dir ./data \
    --epochs 10 \
    --batch_size 128 \
    --lr 0.0001 \
    --use_wandb \
    --wandb_project "vgg16-cats-vs-dogs-colab"

# Evaluate
!python eval.py --checkpoint ./checkpoints/best_model.pth --data_dir ./data

print("\\n✅ Training e evaluation completati!")
print("Visualizza risultati su: https://wandb.ai")
```

**Fatto!** In una singola cella hai:
- ✅ Clone del repository
- ✅ Install dipendenze
- ✅ Wandb login
- ✅ Download dataset
- ✅ Training completo
- ✅ Evaluation

---

## 📋 Checklist Pre-Training

Prima di lanciare il training su Colab:

- [ ] **GPU Abilitata**: Runtime → Change runtime type → GPU
- [ ] **Wandb Account**: Registrati su [wandb.ai](https://wandb.ai)
- [ ] **Wandb API Key**: Copia da [wandb.ai/authorize](https://wandb.ai/authorize)
- [ ] **Kaggle Auth** (se richiesto): Potrebbe chiederti credenziali Kaggle per il dataset

---

## 🎯 Comandi Utili Durante il Training

### Controllare GPU
```python
!nvidia-smi
```

### Vedere i file
```python
!ls -la ./checkpoints
!ls -la ./data
```

### Monitoring Real-time
```python
# In una cella separata mentre il training è in corso
import time
for i in range(10):
    !tail -n 5 training.log  # se loggi su file
    time.sleep(30)
```

### Salvare su Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r ./checkpoints /content/drive/MyDrive/vgg16_checkpoints
```

---

## ⚡ Tips per Colab

### 1. Evita Disconnessioni
```javascript
// Apri Console del Browser (F12) e incolla:
function ClickConnect(){
console.log("Working"); 
document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect,60000)
```

### 2. Riduci Batch Size se OOM (Out of Memory)
```python
!python train.py --batch_size 64  # invece di 128
```

### 3. Training più Veloce (Less Epochs)
```python
!python train.py --epochs 5  # invece di 10
```

### 4. Mixed Precision Training (Più veloce su GPU moderne)
Aggiungi in `train.py` dopo il commit:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

## 📊 Visualizzare Risultati

### Durante Training
- Apri il link Wandb che appare nell'output
- Vedrai loss curves in tempo reale

### Dopo Training
1. Vai su [wandb.ai](https://wandb.ai)
2. Seleziona il tuo progetto
3. Esplora:
   - 📈 Charts (loss, accuracy)
   - 🔍 System metrics (GPU usage)
   - 📁 Artifacts (model checkpoints)

---

## 🐛 Troubleshooting Comune

### "ModuleNotFoundError"
```python
!pip install -r requirements.txt
```

### "CUDA out of memory"
```python
# Riduci batch size
!python train.py --batch_size 32
```

### "Kaggle authentication error"
```python
# Setup Kaggle credentials
!mkdir -p ~/.kaggle
!echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```

### "Wandb login failed"
```python
# Login manuale
import wandb
wandb.login(key="YOUR_API_KEY")
```

---

## 🎓 Esempio Output Atteso

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
  ✓ New best model saved!
Epoch [2/10] Train Loss: 0.1876 | Valid Loss: 0.1654 | Valid Acc: 0.9456
  ✓ New best model saved!
...

============================================================
TRAINING COMPLETED
Best Validation Loss: 0.1234
============================================================

Test set: Average loss: 0.1156, Accuracy: 2450/2500 (98%)

✅ Training completed! Checkpoints saved in: ./checkpoints
```

---

## 🎉 Prossimi Passi

Dopo il primo training di successo:

1. **Esperimenta con hyperparameters**:
```python
!python train.py --lr 0.001 --batch_size 64 --epochs 15
```

2. **Prova Full Fine-tuning** (senza --freeze_base):
```python
!python train.py --data_dir ./data --epochs 10 --use_wandb
# Nota: senza --freeze_base allena TUTTO il network
```

3. **Confronta runs su Wandb**: Tutti i tuoi esperimenti saranno tracciati!

4. **Scarica il best model**:
```python
from google.colab import files
files.download('./checkpoints/best_model.pth')
```

---

## 📚 Risorse

- **Wandb Docs**: https://docs.wandb.ai
- **Colab Docs**: https://colab.research.google.com/notebooks/intro.ipynb
- **PyTorch Transfer Learning**: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **VGG Paper**: https://arxiv.org/abs/1409.1556

---

## 💡 Pro Tips

1. **Traccia tutto con Wandb**: Ogni esperimento è memorizzato
2. **Usa GPU Colab Pro**: Se hai bisogno di training più lunghi
3. **Salva checkpoint regolarmente**: Usa Google Drive per backup
4. **Testa prima con pochi epochs**: `--epochs 2` per verificare che tutto funzioni

**Buon Training! 🚀**
