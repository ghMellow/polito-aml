# Configuration file for project settings

# Project Info
PROJECT_NAME = "VGG16 Fine-tuning"
VERSION = "1.0.0"

# Data Settings
DEFAULT_DATA_DIR = "./data"
TRAIN_SUBDIR = "training_set/training_set"
TEST_SUBDIR = "test_set/test_set"

# Model Settings
NUM_CLASSES = 2
IMAGE_SIZE = 224
CLASS_NAMES = ['Cat', 'Dog']

# ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training Settings
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.0001
DEFAULT_MOMENTUM = 0.9
DEFAULT_EPOCHS = 10
DEFAULT_VAL_SPLIT = 0.2

# Checkpoint Settings
CHECKPOINT_DIR = "./checkpoints"
SAVE_EVERY_N_EPOCHS = 5

# Wandb Settings
WANDB_PROJECT = "vgg16-finetuning"
WANDB_ENTITY = None  # Set your wandb entity/username here
