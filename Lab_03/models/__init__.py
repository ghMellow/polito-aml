"""
Model architectures for image classification.
"""
from .vgg_finetuning import create_vgg16_model, count_trainable_parameters, unfreeze_layers

__all__ = ['create_vgg16_model', 'count_trainable_parameters', 'unfreeze_layers']
