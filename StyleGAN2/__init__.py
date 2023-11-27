# use file as module
from .model import StyleGAN
from .legacy import load_network_pkl

__all__ = ['StyleGAN', 'load_network_pkl']