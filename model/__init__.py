from .original_model import OriginalModel, VGG
from .fusion_model import FusionlModel
from .fusion_1 import Fusion1, Fusion1Full, Original1Full, Original1
from .fusion_2 import Fusion2
from .vgg16 import VGG16

__all__ = ['OriginalModel', 'VGG', 'FusionModel', 'Fusion1',
           'Fusion1Full', 'Original1Full', 'Original1', 'Fusion2', 'VGG16']
