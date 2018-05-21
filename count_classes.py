import argparse
import functools
import os
import sys
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers
from .. import losses
from .. import models
from ..preprocessing.pascal_voc import PascalVocGenerator

train_generator = PascalVocGenerator(
    '/home/chedy/PASCAL/VOCdevkit/VOC2007',
    'trainval',
    batch_size=1,
    image_min_side=500,
    image_max_side=500
)

class_frequency = np.zeros(20)

for i in range(train_generator.size()):
    annotations = train_generator.load_annotations(i)
    for a in annotations:
        label = a[4]
        class_frequency[int(label)] = class_frequency[int(label)] + 1


print class_frequency
