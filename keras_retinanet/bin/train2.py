#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import functools
import os
import sys
import warnings
import time 
import random 

import numpy as np
import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.pascal_voc_batch import PascalVocBatchGenerator

from ..utils.anchors import make_shapes_callback, anchor_targets_bbox
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator
from ..utils.eval import _get_annotations, _get_detections
from ..utils.active_learning import get_next_acquisition


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False):
    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    callbacks = []

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from ..callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.1,
        patience = 2,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))

    return callbacks


def create_generators(args):
    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    
    if args.dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            batch_size=1,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator, transform_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('--version',  help='The current dataset version is V3.', default='2017_11')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--fixed-labels', help='Use the exact specified labels.', default=False)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',           help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',   help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--num-acquisitions', help='Number of acquisitions to run', type=int, default=10)
    parser.add_argument('--pool-size', help='Number of images in oracle pool, <5011 for PASCAL', type=int, default=5011)
    parser.add_argument('--validation-size', help='Number of images in validation, <4952 for PASCAL', type=int, default=4952)
    parser.add_argument('--epochs-per-acquisition', help="Number of epochs trained per new acquistion", type=int, default=3)
    parser.add_argument('--initial-train-size', help='Number of images in initial training pool, randomly selected from oracle pool, <pool-size', type=int, default=500)
    parser.add_argument('--acquisition-size', help='Number of images from oracle / acquisition', type = int, default=10)
    parser.add_argument('--initial-training-epochs', help="Numbero of epochs to train initial pool", type = int, default=10)

    return check_args(parser.parse_args(args))

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # check acquisition parameters are valid
    if args.dataset_type == 'pascal': 
        assert(args.pool_size <= 5011)
        assert(args.validation_size <= 4952)
    assert(args.initial_train_size + args.acquisition_size * args.num_acquisitions <= args.pool_size)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator, transform_generator = create_generators(args)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        prediction_model = retinanet_bbox(model=model)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone
        )

    # og_model, og_training_model, og_prediction_model =  
    # print model summary
    print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        compute_anchor_targets = functools.partial(anchor_targets_bbox, shapes_callback=make_shapes_callback(model))
        train_generator.compute_anchor_targets = compute_anchor_targets
        if validation_generator is not None:
            validation_generator.compute_anchor_targets = compute_anchor_targets
    
    image_names = train_generator.image_names
    validation_image_names = validation_generator.image_names
    
    validation_pool_generator = PascalVocBatchGenerator(args.pascal_path, 
        'test', 
        random.sample(validation_image_names, args.validation_size),
        batch_size = args.batch_size,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side)

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_pool_generator,
        args,
    )

    # train_annotations = _get_annotations(train_generator)

    # start training: we use num-acquisitions and batch_size
    # smaller training generator for faster testing, containing only 10 images 
    initial_pool = random.sample(image_names, args.pool_size)
    initial_training_pool_indices = random.sample(range(args.pool_size), args.initial_train_size)
    initial_oracle_pool = np.delete(initial_pool, initial_training_pool_indices)
    initial_training_pool = np.take(initial_pool, initial_training_pool_indices)
    
    oracle_pool_generator = PascalVocBatchGenerator(args.pascal_path, 
        'trainval',
        initial_oracle_pool,
        batch_size=args.batch_size,
        transform_generator=transform_generator,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side)

    training_pool_generator = PascalVocBatchGenerator(args.pascal_path,
        'trainval',
        initial_training_pool, 
        batch_size=args.batch_size,
        transform_generator=transform_generator,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side)

    training_model.fit_generator(
        generator=training_pool_generator,
        epochs=args.initial_training_epochs,
        steps_per_epoch = int(training_pool_generator.size() / args.batch_size),
        verbose=1,
        callbacks=callbacks,
    )

     
    # Classifaction + Regression model functions 
    model_output_classification = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-1].output]) 
    # model_output_regression = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-2].output])
    # feature pyramid output nodes fed into classification submodel 
    model_output_pyramid = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-8].output, model.layers[-7].output,model.layers[-6].output,model.layers[-11].output,model.layers[-5].output])
    # classification submodel from feature pyramid outputs 
    model_pyramid_classification = keras.backend.function([model.layers[-3].get_input_at(0), model.layers[-3].get_input_at(1),model.layers[-3].get_input_at(2),model.layers[-3].get_input_at(3),model.layers[-3].get_input_at(4), keras.backend.learning_phase()], [model.layers[-1].output])

    for i in range(args.num_acquisitions):
        # keras.backend.clear_session()
        print("Elapsed time since last acquisition cycle", time.strftime("%H:%M:%S", time.gmtime(time.time() - acquisition_cycle_start_time)))
        acquisition_cycle_start_time = time.time()

        print("Starting acquisition", i)
        # get next batch to train on based on acquisition function, batch_size = # samples in each acquisition iteration, default is 1 
        top_scores_index, acquired_images = get_next_acquisition(oracle_pool_generator, training_model, model_output_classification, model_output_pyramid, model_pyramid_classification, args.acquisition_size)

        # generator that feeds acquisition samples 1-by-1 for training in model.fit  
        acquisition_end_time = time.time()
        print("Time to get acquisitions", time.strftime("%H:%M:%S", acquisition_end_time- acquisition_cycle_start_time))
        
        print("Creating new oracle pool and training sets")
        new_oracle_pool = np.delete(oracle_pool_generator.image_names, top_scores_index) 
        new_training_pool = np.concatenate(training_pool_generator.image_names, acquired_images)

        oracle_pool_generator = PascalVocBatchGenerator(
            args.pascal_path, 
            'trainval',
            new_oracle_pool, 
            batch_size=args.batch_size, 
            transform_generator=transform_generator,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )

        training_pool_generator = PascalVocBatchGenerator(args.pascal_path,
            'trainval',
            new_training_pool,
            batch_size=args.batch_size,
            transform_generator=transform_generator,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side)

        generator_end_time = time.time()

        print("Time to create new generators", time.strftime("%H:%M:%S", generator_end_time - acquisition_end_time))


        print("Starting training", i)
        
        training_model.fit_generator(
            generator=training_pool_generator,
            epochs=args.epochs-per-acquisition,
            steps_per_epoch=int(args.training_pool_generator.size() / args.batch_size),
            verbose=1,
            callbacks=callbacks,
        )
        
        training_end_time = time.time() 
        print("Training time", time.strftime("%H:%M:%S", time.gmtime(training_end_time - generator_tned_time)))
              
if __name__ == '__main__':
    main()
