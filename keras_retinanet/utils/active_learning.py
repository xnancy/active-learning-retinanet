from __future__ import print_function

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations
from ..utils.eval import _get_detections

import numpy as np
import os

import keras
import cv2
import pickle

# acquisition function values given detections from model and true annotations
def acquisition_function(model, image): 
        """ 
        # TODO: run all elements of training set through stochastic forward pass of model (TODO) and compute acquisition
        """	
	return np.zeros(generator.size())

def create_batch_generator(file_names):
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
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator

def get_next_batch(
    generator,
    model,
    batch_size, score_threshold = 0.05, max_detections=100, save_path=None
):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    nb_MC_samples = 100
    print("building function")
    MC_output_classification = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-1].output]) 
    MC_output_regression = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-2].output]) 

    print(keras.backend.shape(model.layers[-1].output))
    learning_phase = True  # use dropout at test time

    # all of the pool images
    # image_group: # batches x [image dims]
    print("starting image group")
    image_group = generator.load_image_group(np.arange(generator.size()))
    image_group = np.asarray(image_group)
    print("finish image group")
    # image_group = generator.load_image(0)
    # image_group = np.expand_dims(image_group, axis = 0)
    print(image_group.shape)
    # max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
    # image_batch = np.zeros((generator.size(),)+max_shape, dtype=keras.backend.floatx())
    scores = np.zeros((generator.size(),))
   
    for i in range(generator.size()): 
        image = image_group[i]
        image = np.expand_dims(image, axis = 0)
        print(image.shape)
        MC_samples_classification = [MC_output_classification([image, learning_phase])[0] for _ in xrange(nb_MC_samples)]
        MC_samples_classification = np.array(MC_samples_classification)
        MC_samples_regression = [MC_output_regression([image, learning_phase])[0] for _ in xrange(nb_MC_samples)]
        MC_samples_regression = np.array(MC_samples_regression)

        
    # for image_index, image in enumerate(image_group):
        #score[image_index] = acquisition_function()
        # image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image
    # MC_samples_classification: nb_samples x 1 x # anchors x 20 (# classes) 
    # MC_samples_regression: nb_samples x 1 x # anchors x 4 (anchor)
        # compute acquisition as average of the standard BALD acquisition function over all anchors
        MC_samples_classification_squeezed = np.squeeze(MC_samples_classification)
        expected_entropy = - np.mean(np.sum(MC_samples_classification_squeezed * np.log(MC_samples_classification_squeezed + 1e-10), axis=-1), axis=0)  # [batch size]
        expected_p = np.mean(MC_samples_classification_squeezed, axis=0)
        entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
        BALD_acq = entropy_expected_p - expected_entropy
        BALD_acq = np.array(BALD_acq)
        print(BALD_acq)
        scores[i] = np.mean(BALD_acq)

    
    

    # gather all detections and annotations
    # all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    # average_precisions = {}

    # acquisitions = acquisition_function(all_detections, all_annotations, generator)

    # select pool indices with highest acquisition functions 
    # pool_indices = acquisition[0].argsort()[-batch_size:][::-1]

    # return names of pool file
    top_scores = np.argpartition(scores, -batch_size)[-batch_size:] 
    return scores[top_scores]


    """ 
    # process detections and annotations
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        # loop over every sample in the pool 
        for i in range(generator.size()):
            # get the model detections, and ground truth annotations 
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []


            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]
  		
  		# compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision
    """
