from __future__ import print_function

from .anchors import compute_overlap
from .visualization import draw_detections, draw_annotations
from ..utils.eval import _get_detections

import numpy as np
import os
import random 

import keras
import cv2
import pickle

def acquisition_function(image, model, model_output_classification, model_output_regression,  model_output_pyramid, model_pyramid_classification, model_pyramid_regression,  nb_MC_samples = 20, alpha=0.25, gamma = 2): 
        # Classifaction + Regression model functions 
        # model_output_classification = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-1].output]) 
        # model_output_regression = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-2].output])
        # feature pyramid output nodes fed into classification submodel 
        # model_output_pyramid = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-8].output, model.layers[-7].output,model.layers[-6].output,model.layers[-11].output,model.layers[-5].output])
        # classification submodel from feature pyramid outputs 
        # model_pyramid_classification = keras.backend.function([model.layers[-3].get_input_at(0), model.layers[-3].get_input_at(1),model.layers[-3].get_input_at(2),model.layers[-3].get_input_at(3),model.layers[-3].get_input_at(4), keras.backend.learning_phase()], [model.layers[-1].output])
        """ 
        learning_phase = True  # use dropout at test time

        # Classification + Regression MC functions 
        pyramid_output = model_output_pyramid([image, learning_phase])
        # tiled_input = np.tile(pyramid_output, (20,1,1,1) )
        # MC_samples_classification = [model_pyramid_classification([pyramid_output[0], pyramid_output[1], pyramid_output[2], pyramid_output[3], pyramid_output[4], learning_phase])[0] for _ in xrange(nb_MC_samples)]
        MC_samples_classification = model_pyramid_classification([np.tile(pyramid_output[0], (20,1,1,1)), np.tile(pyramid_output[1], (20,1,1,1)), np.tile(pyramid_output[2],(20,1,1,1)), np.tile(pyramid_output[3],(20,1,1,1)), np.tile(pyramid_output[4],(20,1,1,1)), learning_phase])
        MC_samples_classification = np.array(MC_samples_classification)
        # MC_samples_regression = [model_output_regression([image, learning_phase])[0] for _ in xrange(nb_MC_samples)]
        # MC_samples_regression = np.array(MC_samples_regression)
        MC_samples_classification_squeezed = np.squeeze(MC_samples_classification)
        # compute BALD Acquisition using MC samples on image 
        expected_entropy = - np.mean(np.sum(MC_samples_classification_squeezed * np.log(MC_samples_classification_squeezed + 1e-10), axis=-1), axis=0)  # [batch size]
         expected_p = np.mean(MC_samples_classification_squeezed, axis=0)
        entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
        BALD_acq = entropy_expected_p - expected_entropy
        BALD_acq = np.array(BALD_acq)
        
        return np.mean(BALD_acq)
        """
        nb_MC_samples = 30
        learning_phase = True 
        pyramid_output = model_output_pyramid([image, learning_phase])
        MC_samples_classification = model_pyramid_classification([np.tile(pyramid_output[0], (nb_MC_samples,1,1,1)), np.tile(pyramid_output[1], (nb_MC_samples,1,1,1)), np.tile(pyramid_output[2],(nb_MC_samples,1,1,1)), np.tile(pyramid_output[3],(nb_MC_samples,1,1,1)), np.tile(pyramid_output[4],(nb_MC_samples,1,1,1)), learning_phase])
        MC_samples_regression = model_pyramid_regression([np.tile(pyramid_output[0], (nb_MC_samples,1,1,1)), np.tile(pyramid_output[1], (nb_MC_samples,1,1,1)), np.tile(pyramid_output[2],(nb_MC_samples,1,1,1)), np.tile(pyramid_output[3],(nb_MC_samples,1,1,1)), np.tile(pyramid_output[4],(nb_MC_samples,1,1,1)), learning_phase])

        MC_samples_classification = np.array(MC_samples_classification)
        MC_samples_regression = np.array(MC_samples_regression)
        # shape of MC_samples_classification_squeezed is (# MC sample x # anchors x # classes / 4 or 8 for regression laplace)
        MC_samples_classification_squeezed = np.squeeze(MC_samples_classification)
        
        MC_samples_regression_squeezed = np.squeeze(MC_samples_regression)
        MC_samples_laplacian_squeezed = MC_samples_regression_squeezed[:,:,4:]
        MC_samples_laplacian_squeezed = np.exp(-MC_samples_laplacian_squeezed)
        MC_samples_regression_squeezed = MC_samples_regression_squeezed[:,:,:4]
       
        """
        # ALPHA ENTROPY 
        num1 = np.power(1 - MC_samples_classification_squeezed, 1 - alpha)
        num2 = np.power(MC_samples_classification_squeezed, alpha)
        den = num1 + num2
   
        # a = np.mean(np.divide(num1,den), axis = 0)
        # b = np.mean(np.log(np.divide(num1, den)), axis=0)
        # c = np.mean(np.divide(num2,den), axis = 0)
        # d = np.mean(np.log(np.divide(num2, den)), axis = 0)
        
        return np.sum(- np.multiply(np.mean(np.divide(num1,den), axis = 0), np.mean((1 - alpha) * np.log(1 - MC_samples_classification_squeezed), axis=0)) - np.multiply(np.mean(np.divide(num2,den), axis = 0), np.mean(alpha * np.log(MC_samples_classification_squeezed), axis = 0)) + np.mean(np.log(den),axis=0))
        """
        """
        # Gamma + Alpha Variation Ratio 
        a = np.power(MC_samples_classification_squeezed, alpha * np.power(1 - MC_samples_classification_squeezed, gamma))
        b = np.power(1 - MC_samples_classification_squeezed, (1 - alpha) * np.power(MC_samples_classification_squeezed, gamma))
        # p_i = prob that class is 0  
        p_i = np.mean(np.divide(b, a + b), axis = 0)

        # mask = p_i > 0.5 
        # mask = mask.astype(int)
        # p = np.multiply(mask, p_i) + np.multiply(1 - mask, 1 - p_i)
       
        # # anchors x # classes  
        min_prob_class_per_anchor = np.amin(p_i, axis = 1)
        zero_class_product = np.sum(np.log(p_i), axis = 1)
        # the max for the classification section 
        max_onehot_prob = -np.log(min_prob_class_per_anchor ) + zero_class_product + np.log( 1 - min_prob_class_per_anchor)
        # assume acquisition is based on anchor being posibite (ie. variance vs constant) 
        reciprocals = -np.log(np.std(MC_samples_regression_squeezed, axis = 0) + 0.005)

        # l1 (median not variance) acquisition
        medians = np.sort(MC_samples_regression_squeezed, axis = 0)
        mediansfirst = np.sum(medians[: nb_MC_samples/2, :,:] , axis = 0)
        medianslast = np.sum(medians[nb_MC_samples/2 :, :,:], axis = 0)
        mediandiff = (medianslast - mediansfirst) / nb_MC_samples 
        # std_one_hot_prod = np.sum(reciprocals, axis = 1) + max_onehot_prob
        # maximum_prod = np.maximum(std_one_hot_prod, zero_class_product)
        # score = - np.sum(std_one_hot_prod)
        score = -np.sum(np.maximum(max_onehot_prob, zero_class_product) + np.sum(mediandiff, axis = 1))
        """
        
        # CASE 1, one hot 
        # from Gamma + Alpha bernoulli 
        # MAX ONE HOT CLASS PROBS 
        a = np.power(MC_samples_classification_squeezed, alpha * np.power(1 - MC_samples_classification_squeezed, gamma))
        b = np.power(1 - MC_samples_classification_squeezed, (1 - alpha) * np.power(MC_samples_classification_squeezed, gamma))  
        mean_zero_class_prob = np.mean(np.divide(b, a + b), axis = 0)
        zero_prob_per_anchor = np.sum(np.log(mean_zero_class_prob), axis = 1) 
        # dim: # anochors x # classes
        min_class_prob_per_anchor = np.amin(mean_zero_class_prob, axis = 1)
        # dim: # anchors
        max_one_hotprob_per_anchor = zero_prob_per_anchor - np.log(min_class_prob_per_anchor) + np.log(1 - min_class_prob_per_anchor)
        
        # MAX LAPLACIAN PROBS 
        medians = np.sort(MC_samples_regression_squeezed, axis = 0)
        mediansfirst = np.sum(medians[: nb_MC_samples / 2 , :,:], axis = 0) 
        medianslast = np.sum(medians[nb_MC_samples/ 2 :, :,:], axis = 0) 
        laplacians = MC_samples_laplacian_squeezed[0]
        laplacian_max = np.sum((medianslast - mediansfirst) / laplacians - np.log(2 * laplacians), axis = 1)
        
        # Max case 1 and case 2 (zero hot prob) 
        max_per_anchor = np.maximum(laplacian_max + max_one_hotprob_per_anchor, zero_prob_per_anchor)

        # Assume even number of MC samples 
        return -np.sum(max_per_anchor)
         
        # return 1 - np.prod(p)


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

def get_next_acquisition(
    generator,
    model,
    model_output_classification,
    model_output_regression,
    model_output_pyramid,
    model_pyramid_classification, 
    model_pyramid_regression,
    acquisition_size,
    pool_indices,  
    score_threshold = 0.05, 
    max_detections=100, 
    save_path=None
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

    # all of the pool images
    # image_group: # batches x [image dims]
    image_names = generator.image_names
    
    scores = np.ones((generator.size(),)) * -10000
   
    for i in pool_indices: 
        image = generator.load_image(i)
        image = np.expand_dims(image, axis = 0)
        
        scores[i] = acquisition_function(image, model, model_output_classification, model_output_regression,  model_output_pyramid, model_pyramid_classification, model_pyramid_regression, nb_MC_samples = 20)
    

    # gather all detections and annotations
    # all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    # average_precisions = {}

    # acquisitions = acquisition_function(all_detections, all_annotations, generator)

    # select pool indices with highest acquisition functions 
    # pool_indices = acquisition[0].argsort()[-batch_size:][::-1]

    # return names of pool file
    top_scores_indices = np.argpartition(scores, -acquisition_size)[-acquisition_size:]
    acquired_images = [image_names[i] for i in top_scores_indices]
    return top_scores_indices, acquired_images


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
