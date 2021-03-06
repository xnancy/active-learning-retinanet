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

import keras
from . import backend


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        labels         = y_true
        classification = y_pred

        # filter out "ignore" anchors
        anchor_state   = keras.backend.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)
        
        focal_normalizer = keras.backend.log(keras.backend.pow(classification , (alpha * keras.backend.pow((1 - classification) , gamma))) + keras.backend.pow(1 - classification , ((1 - alpha) * keras.backend.pow(classification, gamma))))

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(1.0, normalizer)
        normalizer0 = backend.where(keras.backend.equal(anchor_state, 0))
        normalizer0 = keras.backend.cast(keras.backend.shape(normalizer0)[0], keras.backend.floatx()) 
        normalizer0 = keras.backend.maximum(1.0, normalizer0) 
        return keras.backend.sum(cls_loss ) 

    return _focal

def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2


    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression        = y_pred[:,:,:4]
        laplacian         = y_pred[:,:,4:]
        regression_target = y_true[:, :, :4]
        anchor_state      = y_true[:, :, 4]
        
        # 4 x 1 
        laplacian_weights = keras.backend.constant([100,100,100,100])

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        laplacian 	  = backend.gather_nd(laplacian, indices) 
        regression_target = backend.gather_nd(regression_target, indices)
        laplacian         = keras.backend.exp(-laplacian)
        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer0 = backend.where(keras.backend.equal(anchor_state, 0))
        normalizer0 = keras.backend.cast(keras.backend.shape(normalizer0)[0], keras.backend.floatx())
        normalizer0 = keras.backend.maximum(1.0, normalizer0)
        normalizer1 = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer1 = keras.backend.cast(keras.backend.shape(normalizer1)[0], keras.backend.floatx())
        normalizer1 = keras.backend.maximum(1.0, normalizer1)

        regression_loss_laplacian = keras.layers.multiply([regression_loss, 1 / laplacian]) 
        return (keras.backend.sum(regression_loss_laplacian) + keras.backend.sum(keras.backend.log(2 * laplacian_weights))) 

    return _smooth_l1

