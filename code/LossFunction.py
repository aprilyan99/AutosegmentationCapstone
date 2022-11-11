import numpy as np
import tensorflow as tf

from itertools import product
from sorcery import unpack_keys
from model_callbacks import DecayAlphaParameter

#multi-class sorensen-dice coefficient metric
def dice_coef_metric(y_true, y_pred):
    #activate outputs
    if y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true[...,0], -1)
    y_true_onehot, y_pred_prob = activate_ouputs(y_true, y_pred)
    #can treat batch as a "pseudo-volume", or collect dice metrics on each volume in the batch individually
    axes_to_sum = find_axes_to_sum(y_true_onehot)
    #calculate dice metric per class
    intersection = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_prob), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true_onehot, axis=axes_to_sum), tf.reduce_sum(y_pred_prob, axis=axes_to_sum))
    numerator = tf.add(tf.multiply(intersection, 2.), LossParameters.smooth)
    denominator = tf.add(union, LossParameters.smooth)
    dice_metric_per_class = tf.divide(numerator, denominator)
    #return average dice metric over classes (choosing to use or not use the background class)
    return calculate_final_dice_metric(dice_metric_per_class)

def dice_coef_loss(y_true, y_pred):
    return tf.subtract(1., dice_coef_metric(y_true, y_pred))

def weighted_cross_entropy_loss(y_true, y_pred):
    if y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true[...,0], -1)
    y_true_onehot, y_pred_prob, cross_entropy_matrix = cross_entropy_loss_matrix(y_true, y_pred)
    return tf.multiply(weight_matrix(y_true_onehot, y_pred_prob), cross_entropy_matrix)

def joint_dice_cross_entropy_loss(y_true, y_pred):
    loss_contribution1 = tf.multiply(DecayAlphaParameter.alpha1, dice_coef_loss(y_true, y_pred))
    loss_contribution2 = tf.multiply(DecayAlphaParameter.alpha2, weighted_cross_entropy_loss(y_true, y_pred))
    return tf.add(loss_contribution1, loss_contribution2)

def one_hot_encode(y, num_classes):
    return tf.cast(tf.one_hot(tf.cast(y, tf.int32), num_classes), tf.float32)

def sigmoid_probability(y):
    return tf.keras.activations.sigmoid(y)

def softmax_probability(y):
    return tf.keras.activations.softmax(y)
