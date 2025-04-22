"""This script defines all the loss functions required to train the CycleGAN."""

import tensorflow as tf


def real_mse_loss(D_out, adv_weight):
    mse_loss = tf.reduce_mean((D_out-1)**2)*adv_weight
    return mse_loss


def fake_mse_loss(D_out, adv_weight=1):
    mse_loss = tf.reduce_mean((D_out)**2)*adv_weight
    return mse_loss


def cycle_loss(real_signal, reconstructed_signal, lambda_weight=1):
    reconstr_loss = tf.reduce_mean(tf.abs(real_signal-reconstructed_signal))
    return lambda_weight*reconstr_loss


def id_loss(real_signal, generated_signal, id_weight=1):
    gen_loss = tf.reduce_mean(tf.abs(real_signal-generated_signal))
    return id_weight*gen_loss