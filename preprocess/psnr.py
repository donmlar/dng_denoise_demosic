
import tensorflow as tf



def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def getPSNR(gt_image,out_image):
    mse = tf.reduce_mean(tf.squared_difference(gt_image, out_image))
    PSNR = tf.constant(1 ** 2, dtype=tf.float32) / mse
    PSNR = tf.constant(10, dtype=tf.float32) * log10(PSNR)
    return PSNR