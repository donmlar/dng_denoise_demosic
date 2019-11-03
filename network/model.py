
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import pdb
import rawpy
import glob
import math
import preprocess.msssim
from tqdm import tqdm


scaling_factor = 0.1


def resBlock(x,channels=64,kernel_size=[3,3],scale=1 ):
    tmp = slim.conv2d(x, channels, kernel_size, activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, channels, kernel_size, activation_fn=None)
    tmp *= scale
    x *= (1-scale)
    return x + tmp


def resBlock_with_name(x,channels=64,kernel_size=[3,3],scale=1 ,name = 'res' ):
    tmp = slim.conv2d(x, channels, kernel_size, activation_fn=None,scope=name+ '_conv0')
    tmp = tf.nn.relu(tmp ,name=name + '_relu')
    tmp = slim.conv2d(tmp, channels, kernel_size, activation_fn=None,scope=name+ '_conv1')
    tmp *= scale
    x *= (1-scale)
    return x + tmp


def lrelu(x):
    # return None
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels , name):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02, name= name+'_v1'), name= name+'_v3')
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] ,name= name+'_v2' )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output




def rvr(input ):

    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=None,scope='g_conv0')

    for i in range(16):
        conv1 = resBlock(conv1, 32, scale=0.1)

    conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=None,scope='g_conv0_0_0')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=None,scope='g_conv0_1_0')


    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

    conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=None,scope='g_conv0_2_0')

    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=None,scope='g_conv0_3_0')

    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=None,scope='g_conv0_4_0')

    up6 =  upsample_and_concat( conv5, conv4, 256, 512 ,name='g_uc_0_0' )
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=None,scope='g_conv0_5_0')


    up7 =  upsample_and_concat( conv6, conv3, 128, 256  ,name='g_uc_0_1' )
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=None,scope='g_conv0_6_0')

    up8 =  upsample_and_concat( conv7, conv2, 64, 128   ,name='g_uc_0_2')
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=None,scope='g_conv0_7_0')

    up9 =  upsample_and_concat( conv8, conv1, 32, 64  ,name='g_uc_0_3')
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=None,scope='g_conv0_8_0')

    conv1_1=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=None,scope='g_conv1_0_0')
    pool1_1=slim.max_pool2d(conv1_1, [2, 2], padding='SAME' )


    conv2_1=slim.conv2d(pool1_1,64,[3,3], rate=1, activation_fn=None,scope='g_conv1_1_0')


    pool2_1=slim.max_pool2d(conv2_1, [2, 2], padding='SAME' )

    conv3_1=slim.conv2d(pool2_1,128,[3,3], rate=1, activation_fn=None,scope='g_conv1_2_0')

    pool3_1=slim.max_pool2d(conv3_1, [2, 2], padding='SAME' )

    conv4_1=slim.conv2d(pool3_1,256,[3,3], rate=1, activation_fn=None,scope='g_conv1_3_0')

    pool4_1=slim.max_pool2d(conv4_1, [2, 2], padding='SAME' )

    conv5_1=slim.conv2d(pool4_1,512,[3,3], rate=1, activation_fn=None,scope='g_conv1_4_0')

    up6_1 =  upsample_and_concat( conv5_1, conv4_1, 256, 512  ,name='g_uc_1_0' )
    conv6_1=slim.conv2d(up6_1,  256,[3,3], rate=1, activation_fn=None,scope='g_conv1_5_0')



    up7_1 =  upsample_and_concat( conv6_1, conv3_1, 128, 256  ,name='g_uc_1_1' )
    conv7_1=slim.conv2d(up7_1,  128,[3,3], rate=1, activation_fn=None,scope='g_conv1_6_0')

    up8_1 =  upsample_and_concat( conv7_1, conv2_1, 64, 128  ,name='g_uc_1_2')
    conv8_1=slim.conv2d(up8_1,  64,[3,3], rate=1, activation_fn=None,scope='g_conv1_7_0')

    up9_1 =  upsample_and_concat( conv8_1, conv1_1, 32, 64  ,name='g_uc_1_3')
    conv9_1=slim.conv2d(up9_1,  32,[3,3], rate=1, activation_fn=None,scope='g_conv1_8_0')




    for i in range(16):
        conv9_3 = resBlock(conv9_1, 32, scale=0.1)


    conv9_4=slim.conv2d(conv9_3,  32,[3,3], rate=1, activation_fn=None,scope='g_conv1_9_0')

    conv10=slim.conv2d(conv9_4,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')

    out = tf.depth_to_space(conv10,2,name='output')


    return out



# 修正版本，之前的后面res模块没用
def rvr_2(input ):

    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=None,scope='g_conv0')

    for i in range(16):
        conv1 = resBlock(conv1, 32, scale=0.1)

    conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=None,scope='g_conv0_0_0')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=None,scope='g_conv0_1_0')


    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

    conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=None,scope='g_conv0_2_0')

    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=None,scope='g_conv0_3_0')

    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=None,scope='g_conv0_4_0')

    up6 =  upsample_and_concat( conv5, conv4, 256, 512 ,name='g_uc_0_0' )
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=None,scope='g_conv0_5_0')


    up7 =  upsample_and_concat( conv6, conv3, 128, 256  ,name='g_uc_0_1' )
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=None,scope='g_conv0_6_0')

    up8 =  upsample_and_concat( conv7, conv2, 64, 128   ,name='g_uc_0_2')
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=None,scope='g_conv0_7_0')

    up9 =  upsample_and_concat( conv8, conv1, 32, 64  ,name='g_uc_0_3')
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=None,scope='g_conv0_8_0')

    conv1_1=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=None,scope='g_conv1_0_0')
    pool1_1=slim.max_pool2d(conv1_1, [2, 2], padding='SAME' )


    conv2_1=slim.conv2d(pool1_1,64,[3,3], rate=1, activation_fn=None,scope='g_conv1_1_0')


    pool2_1=slim.max_pool2d(conv2_1, [2, 2], padding='SAME' )

    conv3_1=slim.conv2d(pool2_1,128,[3,3], rate=1, activation_fn=None,scope='g_conv1_2_0')

    pool3_1=slim.max_pool2d(conv3_1, [2, 2], padding='SAME' )

    conv4_1=slim.conv2d(pool3_1,256,[3,3], rate=1, activation_fn=None,scope='g_conv1_3_0')

    pool4_1=slim.max_pool2d(conv4_1, [2, 2], padding='SAME' )

    conv5_1=slim.conv2d(pool4_1,512,[3,3], rate=1, activation_fn=None,scope='g_conv1_4_0')

    up6_1 =  upsample_and_concat( conv5_1, conv4_1, 256, 512  ,name='g_uc_1_0' )
    conv6_1=slim.conv2d(up6_1,  256,[3,3], rate=1, activation_fn=None,scope='g_conv1_5_0')



    up7_1 =  upsample_and_concat( conv6_1, conv3_1, 128, 256  ,name='g_uc_1_1' )
    conv7_1=slim.conv2d(up7_1,  128,[3,3], rate=1, activation_fn=None,scope='g_conv1_6_0')

    up8_1 =  upsample_and_concat( conv7_1, conv2_1, 64, 128  ,name='g_uc_1_2')
    conv8_1=slim.conv2d(up8_1,  64,[3,3], rate=1, activation_fn=None,scope='g_conv1_7_0')

    up9_1 =  upsample_and_concat( conv8_1, conv1_1, 32, 64  ,name='g_uc_1_3')
    conv9_1=slim.conv2d(up9_1,  32,[3,3], rate=1, activation_fn=None,scope='g_conv1_8_0')




    for i in range(16):
        conv9_1 = resBlock(conv9_1, 32, scale=0.1)


    conv9_4=slim.conv2d(conv9_1,  32,[3,3], rate=1, activation_fn=None,scope='g_conv1_9_0')

    conv10=slim.conv2d(conv9_4,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')

    out = tf.depth_to_space(conv10,2,name='output')


    return out

def rrv(input ):

    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=None,scope='g_conv0')

    for i in range(32):
        conv1 = resBlock(conv1, 32, scale=0.1)

    conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=None,scope='g_conv0_0_0')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=None,scope='g_conv0_1_0')


    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

    conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=None,scope='g_conv0_2_0')

    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=None,scope='g_conv0_3_0')

    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=None,scope='g_conv0_4_0')

    up6 =  upsample_and_concat( conv5, conv4, 256, 512 ,name='g_uc_0_0' )
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=None,scope='g_conv0_5_0')


    up7 =  upsample_and_concat( conv6, conv3, 128, 256  ,name='g_uc_0_1' )
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=None,scope='g_conv0_6_0')

    up8 =  upsample_and_concat( conv7, conv2, 64, 128   ,name='g_uc_0_2')
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=None,scope='g_conv0_7_0')

    up9 =  upsample_and_concat( conv8, conv1, 32, 64  ,name='g_uc_0_3')
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=None,scope='g_conv0_8_0')

    conv1_1=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=None,scope='g_conv1_0_0')
    pool1_1=slim.max_pool2d(conv1_1, [2, 2], padding='SAME' )


    conv2_1=slim.conv2d(pool1_1,64,[3,3], rate=1, activation_fn=None,scope='g_conv1_1_0')


    pool2_1=slim.max_pool2d(conv2_1, [2, 2], padding='SAME' )

    conv3_1=slim.conv2d(pool2_1,128,[3,3], rate=1, activation_fn=None,scope='g_conv1_2_0')

    pool3_1=slim.max_pool2d(conv3_1, [2, 2], padding='SAME' )

    conv4_1=slim.conv2d(pool3_1,256,[3,3], rate=1, activation_fn=None,scope='g_conv1_3_0')

    pool4_1=slim.max_pool2d(conv4_1, [2, 2], padding='SAME' )

    conv5_1=slim.conv2d(pool4_1,512,[3,3], rate=1, activation_fn=None,scope='g_conv1_4_0')

    up6_1 =  upsample_and_concat( conv5_1, conv4_1, 256, 512  ,name='g_uc_1_0' )
    conv6_1=slim.conv2d(up6_1,  256,[3,3], rate=1, activation_fn=None,scope='g_conv1_5_0')



    up7_1 =  upsample_and_concat( conv6_1, conv3_1, 128, 256  ,name='g_uc_1_1' )
    conv7_1=slim.conv2d(up7_1,  128,[3,3], rate=1, activation_fn=None,scope='g_conv1_6_0')

    up8_1 =  upsample_and_concat( conv7_1, conv2_1, 64, 128  ,name='g_uc_1_2')
    conv8_1=slim.conv2d(up8_1,  64,[3,3], rate=1, activation_fn=None,scope='g_conv1_7_0')

    up9_1 =  upsample_and_concat( conv8_1, conv1_1, 32, 64  ,name='g_uc_1_3')
    conv9_1=slim.conv2d(up9_1,  32,[3,3], rate=1, activation_fn=None,scope='g_conv1_8_0')



    conv9_4=slim.conv2d(conv9_1,  32,[3,3], rate=1, activation_fn=None,scope='g_conv1_9_0')

    conv10=slim.conv2d(conv9_4,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')

    out = tf.depth_to_space(conv10,2,name='output')


    return out


def vrr(input ):

    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=None,scope='g_conv0')


    conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=None,scope='g_conv0_0_0')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=None,scope='g_conv0_1_0')


    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

    conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=None,scope='g_conv0_2_0')

    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=None,scope='g_conv0_3_0')

    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=None,scope='g_conv0_4_0')

    up6 =  upsample_and_concat( conv5, conv4, 256, 512 ,name='g_uc_0_0' )
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=None,scope='g_conv0_5_0')


    up7 =  upsample_and_concat( conv6, conv3, 128, 256  ,name='g_uc_0_1' )
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=None,scope='g_conv0_6_0')

    up8 =  upsample_and_concat( conv7, conv2, 64, 128   ,name='g_uc_0_2')
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=None,scope='g_conv0_7_0')

    up9 =  upsample_and_concat( conv8, conv1, 32, 64  ,name='g_uc_0_3')
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=None,scope='g_conv0_8_0')

    conv1_1=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=None,scope='g_conv1_0_0')
    pool1_1=slim.max_pool2d(conv1_1, [2, 2], padding='SAME' )


    conv2_1=slim.conv2d(pool1_1,64,[3,3], rate=1, activation_fn=None,scope='g_conv1_1_0')


    pool2_1=slim.max_pool2d(conv2_1, [2, 2], padding='SAME' )

    conv3_1=slim.conv2d(pool2_1,128,[3,3], rate=1, activation_fn=None,scope='g_conv1_2_0')

    pool3_1=slim.max_pool2d(conv3_1, [2, 2], padding='SAME' )

    conv4_1=slim.conv2d(pool3_1,256,[3,3], rate=1, activation_fn=None,scope='g_conv1_3_0')

    pool4_1=slim.max_pool2d(conv4_1, [2, 2], padding='SAME' )

    conv5_1=slim.conv2d(pool4_1,512,[3,3], rate=1, activation_fn=None,scope='g_conv1_4_0')

    up6_1 =  upsample_and_concat( conv5_1, conv4_1, 256, 512  ,name='g_uc_1_0' )
    conv6_1=slim.conv2d(up6_1,  256,[3,3], rate=1, activation_fn=None,scope='g_conv1_5_0')



    up7_1 =  upsample_and_concat( conv6_1, conv3_1, 128, 256  ,name='g_uc_1_1' )
    conv7_1=slim.conv2d(up7_1,  128,[3,3], rate=1, activation_fn=None,scope='g_conv1_6_0')

    up8_1 =  upsample_and_concat( conv7_1, conv2_1, 64, 128  ,name='g_uc_1_2')
    conv8_1=slim.conv2d(up8_1,  64,[3,3], rate=1, activation_fn=None,scope='g_conv1_7_0')

    up9_1 =  upsample_and_concat( conv8_1, conv1_1, 32, 64  ,name='g_uc_1_3')
    conv9_1=slim.conv2d(up9_1,  32,[3,3], rate=1, activation_fn=None,scope='g_conv1_8_0')


    for i in range(32):
        conv9_1 = resBlock(conv9_1, 32, scale=0.1)

    conv9_4=slim.conv2d(conv9_1,  32,[3,3], rate=1, activation_fn=None,scope='g_conv1_9_0')

    conv10=slim.conv2d(conv9_4,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')

    out = tf.depth_to_space(conv10,2,name='output')


    return out


def rrr(input ):

    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=None,scope='g_conv0')

    for i in range(42):
        conv1 = resBlock(conv1, 32, scale=0.1)

    conv9_4=slim.conv2d(conv1,  32,[3,3], rate=1, activation_fn=None,scope='g_conv1_9_0')

    conv10=slim.conv2d(conv9_4,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')

    out = tf.depth_to_space(conv10,2,name='output')


    return out

def tosharp_lapu(img):
    # a = tf.constant([0, -1, 0, -1, 4, -1, 0, -1, 0], shape=[3, 3], dtype=tf.int8)
    a = tf.constant([-1, -1, -1, -1, 8, -1, -1, -1, -1], shape=[3, 3], dtype=tf.int8)
    a1 = tf.cast(a, tf.float32)
    a2 = tf.reshape(a1, [3, 3, 1, 1])
    img = tf.image.rgb_to_grayscale(img)
    img = tf.nn.conv2d(img , a2,strides=[1, 1,1, 1] ,padding='SAME')
    # img = tf.scalar_mul(8, img)
    return img


def tosharp_sobel(img):
    # a = tf.constant([0, -1, 0, -1, 4, -1, 0, -1, 0], shape=[3, 3], dtype=tf.int8)
    a = tf.constant([-1, 0, 1, -2, 0, 2, -1, 0, 1], shape=[3, 3], dtype=tf.int8)
    a = tf.cast(a, tf.float32)
    a = tf.reshape(a, [3, 3, 1, 1])

    b = tf.constant([-1, -2, -1, 0, 0, 0, 1, 2, 1], shape=[3, 3], dtype=tf.int8)
    b = tf.cast(b, tf.float32)
    b = tf.reshape(b, [3, 3, 1, 1])


    img = tf.image.rgb_to_grayscale(img)
    img_x = tf.nn.conv2d(img , a,strides=[1, 1,1, 1] ,padding='SAME')
    img_y = tf.nn.conv2d(img , b,strides=[1, 1,1, 1] ,padding='SAME')
    img = tf.abs(img_x)+tf.abs(img_y)

    # img = tf.math.subtract(img ,0.15  )
    # img = tf.nn.relu(img)
    # img = tf.math.add(img, 0.15)
    return img


def tosharp_sobel030(img):
    # a = tf.constant([0, -1, 0, -1, 4, -1, 0, -1, 0], shape=[3, 3], dtype=tf.int8)
    a = tf.constant([-1, 0, 1, -2, 0, 2, -1, 0, 1], shape=[3, 3], dtype=tf.int8)
    a = tf.cast(a, tf.float32)
    a = tf.reshape(a, [3, 3, 1, 1])

    b = tf.constant([-1, -2, -1, 0, 0, 0, 1, 2, 1], shape=[3, 3], dtype=tf.int8)
    b = tf.cast(b, tf.float32)
    b = tf.reshape(b, [3, 3, 1, 1])


    img = tf.image.rgb_to_grayscale(img)
    img_x = tf.nn.conv2d(img , a,strides=[1, 1,1, 1] ,padding='SAME')
    img_y = tf.nn.conv2d(img , b,strides=[1, 1,1, 1] ,padding='SAME')
    img = tf.abs(img_x)+tf.abs(img_y)

    img = tf.math.subtract(img ,0.3  )
    img = tf.nn.relu(img)
    img = tf.math.add(img, 0.3)
    return img

def tolight4x(img):
    # a = tf.constant([0, -1, 0, -1, 4, -1, 0, -1, 0], shape=[3, 3], dtype=tf.int8)

    img4x = tf.scalar_mul(4,img)

    img4x = tf.minimum(img4x, 1)

    return img4x


def tolight2x(img):
    # a = tf.constant([0, -1, 0, -1, 4, -1, 0, -1, 0], shape=[3, 3], dtype=tf.int8)

    img2x = tf.scalar_mul(2,img)

    img2x = tf.minimum(img2x, 1)

    return img2x

def tolight3x(img):
    # a = tf.constant([0, -1, 0, -1, 4, -1, 0, -1, 0], shape=[3, 3], dtype=tf.int8)

    img2x = tf.scalar_mul(3,img)

    img2x = tf.minimum(img2x, 1)

    return img2x