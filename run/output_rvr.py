# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import math
import rawpy
import glob
import exifread
from network.model import  *
import imageio
# input_dir = '../dng/'
# input_dir = 'G:/dng/ps/800/'
input_dir = 'E:/Git_work/dng_Analysis/high/raw/'
# input_dir = 'E:/Texture/0622204527/'
# input_dir = '../dataset/nex/short/'
# input_dir = 'E:/Git_work/vivo_denoise/bm3d/'
# input_dir = 'E:/Git_work/dng_Analysis/high/raw/'
# input_dir = 'E:/Git_work/decode/222/'

# input_dir = '../dataset/samsung/long/'
# gt_dir = '../dataset/nex/long/'
# checkpoint_dir = './result/mix_1_0.84u3net/'
checkpoint_dir = '../resultset/rvr/'





def pack_raw(raw , black , white):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    # im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im1 = convert_2d(im ,white).astype(np.float32)

    im = np.maximum(im1 -black , 0) / (white)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape

    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:],
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    # out1 = convert_3d(out)

    return out


def pack_raw_nex3(raw , black , white):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    # im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    im = np.maximum(im - black, 0) / (white)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape

    H = img_shape[0]-2
    W = img_shape[1]-2

    imt = np.zeros((H, W, 1), dtype=np.float32)

    imt = im[1:img_shape[0]-1 , 1:img_shape[1]-1 , :]

    # H = img_shape[0]
    # W = img_shape[1]-4
    #
    # imt = np.zeros((H, W, 1), dtype=np.float32)
    # imt = im[0:img_shape[0] , 1:img_shape[1]-3 , :]

    out = np.concatenate((imt[0:H:2,0:W:2,:],#绿1
                          imt[0:H:2,1:W:2,:],#红
                          imt[1:H:2,1:W:2,:],#绿2
                          imt[1:H:2,0:W:2,:]), axis=2)#蓝
    return out

def read_iso(path_name):
    f = open(path_name, 'rb')

    # Return Exif tags
    tags = exifread.process_file(f)


    iso = tags.get('Image ISOSpeedRatings')

    if iso is None :
        iso = tags.get('EXIF ISOSpeedRatings')
    # print(tags.get('Image ISOSpeedRatings'))

    f.close()
    return int(str(iso))

def toFloat(ratio):

    if ratio.num == 0 :
        return 0
    else:
        return ratio.den/ratio.num


def toFloat222(ratio):

    if ratio.den == 0 :
        return 0
    else:
        return ratio.num/ratio.den

def getWB(path):
    # Open image file for reading (binary mode)
    f = open(path, 'rb' )

    # Return Exif tags
    tags = exifread.process_file(f)
    wb  = tags.get('Image Tag 0xC628')


    nz3 = np.zeros((3,3) ,dtype=np.float32)
    nz3[0 , 0] = toFloat(wb.values[0])/toFloat(wb.values[1])
    nz3[1 , 1] = 1
    nz3[2 , 2] = toFloat(wb.values[2])/toFloat(wb.values[1])

    be  = tags.get('Image Tag 0xC62A')
    berate = 2 ** (toFloat222(be.values[0])+0)

    return nz3 , berate

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
iso_rate_ph = tf.placeholder(tf.float32,[] )
baseline_ph = tf.placeholder(tf.float32,[] )
wb_ph = tf.placeholder(tf.float32,[3,3] )
size_ph = tf.placeholder(tf.int32,[4] )

out_image = rvr_2(in_image)

# a = tf.constant([0, -1, 0, -1, 4, -1, 0, -1, 0], shape=[3, 3] ,dtype=tf.int8)
# # a = tf.constant([-1, -1, -1, -1, 8, -1, -1, -1, -1], shape=[3, 3] ,dtype=tf.int8)
# a1 = tf.cast(a , tf.float32)
# a2 = tf.reshape(a1, [3, 3, 1,1])
# out_image =tf.image.rgb_to_grayscale(out_image)
# out_image = tf.nn.conv2d(out_image , a2,strides=[1, 1,1, 1] ,padding='SAME')
# out_image = tf.scalar_mul(8, out_image)


saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)



# test the first image in each sequence
# in_files = glob.glob(input_dir + '*.ARW')
# in_files = glob.glob(input_dir + '*.NEF')
train_fns = glob.glob(input_dir + '*.dng')
train_ids = []
for i in range(len(train_fns)):
# for i in range(10):
    _, train_fn = os.path.split(train_fns[i])
    # train_ids.append(train_fn.split('_'))
    train_ids.append(train_fn)




for k in range(len(train_ids)):

    for j in range(1):
        train_id = train_ids[k]
        in_files = glob.glob(input_dir + train_id)

        in_files = in_files[0]
        tmp_wb, tmp_be = getWB(in_files)
        #
        # long_path = gt_dir+ in_fn
        #
        raw = rawpy.imread(in_files)
        # input_full = np.expand_dims(pack_raw_nex3(raw), axis=0)
        # input_full = np.expand_dims(pack_raw(raw ,143 , 4096), axis=0)*tmp_be
        input_full = np.expand_dims(pack_raw(raw ,64 , 1024), axis=0)*(tmp_be)
        # input_full = np.expand_dims(pack_raw(raw ,1024 , 16368), axis=0)*(tmp_be)
        # input_full = np.expand_dims(pack_raw_nex3(raw, 0, 15892), axis=0) * (max([tmp_be,1])+1)

        raw = rawpy.imread(in_files)

        # im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=16)
        # # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        # scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # input_full = np.minimum(input_full, 1.0)

        in_exposure =  read_iso(in_files)

        # ratio = (1 - ((in_exposure - 400) / 2800)) / 4

        # in_exposure = max([in_exposure,800])
        # in_exposure = 800

        tick1 = time.time()

        # ratio = math.log(in_exposure / 100, 2)
        # ratio = (-0.171 * math.log(in_exposure / 50) + 0.815) / 3


        ssim_rate = j *0.3

        ratio =  4.1002*(ssim_rate**4) - 11.791*(ssim_rate**3) + 12.632*(ssim_rate**2) - 6.4746*(ssim_rate**1) + 1.535

        input_x = input_full.shape[1]
        input_y = input_full.shape[2]



        output = sess.run(out_image, feed_dict={in_image: input_full } )
        output = np.minimum(np.maximum(output, 0), 1)

        output = output[0, :, :, :]
        # scale_full = scale_full[0, :, :, :]





        b = output[:, :, np.newaxis, :]

        b = np.matmul(b, tmp_wb)

        c = np.squeeze(b)

        # c = c * (2**16 - 1)

        # c = c.astype(np.uint16)

        # imageio.imwrite(in_files+'_out.tiff', c)

        # output = output * (2**8 - 1)
        # output = output.astype(np.uint8)
        # imageio.imwrite(in_files+'_1out.png', c )

        scipy.misc.toimage(c * 255, high=255, low=0, cmin=0, cmax=255).save(
            in_files+'_'+str(j)+'_rvr22.png')

        # scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
        #     in_files+'_short.png')

        tick2 = time.time()

        print(tick2 - tick1)


