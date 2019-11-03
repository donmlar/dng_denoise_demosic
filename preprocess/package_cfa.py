# 根据不同的cfa排列，对cfa图像按照像素颜色进行分割
# bayer排列中，如果顺序不同，对图像边缘进行切割，使切割后的图像排列和顺序与训练顺序相同
# 如果训练和推理图像排列不同，会出现色彩和亮度异常


from preprocess.dilated_convolution import *

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