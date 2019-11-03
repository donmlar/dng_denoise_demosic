# 定义损失函数

from network.model import  *


def get_loss(gt_image,out_image):


    out_image_4x = tolight4x(out_image)
    gt_image_4x = tolight4x(gt_image)

    out_image_2x = tolight2x(out_image)
    gt_image_2x = tolight2x(gt_image)

    out_image_sharp = tosharp_sobel(out_image_2x)
    gt_image_sharp = tosharp_sobel(gt_image_2x)

    out_image_sharp_4x = tosharp_sobel(out_image_4x)
    gt_image_sharp_4x = tosharp_sobel(gt_image_4x)

    loss1 = tf.reduce_mean(tf.losses.absolute_difference(gt_image_2x, out_image_2x))
    loss3 = 1-tf.image.ssim_multiscale(gt_image_2x, out_image_2x ,max_val=1)

    loss4 = 1-tf.image.ssim_multiscale(gt_image_sharp, out_image_sharp ,max_val=1)

    loss5 = tf.reduce_mean(tf.losses.absolute_difference(gt_image_sharp, out_image_sharp))

    loss6 = 1-tf.image.ssim_multiscale(gt_image_4x, out_image_4x ,max_val=1)

    loss7 = 1-tf.image.ssim_multiscale(gt_image_sharp_4x, out_image_sharp_4x ,max_val=1)

    G_loss = 0.16* loss1 +0.64*loss3+0.2*loss6

    return G_loss

