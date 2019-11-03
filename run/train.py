from __future__ import division
import os, time, scipy.io
import PIL.Image
import scipy.misc
import scipy.signal
import imageio
from preprocess.package_cfa import *
from preprocess.get_exif import *
from preprocess.psnr import *
from loss.loss import *
from network.model import  *
import random
import exifread
import threading
# 网络输出使用两层unet，层间增加iso倍率残差
# 输出降维到3层后 ,直接输出
# 对低感光度照片使用反向白平衡计算，使用照片EXIF中白平衡信息倒数处理

#增加双线程循环读取加载数据



# # input_dir = '../dataset/nex/short/'
# input_dir = '../dataset/texture/short/'
# # gt_dir = '../dataset/nex/long/'
# gt_dir = '../dataset/texture/long/'
# # gt_png_dir = '../dataset/nex/cbm3d5/'
# gt_png_dir = '../dataset/texture/cbm3d5/'


input_dir = 'E:/Git_work/vivo_denoise/dataset/texture/short/'
# gt_dir = '../dataset/nex/long/'
gt_dir = 'E:/Git_work/vivo_denoise/dataset/texture/long/'
# gt_png_dir = '../dataset/nex/cbm3d5/'
gt_png_dir = 'E:/Git_work/vivo_denoise/dataset/texture/cbm3d5/'


train_fns = glob.glob(input_dir + '*.dng')
train_ids = []


for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    # train_ids.append(train_fn.split('_')[-2])
    train_ids.append(train_fn.split('.')[0])

ps_list =[96]
train_size = 5
buffer_size = 30
# train_size = 3
save_freq = 50
log_freq = 5
scaling_factor = 0.1
bs = 4 # batch size for training

black = 64
white = 1023


value = 0.84

mark = "mix1"

# result_dir = './result/'+str(ps)+"_"+str(bs)+"_"+str(value)+mark+"/"
result_dir = '../resultset/'+""+mark+"/"
checkpoint_dir = result_dir




input_buffer_images = [None] * buffer_size
gt_buffer_images = [None] * buffer_size

def thread_reading():
    thread_tmp =0

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        # tick1 = time.time()
        if thread_tmp > (buffer_size - 1):
            break
        train_id_th = train_ids[ind]
        in_files = glob.glob(input_dir + '%s.*dng' % train_id_th)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        _, in_fn = os.path.split(in_path)
        fullpath = gt_png_dir + '%s.png' % train_id_th
        print(fullpath)
        gt_files_th = glob.glob(fullpath)
        gt_path = gt_files_th[0]
        _, gt_fn = os.path.split(gt_path)
        # if input_buffer_images[thread_tmp] is None:
        raw = rawpy.imread(in_path)

        print(str(thread_tmp))
        im = PIL.Image.open(gt_path)
        im = scipy.misc.fromimage(im)
        tmp_wb, tmp_be = getWB(in_path)
        im = upsidedown_wb(im, tmp_wb)
        gt_buffer_images[thread_tmp] = np.expand_dims(np.float32(im / 255.0), axis=0)
        input_buffer_images[thread_tmp] = np.expand_dims(pack_raw(raw, black, white, 0), axis=0) * tmp_be
        thread_tmp = thread_tmp + 1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

in_image = tf.placeholder(tf.float32, [None, None, None, 4] ,name='in_image')
gt_image = tf.placeholder(tf.float32, [None, None, None, 3] )



out_image = rvr_2(in_image)



G_loss = get_loss(gt_image,out_image)
PSNR = getPSNR(gt_image,out_image)

#
# # Scalar to keep track for loss
tf.summary.scalar("PSNR", PSNR)
tf.summary.scalar("loss", tf.reduce_mean(G_loss))

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)


G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss , var_list=t_vars)

saver = tf.train.Saver( t_vars)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

saver_full = tf.train.Saver()

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * train_size
input_images = [None] * train_size

allfolders = glob.glob(result_dir+'*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-5:]))


learning_rate = 1e-4

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(result_dir + "/train", sess.graph)


for epoch in tqdm(range(lastepoch, lastepoch+4001)):
# for epoch in range(lastepoch,2601):
    if os.path.isdir(result_dir+"%05d"%epoch):
        continue
    cnt = 0

    if epoch > 2000:
        learning_rate = 1e-4

    if epoch > 3000:
        learning_rate = 1e-5
    # if epoch > 2000:
    #     learning_rate = 1e-6

    if input_images[0] is None:
    # if epoch==0 or (epoch) % 20001 == 0:

        gt_images = [None] * train_size
        input_images = [None] * train_size
        # input_flat_images = [None] * train_size
        tmp = 0

        for ind in np.random.permutation(len(train_ids)):
            # get the path from image id
            tick1 = time.time()
            if tmp > (train_size-1):
                break

            train_id = train_ids[ind]
            # in_files = glob.glob(input_dir + '%s.dng' % train_id)
            in_files = glob.glob(input_dir + '%s.*dng' % train_id)
            in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
            _, in_fn = os.path.split(in_path)

            # gt_files = glob.glob(gt_dir + '%s_*.dng' % train_id)
            gt_files = glob.glob(gt_dir + '%s.dng' % train_id)
            gt_files = glob.glob(gt_png_dir + '%s.png' % train_id)
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)

            if input_images[tmp] is None:
                raw = rawpy.imread(in_path)
                # input_images[ind] = np.expand_dims(pack_raw_nex3(raw), axis=0)

                im = PIL.Image.open(gt_path)
                im = scipy.misc.fromimage(im)


                # gt_raw = rawpy.imread(gt_path)
                # im = gt_raw.raw_image.astype(np.float32)
                # im = convert_2d(im ,1024).astype(np.float32)
                # gt_raw.raw_image[:, :] = im.astype(np.uint16)
                # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                # im = np.rot90(im, 1)
                tmp_wb , tmp_be = getWB(in_path)
                im = upsidedown_wb(im , tmp_wb)
                gt_images[tmp] = np.expand_dims(np.float32(im / 255.0), axis=0)
                input_images[tmp] = np.expand_dims(pack_raw(raw, black, white), axis=0) *tmp_be
                # input_flat_images[tmp] = (pack_raw_step1(raw, black, white , 0)) *tmp_be
                # input_flat_images[tmp] = np.expand_dims(pack_raw_step1(raw, black, white , 0), axis=0) *tmp_be

            tmp = tmp +1
            tick2 = time.time()

            print('read:' + str(tick2 - tick1))
            print(tmp)

    #
    # if (epoch+1) % (save_freq*4) == 0:
    #
    #     for itmp in range(0, buffer_size):
    #         # input_images.pop(0)
    #         gt_images.pop(0)
    #         input_images.pop(0)
    #
    #     thread_reading()
    #
    #
    #     print("change")
    #     # input_images.extend(input_buffer_images)
    #     gt_images.extend(gt_buffer_images)
    #     input_images.extend(input_buffer_images)
    #
    #     gt_buffer_images = [None] * buffer_size
    #     input_buffer_images = [None] * buffer_size
    #     # read_thread = threading.Thread(target=thread_reading())
    #     # read_thread.start()

    for ind in range(0,train_size):

        be = 1
        st = time.time()
        cnt += 1


        input_image = input_images[ind]
        ps = ps_list[random.randint(0,0)]


        # crop
        H = input_image.shape[1]
        W = input_image.shape[2]


        input_batch = np.zeros(dtype=np.float32, shape=[bs, ps, ps, 4])
        gt_batch = np.zeros(dtype=np.float32, shape=[bs, ps * 2, ps * 2, 3])


        for batch in range(0,bs):

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)

            input_patch = input_image[:, yy:yy + ps, xx:xx + ps, :]
            gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

            # if np.random.randint(2, size=1)[0] == 1:  # random flip
            #     input_patch = np.flip(input_patch, axis=1)
            #     # input_gt_patch = np.flip(input_gt_patch, axis=1)
            #     gt_patch = np.flip(gt_patch, axis=1)
            # if np.random.randint(2, size=1)[0] == 1:
            #     input_patch = np.flip(input_patch, axis=0)
            #     # input_gt_patch = np.flip(input_gt_patch, axis=0)
            #     gt_patch = np.flip(gt_patch, axis=0)
            # if np.random.randint(2, size=1)[0] == 1:  # random transpose
            #     input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            #     # input_gt_patch = np.transpose(input_gt_patch, (0, 2, 1, 3))
            #     gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

            # input_batch[batch,...] = np.minimum(input_patch, 1.0)
            input_batch[batch,...] = input_patch
            gt_batch[batch,...] = gt_patch



        if epoch % save_freq == 0:

            _,G_current,G_psnr, summary , output = sess.run([G_opt,G_loss,PSNR ,merged, out_image],
                                            feed_dict={in_image: input_batch, gt_image: gt_batch
                                                       ,lr: learning_rate})


        elif cnt == 1 and epoch % log_freq == 0:




            _,G_current,G_psnr, summary = sess.run([G_opt,G_loss,PSNR ,merged],
                                            feed_dict={in_image: input_batch, gt_image: gt_batch
                                                       ,lr: learning_rate})
            tick2 = time.time()


            train_writer.add_summary(summary, epoch)

        else:

            _,G_current = sess.run([G_opt,G_loss],
                                            feed_dict={in_image: input_batch, gt_image: gt_batch
                                                       ,lr: learning_rate})



        if  epoch % save_freq == 0:

            output = np.minimum(np.maximum(output, 0), 1)
            # gtsharp = np.minimum(np.maximum(gtsharp, 0), 1)
            # outsharp = np.minimum(np.maximum(outsharp, 0), 1)
            # g_loss[ind] = G_current

            if not os.path.isdir(result_dir + '%05d' % epoch):
                os.makedirs(result_dir + '%05d' % epoch)


            # scipy.misc.toimage(gt_images[ind][0, :, :, :]*255, high=255, low=0, cmin=0, cmax=255).save(
            #     result_dir + '%05d/%s_00_gt.jpg' % (epoch, ind))
            #
            # scipy.misc.toimage(gt_buffer_images[ind][0, :, :, :]*255, high=255, low=0, cmin=0, cmax=255).save(
            #     result_dir + '%05d/%s_00_gtbuff.jpg' % (epoch, ind))

            temp = np.concatenate((gt_batch[0, :, :, :], output[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%05d/%s_00_train.jpg' % (epoch, ind))
            # temp2 = np.concatenate((gtsharp[0, :, :, :], outsharp[0, :, :, :]), axis=1)
            # temp2 = temp2 * (2 ** 8 - 1)
            # temp2 = temp2.astype(np.uint8)
            # imageio.imwrite(result_dir + '%05d/%s_02_train.jpg' % (epoch, ind), temp2)

    #
    # if epoch > 10 and  (G_current[0]>1.8 or math.isnan(G_current[0])) :
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt:
    #         print('loaded ' + ckpt.model_checkpoint_path)
    #         saver_full.restore(sess, ckpt.model_checkpoint_path)
    #         learning_rate = learning_rate/10
    #         print('learning_rate: %.6f' %(learning_rate))
    # else:
    #     if epoch % save_freq == 0:
    #         saver_full.save(sess, checkpoint_dir + 'model.ckpt')

    if epoch % save_freq == 0:
         saver.save(sess, checkpoint_dir + 'model.ckpt')