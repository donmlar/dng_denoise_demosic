# 配置基础目录

base_data_path = 'E:/Git_work/vivo_denoise'

def get_data(mark):


    if mark == 'texture':
        input_dir = base_data_path+'/dataset/texture/short/'
        gt_png_dir = base_data_path+'/dataset/texture/cbm3d5/'

        test_input_dir = base_data_path+'/dataset/texture/short/'
        test_gt_png_dir = base_data_path+'/dataset/texture/cbm3d5/'


        return input_dir,gt_png_dir,test_input_dir,test_gt_png_dir

    elif mark == 'full':
        input_dir = base_data_path+'/dataset/texture/short/'
        gt_png_dir = base_data_path+'/dataset/nex/cbm3d5/'

        test_input_dir = base_data_path+'/dataset/texture/short/'
        test_gt_png_dir = base_data_path+'/dataset/texture/cbm3d5/'

        return input_dir,gt_png_dir,test_input_dir,test_gt_png_dir

def black_white():
    black = 64
    white = 1023
    return black,white

def train_param():

    train_size = 5
    buffer_size = 30
    save_freq = 50
    log_freq = 5
    bs = 4

    return train_size,buffer_size,bs,save_freq,log_freq


def patch_size():

    ps= 96
    return ps

