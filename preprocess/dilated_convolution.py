# 空洞卷积
# 过滤图像中明显的亮点（热噪声）

import numpy as np
import scipy.signal
# 去除热燥（亮点）
def convert_2d(r,white):
    # 滤波掩模
    # bayer阵列计算热燥位置
    window1 = np.array([
        [-1, 0,-1,0, -1],
        [0, 0, 0, 0, 0],
        [-1, 0, 8, 0,-1],
        [0, 0, 0, 0, 0],
        [-1, 0, -1, 0, -1],
    ])
    # 计算热燥替换值
    window2 = np.array([
        [1/8, 0,1/8,0, 1/8],
        [0, 0, 0, 0, 0],
        [1/8, 0, 0, 0,1/8],
        [0, 0, 0, 0, 0],
        [1/8, 0, 1/8, 0, 1/8],
    ])
    # 排除高光边缘区域
    window3 = np.array([
        [0, -1, 0],
        [-1, 1, -1],
        [0, -1,0]
    ])
    s = scipy.signal.convolve2d(r, window1, mode='same', boundary='symm')
    g = scipy.signal.convolve2d(r, window2, mode='same', boundary='symm')
    m = scipy.signal.convolve2d(r, window3, mode='same', boundary='symm')
    # 像素值如果大于 255 则取 255, 小于 0 则取 0


    #
    # # for循环速度较慢，使用frompyfunc加速
    # for i in range(s.shape[0]):
    #     for j in range(s.shape[1]):
    #         if  s[i][j] >white/5 and m[i][j] >-(white/10):
    #             r[i][j] = g[i][j]
    #             s[i][j] = 255
    #
    #         else:
    #             r[i][j] = r[i][j]
    #             s[i][j] = 0

    # for循环速度较慢，使用frompyfunc加速
    def subf(s_sub,g_sub,m_sub,r_sub):
        if  s_sub >white/10 and m_sub >-(white/5):
            return g_sub
            s
        else:
            return r_sub
    subf_ufunc = np.frompyfunc(subf, 4, 1)
    r = subf_ufunc(s, g, m ,r)



    return r