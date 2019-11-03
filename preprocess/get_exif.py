# 读取dug文件的信息



import exifread
import numpy as np



def toFloatUpDown(ratio):
    return ratio.num/ratio.den




def getWB(path):
    # Open image file for reading (binary mode)
    f = open(path, 'rb' )

    # Return Exif tags
    tags = exifread.process_file(f)
    wb  = tags.get('Image Tag 0xC628')


    nz3 = np.zeros((3,3) ,dtype=np.float32)
    nz3[0 , 0] = toFloatUpDown(wb.values[0])/toFloatUpDown(wb.values[1])
    nz3[1 , 1] = 1
    nz3[2 , 2] = toFloatUpDown(wb.values[2])/toFloatUpDown(wb.values[1])

    be  = tags.get('Image Tag 0xC62A')
    berate = 2 ** (toFloatUpDown(be.values[0]))

    return nz3 , berate



def upsidedown_wb(gt_img , wb_matrix):
    b = gt_img[:, :, np.newaxis, :]

    b = np.matmul(b, wb_matrix)

    c = np.squeeze(b)

    return c
