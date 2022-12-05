import openslide
import numpy as np
import imageio
import cv2
import os
from openslide.deepzoom import DeepZoomGenerator

from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None



tcga_path = "./Test_TCGA-LUSC/TCGA-LUSC/"
tcga_files = os.listdir(tcga_path)
N = len(tcga_files)
i = 0
for tcga_file in tcga_files:
    i += 1
    slide = openslide.open_slide(tcga_path+tcga_file)

    slide_thumbnail = slide.get_thumbnail((512 , 512))
    
    #保存到指定位置
    path_thumbnail = './1/' + str(tcga_file)[0:23] + '.jpg'
    path_out = './1/' + str(tcga_file)[0:23] + '.jpg'
    
    slide_thumbnail.save(path_thumbnail)
    img = cv2.imread(path_thumbnail, 1)
    img_test1 = cv2.resize(img, (512, 512))
    cv2.imwrite(path_out, img_test1)

    #关闭io
    slide.close()
    print(i, tcga_file)

# data_gen = DeepZoomGenerator(slide, tile_size=highth, overlap=0, limit_bounds=False)

# print(data_gen.tile_count)
# print(data_gen.level_count)
# print(slide.level_dimensions[0])

# w = slide.level_dimensions[0][0]
# h = slide.level_dimensions[0][1]

# num_w = int(np.floor(w/width))+1
# num_h = int(np.floor(h/highth))+1
# result_path = 'output/'
# for i in range(num_w):
    # for j in range(num_h):
        # img = np.array(data_gen.get_tile(16, (i, j))) #切图
        # imageio.imsave(result_path + "02"+str(i)+'_'+str(j)+".png", img) #保存图像