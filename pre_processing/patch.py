import os
import shutil
import cv2
vipshome = r'D:\B代码\vips-dev-8.12\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
import pyvips
from entropy2dSpeedUp import calcEntropy2dSpeedUp

tcga_path = "Test/"
tcga_files = os.listdir(tcga_path)
dir_N = len(tcga_files)
dir_i = 0
for tcga_file in tcga_files:
    dir_i += 1
    TCGA_svs = tcga_path + tcga_file
    print('Processing the .svs file...')
    img = pyvips.Image.new_from_file(TCGA_svs, access='sequential')
    img.dzsave('output', tile_height=1024, tile_width=1024, overlap=0)  # 在这里定义输出patch的大小
    print('Sucessful processing the .svs file')
    os.remove("output.dzi")

    output = 'output/'
    if not os.path.exists(output):
        os.makedirs(output)

    path = 'output_files/'
    if os.path.exists(path):
        files = os.listdir(path)
        num = []
        for file in files:
            num.append(int(file))
        n = max(num)
        img_path = path + str(num) + '/'
        imgs = os.listdir(img_path)

        var = []
        var_path = []
        image_index = [] * 20
        image_path = [] * 20
        i = 0
        N = len(imgs)
        for img in imgs:
            i += 1
            print(i, '/', N, dir_i, '/', dir_N)
            svs_path = img_path + img
            image = cv2.imread(svs_path)
            image_entropy = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
            image_entropy = cv2.cvtColor(image_entropy, cv2.COLOR_RGB2GRAY)
            var.append(calcEntropy2dSpeedUp(image_entropy, 3, 3))
            var_path.append(svs_path)

        for _ in range(20):
            number = max(var)
            index = var.index(number)
            print(index, ':', var[index])
            var[index] = 0
            image_path.append(var_path[index])
            image_index.append(index)

        img_out = output + tcga_file + '/'
        os.makedirs(img_out)
        for path in image_path:
            shutil.move(path, img_out)
        shutil.rmtree("output_files/")
    else:
        print("No output_files...")
