from model import *
from network import U_Net
import Loader
from NegativeLogLikelihood import NegativeLogLikelihood
from score import c_index, concordance_index

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torchsummary import summary
from lifelines.utils import concordance_index as con_index
from torch import autograd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
print ('Current cuda device: ', torch.cuda.current_device()) # check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------ Options -------
bch_size_test = 16
# ----------------------

transfers = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.022, 0.02, 0.021], std=[0.136, 0.128, 0.133])  # slide
#         transforms.Normalize(mean=[0.04, 0.037, 0.039], std=[0.179, 0.169, 0.176])  # slide_lusc
    ])

# valid0 = Loader.PatchData.split_cluster('./data/TCGA-LUSC-slide.csv', 'Valid', transfer=transfers)
valid0 = Loader.PatchData.split_cluster('./data/TCGA-GBM-slide.csv', 'Valid', transfer=transfers)
dataloader_var = DataLoader(valid0, batch_size=bch_size_test, shuffle=False, num_workers=0)

model = FC()
model.load_state_dict(torch.load('./model/gbm_epoch_80_model.pt'))
#model.load_state_dict(torch.load('./model/inpainting_patch20-200/epoch_20_model.pt'))
model.to(device)
# summary(model, (1024, 16, 16))
model.eval()

pre_model_slide = U_Net()
pre_model_slide.load_state_dict(torch.load('./pre_train/model/inpainting_wsi_60/epoch_60_model.pt'))
pre_model_slide.to(device)
pre_model_slide.eval()

pre_model_patch = U_Net()
pre_model_patch.load_state_dict(torch.load('./pre_train/model/inpainting_patch_60/epoch_60_model.pt'))
pre_model_patch.to(device)
pre_model_patch.eval()

accuracy = 0
n = 0
for iteration, data in enumerate(dataloader_var):
    n += 1.0
    img_var, img1, img2, img3, label_var = data  #, img4, img5, img6, img7, img8, img9, img0
    ytime, yevent = label_var[:, 0], label_var[:, 1]
    img_var = Variable(img_var, requires_grad=False).cuda()
    label_var = Variable(label_var, requires_grad=False).cuda()
    img1 = Variable(img1, requires_grad=False).cuda()
    img2 = Variable(img2, requires_grad=False).cuda()
    img3 = Variable(img3, requires_grad=False).cuda()
    img4 = Variable(img4, requires_grad=False).cuda()
    img5 = Variable(img5, requires_grad=False).cuda()
    img6 = Variable(img6, requires_grad=False).cuda()
    img7 = Variable(img7, requires_grad=False).cuda()
    img8 = Variable(img8, requires_grad=False).cuda()
    img9 = Variable(img9, requires_grad=False).cuda()
    img0 = Variable(img0, requires_grad=False).cuda()

    with torch.no_grad():
        # pred_feature = pre_model(img_var)
        # pred_test = model(pred_feature)
        feature = pre_model_slide(img_var)  # [8, 1024, 16, 16]
        feature1 = pre_model_patch(img1)
        feature2 = pre_model_patch(img2)
        feature3 = pre_model_patch(img3)
        feature4 = pre_model_patch(img4)
        feature5 = pre_model_patch(img5)
        feature6 = pre_model_patch(img6)
        feature7 = pre_model_patch(img7)
        feature8 = pre_model_patch(img8)
        feature9 = pre_model_patch(img9)
        feature0 = pre_model_patch(img0)
        feature_cat = torch.cat((feature, feature1, feature2, feature3, feature4, feature5, feature6, feature7,
                                 feature8, feature9, feature0,), 1)
        # feature_patch = torch.cat((feature1, feature2, feature3, feature4, feature5, feature6, feature7,
        #                          feature8, feature9, feature0,), 1)
        # feature_patch = torch.cat((feature1, feature2, feature3, feature4, feature5), 1)
#         feature_patch = torch.cat((feature1, feature2, feature3), 1)
        # with autograd.detect_anomaly():
        pred_test = model(feature, feature_patch)
    accuracy += concordance_index(pred_test, label_var)
    #print('+++++', pred_test.data, '-----', label_var.data)
    print(int(n), ':', concordance_index(pred_test, label_var))
    # y, risk_pred, e = ytime.detach().cpu().numpy(), pred_test.detach().cpu().numpy(), yevent.detach().cpu().numpy()
    # accuracy += con_index(y, risk_pred, e)
    # print(n, ':', con_index(y, risk_pred, e))
print('Accuracy: %.4f' % (accuracy/n))

