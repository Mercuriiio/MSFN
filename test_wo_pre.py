from model import *
from network import U_Net
from vgg16 import VGG16
from resnet34 import ResNet34
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
print('Current cuda device: ', torch.cuda.current_device()) # check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------ Options -------
bch_size_test = 51
# ----------------------

transfers = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.734, 0.519, 0.659], std=[0.196, 0.240, 0.195])  # patch
        #transforms.Normalize(mean=[0.022, 0.02, 0.021], std=[0.136, 0.128, 0.133])  # slide
    ])

#valid0 = Loader.PatchData.split_cluster('./data/TCGA-GBM-slide.csv', 'Valid', transfer=transfers)
valid0 = Loader.PatchData.split_cluster('./data/TCGA-GBM-inpainting.csv', 'Valid', transfer=transfers)
dataloader_var = DataLoader(valid0, batch_size=bch_size_test, shuffle=True, num_workers=0)

model = ResNet34()
model.load_state_dict(torch.load('./model/epoch_20_model.pt'))
model.to(device)
# summary(model, (1024, 16, 16))
model.eval()

accuracy = 0
n = 0
for iteration, data in enumerate(dataloader_var):
    n += 1.0
    img_var, label_var = data
    img_var = Variable(img_var, requires_grad=False).cuda()
    label_var = Variable(label_var, requires_grad=False).cuda()
    ytime, yevent = label_var[:, 0], label_var[:, 1]

    with torch.no_grad():
        pred_test = model(img_var)
    #accuracy += concordance_index(pred_test, label_var)
    y, risk_pred, e = ytime.detach().cpu().numpy(), pred_test.detach().cpu().numpy(), yevent.detach().cpu().numpy()
    accuracy += con_index(y, risk_pred, e)
    print(n, ':', con_index(y, risk_pred, e))
print('Accuracy: %.4f' % (accuracy/n))

