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
from torch.utils.tensorboard import SummaryWriter
from torch import autograd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
print('Current cuda device: ', torch.cuda.current_device()) # check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------ Options -------
bch_size_train = 8
bch_size_test = 8
epoch_size = 200
base_lr = 0.00001
writer = SummaryWriter('./log')
# ----------------------

transfers = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.734, 0.519, 0.659], std=[0.196, 0.240, 0.195])  # patch
        #transforms.Normalize(mean=[0.022, 0.02, 0.021], std=[0.136, 0.128, 0.133])  # slide
    ])

train0 = Loader.PatchData.split_cluster('./data/TCGA-GBM-inpainting.csv', 'Train', transfer=transfers)  # slide
valid0 = Loader.PatchData.split_cluster('./data/TCGA-GBM-inpainting.csv', 'Valid', transfer=transfers)  # inpainting
dataloader = DataLoader(train0, batch_size=bch_size_train, shuffle=True, num_workers=0)
dataloader_var = DataLoader(valid0, batch_size=bch_size_test, shuffle=True, num_workers=0)
criterion = NegativeLogLikelihood()

model = ResNet34()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0])
model.to(device)
#summary(model, (3, 256, 256))
model.train()

# optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

for epoch in range(epoch_size):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    loss_board = 0
    for iteration, data in loop:
        img, label = data
        img = Variable(img, requires_grad=False).cuda()
        label = Variable(label, requires_grad=False).cuda()
        batch_size = img.size(0)

        # with autograd.detect_anomaly():
        pred = model(img)
        loss = criterion(pred, label)
        # Backward and update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_board += loss
        # print('Epoch('+str(epoch+1)+'), iteration('+str(iteration+1)+'): '+str(loss.item()))
        loop.set_description(f'Epoch [{epoch}/{epoch_size}]')
        loop.set_postfix(loss=loss.item()/batch_size)

    writer.add_scalar("Train_loss", loss_board, epoch)
    # scheduler.step(loss_board)

    if epoch % 20 == 19:
        torch.save(model.state_dict(), './model/epoch_'+str(epoch+1)+'_model.pt')

    if epoch in [40]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * base_lr
    if epoch in [80]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.05 * base_lr
    if epoch in [100]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001 * base_lr
    if epoch in [120]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005 * base_lr
    if epoch in [140]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001 * base_lr

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
    y, risk_pred, e = ytime.detach().cpu().numpy(), pred_test.detach().cpu().numpy(), yevent.detach().cpu().numpy()
    accuracy += con_index(y, risk_pred, e)
    print(n, ':', con_index(y, risk_pred, e))
print('Accuracy: %.4f' % (accuracy/n))

