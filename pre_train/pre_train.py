# from eccv16 import ECCVGenerator
from network import U_Net
import pre_loader
from util import *
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import random
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device: ', torch.cuda.current_device()) # check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#------ Options -------
bch_size_train = 16
epoch_size = 80
base_lr = 0.0005
writer = SummaryWriter('./log')
#----------------------

transfers = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.04, 0.037, 0.039], std=[0.179, 0.169, 0.176])
    ])

train0 = pre_loader.PatchData.split_cluster('../data/TCGA-LUSC-slide.csv', 'Train', transfer=transfers)
dataloader = DataLoader(train0, batch_size=bch_size_train, shuffle=True, num_workers=0)

model = U_Net()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0])
model.to(device)
# summary(model, (1, 256, 256))
model.train()

# optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.001)
optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.5, 0.99))  # eps=1e-08, weight_decay=0.001
criterion = nn.MSELoss(reduction='mean')


for epoch in range(epoch_size):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    loss_board = 0
    i = 0
    for iteration, data in loop:
        mask_img, img = data
        mask_img = Variable(mask_img, requires_grad=False).cuda()
        img = Variable(img, requires_grad=False).cuda()
        batch_size = img.size(0)
        i += 1

        pred = model(mask_img)
        loss = criterion(pred, img)
        # Backward and update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_board += loss
        #print('Epoch('+str(epoch+1)+'), iteration('+str(iteration+1)+'): '+str(loss.item()))
        loop.set_description(f'Epoch [{epoch}/{epoch_size}]')
        loop.set_postfix(loss=loss.item()/batch_size)

    writer.add_scalar("Loss", loss_board/i, epoch)

    if epoch % 20 == 19:
        torch.save(model.state_dict(), './model/epoch_'+str(epoch+1)+'_model.pt')
    if epoch in [20]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1 * base_lr
    if epoch in [40]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01 * base_lr
    if epoch in [60]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001 * base_lr

writer.close()
