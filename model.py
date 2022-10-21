import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

def attn(f_s, f_p):
    f_s_temp = torch.squeeze(f_s, dim=3)  # [b, 256, 1]
    f_p_temp = torch.squeeze(f_p, dim=3)  # [b, 2560, 1]
    b, w, h = f_p_temp.size(0), f_p_temp.size(1), f_p_temp.size(2)
    #print('---------', f_s_temp.size(), f_p_temp.size())
    f_p_temp_view = f_p_temp.view(int(b), int(w/3), int(h*3))  # [b, 256, 10] !!!
    att_map = torch.matmul(f_s_temp.transpose(1, 2), f_p_temp_view)  # [b, 1, 10]
    att_map = F.normalize(att_map, p=2, dim=2)
    f_s_out = torch.matmul(f_s_temp, att_map)  # [b, 256, 10]
    f_p_out = f_p_temp_view * att_map.repeat(1, 128, 1)  # [b, 256, 10]

    f_s_out = f_s_out.view(int(b), -1, 1)  # [b, 2560, 1]
    f_p_out = f_p_out.view(int(b), -1, 1)  # [b, 2560, 1]

    f_s_out = torch.unsqueeze(f_s_out, 3)  # [b, 2560, 1, 1]
    f_p_out = torch.unsqueeze(f_p_out, 3)  # [b, 2560, 1, 1]
    #print('---------', f_s_out.size(), f_p_out.size())

    return f_s_out, f_p_out

class FC(nn.Module):  # [b, 1024, 16, 16]
    def __init__(self):
        super(FC, self).__init__()
        self.conv11 = nn.Conv2d(1024, 256, 3, stride=2, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv12 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.nor1 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 128, 3, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv14 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.nor2 = nn.BatchNorm2d(128)

        self.conv21 = nn.Conv2d(1024*3, 256*3, 3, stride=2, padding=1)  # 1024*10, 1024*5, 1024*3
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv22 = nn.Conv2d(256*3, 256*3, 3, stride=1, padding=1)
        self.nor3 = nn.BatchNorm2d(256*3)
        self.conv23 = nn.Conv2d(256*3, 128*3, 3, stride=2, padding=1)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv24 = nn.Conv2d(128*3, 128*3, 3, stride=1, padding=1)
        self.nor4 = nn.BatchNorm2d(128*3)

        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.conv_slide = nn.Sequential(
            self.conv11,
            self.pool1,
            self.relu,
            self.conv12,
            self.relu,
            self.nor1,
            self.conv13,
            #self.pool2,
            self.relu,
            self.conv14,
            self.relu,
            self.nor2
        )

        self.conv_patch = nn.Sequential(
            self.conv21,
            self.pool3,
            self.relu,
            self.conv22,
            self.relu,
            self.nor3,
            self.conv23,
            #self.pool4,
            self.relu,
            self.conv24,
            self.relu,
            self.nor4
        )

        self.fc1 = nn.Linear(128*3*2, 32)  # *10 *5 *3
        self.fc2 = nn.Linear(32, 1)

        self.params = list(self.parameters())

    def forward(self, fs, fp):
        f_s = self.conv_slide(fs)
        f_p = self.conv_patch(fp)
        f_s_out, f_p_out = attn(f_s, f_p)  # [16, 2560, 1, 1]
        f_cat = torch.cat((f_s_out, f_p_out), 1)

        f_cat = f_cat.view(-1, 128*3*2)  # *10 *5 *3
        f1 = self.fc1(f_cat)
        f1_r = self.relu(f1)
        f2 = self.fc2(f1_r)

        return f2
