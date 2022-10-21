import torch
import torch.nn as nn

class Basic(nn.Module):
	def __init__(self, in_planes, planes, stride):
		super(Basic, self).__init__()
		self.flag = False

		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,bias=False)
		self.bn1 = nn.BatchNorm2d(planes)

		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		if stride != 1 or in_planes != planes:
			self.flag = True
		self.shortcut_conv = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
		self.shortcut_bn = nn.BatchNorm2d(planes)

		self.relu = torch.nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		if self.flag:
			x = self.shortcut_bn(self.shortcut_conv(x))
		out += x
		out = self.relu(out)

		return out


class ResNet34(nn.Module):
	def __init__(self):
		super(ResNet34, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)

		self.layer1_block1 = Basic(in_planes=64, planes=64, stride=1)
		self.layer1_block2 = Basic(in_planes=64, planes=64, stride=1)
		self.layer1_block3 = Basic(in_planes=64, planes=64, stride=1)

		self.layer2_block1 = Basic(in_planes=64, planes=128, stride=2)
		self.layer2_block2 = Basic(in_planes=128, planes=128, stride=1)
		self.layer2_block3 = Basic(in_planes=128, planes=128, stride=1)
		self.layer2_block4 = Basic(in_planes=128, planes=128, stride=1)

		self.layer3_block1 = Basic(in_planes=128, planes=256, stride=2)
		self.layer3_block2 = Basic(in_planes=256, planes=256, stride=1)
		self.layer3_block3 = Basic(in_planes=256, planes=256, stride=1)
		self.layer3_block4 = Basic(in_planes=256, planes=256, stride=1)
		self.layer3_block5 = Basic(in_planes=256, planes=256, stride=1)
		self.layer3_block6 = Basic(in_planes=256, planes=256, stride=1)

		self.layer4_block1 = Basic(in_planes=256, planes=512, stride=2)
		self.layer4_block2 = Basic(in_planes=512, planes=512, stride=1)
		self.layer4_block3 = Basic(in_planes=512, planes=512, stride=1)

		self.avg_pool = nn.AvgPool2d(kernel_size=4)
		self.linear = nn.Sequential(
			nn.Linear(512*8*8, 4096),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 1),
		)

		self.relu = torch.nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.bn1(self.conv1(x)))

		out = self.layer1_block1(out)
		out = self.layer1_block2(out)
		out = self.layer1_block3(out)

		out = self.layer2_block1(out)
		out = self.layer2_block2(out)
		out = self.layer2_block3(out)
		out = self.layer2_block4(out)

		out = self.layer3_block1(out)
		out = self.layer3_block2(out)
		out = self.layer3_block3(out)
		out = self.layer3_block4(out)
		out = self.layer3_block5(out)
		out = self.layer3_block6(out)

		out = self.layer4_block1(out)
		out = self.layer4_block2(out)
		out = self.layer4_block3(out)

		out = self.avg_pool(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)

		return out
