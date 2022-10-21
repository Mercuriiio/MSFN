import PIL
import os
import torch
from torch.utils import data
import pandas as pd
from util import *
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split

class PatchData(data.Dataset):
    def __init__(self, dataframe, split = None, transfer = None):
        self.dataframe = dataframe
        if split != None:
            index_split = self.dataframe[dataframe['Split'] == split].index
            self.dataframe = self.dataframe.loc[index_split, :]
        self.transfer = transfer
        self.length = len(self.dataframe)
        self.transfers_patch = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.734, 0.519, 0.659], std=[0.196, 0.240, 0.195])  # 10_patch GBM
            #transforms.Normalize(mean=[0.367, 0.259, 0.329], std=[0.295, 0.250, 0.271])  # 5_patch
            #transforms.Normalize(mean=[0.022, 0.02, 0.021], std=[0.136, 0.128, 0.133])  # 3_patch
            #transforms.Normalize(mean=[0.022, 0.02, 0.021], std=[0.136, 0.128, 0.133])  # slide

            #transforms.Normalize(mean=[1.332, 0.988, 1.246], std=[0.914, 0.728, 0.856])  # 10_patch LUSC
            #transforms.Normalize(mean=[0.666, 0.494, 0.623], std=[0.203, 0.241, 0.194])  # 5_patch
            transforms.Normalize(mean=[0.399, 0.296, 0.373], std=[0.275, 0.251, 0.259])  # 3_patch
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        patch = self.dataframe.iloc[item, :]
        patientID = patch['PatientID']
        patchID = patch['PatchID']
        #slide_path = "./data/TCGA-GBM-slide/" + patchID  # WSI Loader
        #patch_path = "./data/TCGA-GBM-10p-wsi_wise/" + patientID + "/"
        #patch_path = "./data/TCGA-GBM-5p-wsi_wise/" + patientID + "/"
        #patch_path = "./data/TCGA-GBM-3p-wsi_wise/" + patientID + "/"
        #patch_path = "./data/TCGA-GBM-10p/" + patientID + "/" + patchID

        slide_path = "./data/TCGA-LUSC-slide/" + patchID  # WSI Loader
        patch_path = "./data/TCGA-LUSC-3p-wsi_wise/" + patientID + "/"
        label = patch[['Month', 'Event']].values.astype('float')
        img = PIL.Image.open(slide_path)
        img = img.resize((128, 128), PIL.Image.LANCZOS)  # 256, 256
        patch = []
        dirs = os.listdir(patch_path)
        for dir in dirs:
            patch.append(PIL.Image.open(patch_path + dir))
        img1 = patch[0].resize((128, 128), PIL.Image.LANCZOS)
        img2 = patch[1].resize((128, 128), PIL.Image.LANCZOS)
        img3 = patch[2].resize((128, 128), PIL.Image.LANCZOS)
        # img4 = patch[3].resize((128, 128), PIL.Image.LANCZOS)
        # img5 = patch[4].resize((128, 128), PIL.Image.LANCZOS)
        # img6 = patch[5].resize((128, 128), PIL.Image.LANCZOS)
        # img7 = patch[6].resize((128, 128), PIL.Image.LANCZOS)
        # img8 = patch[7].resize((128, 128), PIL.Image.LANCZOS)
        # img9 = patch[8].resize((128, 128), PIL.Image.LANCZOS)
        # img0 = patch[9].resize((128, 128), PIL.Image.LANCZOS)
        if self.transfer != None:
            img = self.transfer(img)
            img1 = self.transfers_patch(img1)
            img2 = self.transfers_patch(img2)
            img3 = self.transfers_patch(img3)
            # img4 = self.transfers_patch(img4)
            # img5 = self.transfers_patch(img5)
            # img6 = self.transfers_patch(img6)
            # img7 = self.transfers_patch(img7)
            # img8 = self.transfers_patch(img8)
            # img9 = self.transfers_patch(img9)
            # img0 = self.transfers_patch(img0)
        return img, img1, img2, img3, label  #, img4, img5, img6, img7, img8, img9, img0

    @property
    def numofPatients(self):
        return len(self.dataframe['PatientID'].value_counts())

    @property
    def numofPatch(self):
        return self.dataframe['PatientID'].value_counts()

    @property
    def numoftotalPatch(self):
        return self.length

    def split_cluster(file, split, transfer = None):
        df = pd.read_csv(file)
        index0 = df[df['Cluster'] == 0].index  # 0,1,2...,125

        return PatchData(df.loc[index0, :], split=split, transfer=transfer)

    def load_split(file, split, transfer = None):
        df = pd.read_csv(file)
        return PatchData(df, split = split, transfer=transfer)

    def get_label(self):
        label = self.dataframe[['Month', 'Event']].values.astype('float')
        return label
