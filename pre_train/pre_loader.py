import PIL
import os
import numpy as np
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

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        patch = self.dataframe.iloc[item, :]
        patientID = patch['PatientID']
        patchID = patch['PatchID']
        #patch_path = "../data/TCGA-LUSC-10p-wsi_wise/" + patientID + "/" + patchID
        patch_path = "../data/TCGA-LUSC-slide/" + patchID  # WSI Loader
        # label = patch[['Month', 'Event']].values.astype('float')
        img = load_np_image(patch_path)
        masked_img, ori_img, _ = mask_pixel(img, 0.8)
        masked_img = PIL.Image.fromarray(masked_img)
        ori_img = PIL.Image.fromarray(ori_img)
        if self.transfer != None:
            masked_img = self.transfer(masked_img)
            ori_img = self.transfer(ori_img)
        return masked_img, ori_img

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
