import os
import cv2
import PIL 
import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
from skimage import io

from torch.utils.data import DataLoader
from glob import glob 
import matplotlib.pyplot as plt
from random import randrange 



class BreakHis_Dataset_SSL(nn.Module):

    def __init__(self, train_path, training_method=None, transform = None, target_transform = None, augmentation_strategy = None, image_pair = [], pair_sampling_method = "OP"):
        
        self.train_path = train_path
        self.transform = transform
        self.target_transform = target_transform
        # self.pre_processing = pre_processing
        self.pair_sampling_method = pair_sampling_method
        self.image_dict_40x = {}
        self.image_dict_100x = {}
        self.image_dict_200x = {}
        self.image_dict_400x = {}
        



        for patient_dir_name in os.listdir(train_path):
            patient_uid = patient_dir_name.split('-')[1]
            
            #record keeping for 40X images
            path_40x = os.path.join(train_path, patient_dir_name, '40X')
            for image_name in os.listdir(path_40x):
                image_seq = image_name.split('.')[0].split('-')[4]
                self.image_dict_40x[patient_uid+'_'+ image_seq] = os.path.join(path_40x, image_name)
            
            #record keeping for 100X images
            path_100x = os.path.join(train_path, patient_dir_name, '100X')
            for image_name in os.listdir(path_100x):
                image_seq = image_name.split('.')[0].split('-')[4]
                if (patient_uid+'_'+ image_seq in list(self.image_dict_40x.keys())):
                    self.image_dict_100x[patient_uid+'_'+ image_seq] = os.path.join(path_100x, image_name)

            #record keeping for 200X images
            path_200x = os.path.join(train_path, patient_dir_name, '200X')
            for image_name in os.listdir(path_200x):
                image_seq = image_name.split('.')[0].split('-')[4]
                if ((patient_uid+'_'+ image_seq in list(self.image_dict_40x.keys())) and (patient_uid+'_'+ image_seq in list(self.image_dict_100x.keys()))):
                    self.image_dict_200x[patient_uid+'_'+ image_seq] = os.path.join(path_200x, image_name)

            #record keeping for 400X images
            path_400x = os.path.join(train_path, patient_dir_name, '400X')
            for image_name in os.listdir(path_400x):
                image_seq = image_name.split('.')[0].split('-')[4]
                if ((patient_uid+'_'+ image_seq in list(self.image_dict_40x.keys())) and (patient_uid+'_'+ image_seq in list(self.image_dict_100x.keys())) and (patient_uid+'_'+ image_seq in list(self.image_dict_200x.keys()))):
                    self.image_dict_400x[patient_uid+'_'+ image_seq] = os.path.join(path_400x, image_name)


        #SSL specific
        self.augmentation_strategy_1 = augmentation_strategy
        self.training_method = training_method
        self.image_pair = image_pair
        
        self.list_40X = list(self.image_dict_40x.keys())
        self.list_100X = list(self.image_dict_100x.keys())
        self.list_200X = list(self.image_dict_200x.keys())
        self.list_400X = list(self.image_dict_400x.keys())
        temp = list(set(self.list_40X) & set(self.list_100X) & set(self.list_200X) & set(self.list_400X))
        self.image_list = temp #list(self.image_dict_400x.keys())
        
        print ('pair_sampling_method - ', self.pair_sampling_method)
        print('len of 40x,100x,200x,400x', len(self.list_40X), len(self.list_100X), len(self.list_200X), len(self.list_400X))
       
                        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        
        image1_path, image2_path = None,None
        #Ordered sampling method - one magnification randomly, next is chosen orderly
        if "OP" == self.pair_sampling_method:
            randon_mgnification = randrange(4)
            if randon_mgnification == 0:
                image1_path = self.image_dict_40x[self.image_list[index]]
                image2_path = self.image_dict_100x[self.image_list[index]]
            elif randon_mgnification == 1:
                image1_path = self.image_dict_100x[self.image_list[index]]
                image2_path = self.image_dict_200x[self.image_list[index]]
            elif randon_mgnification == 2:
                image1_path = self.image_dict_200x[self.image_list[index]]
                image2_path = self.image_dict_400x[self.image_list[index]]
            elif randon_mgnification == 3:
                image1_path = self.image_dict_200x[self.image_list[index]]
                image2_path = self.image_dict_400x[self.image_list[index]]
                

        image1  = PIL.Image.open(image1_path)
        image2  = PIL.Image.open(image2_path)
        

        transformed_view1, transformed_view2 = None, None
        
        if self.training_method == "MPCS" : #current work only consider MPCS
            state = torch.get_rng_state()
            transformed_view1 = self.augmentation_strategy_1(image = np.array(image1))
            torch.set_rng_state(state)
            transformed_view2 = self.augmentation_strategy_1(image = np.array(image2))

            if self.transform:
                transformed_view1 = self.transform(transformed_view1['image'])
                transformed_view2 = self.transform(transformed_view2['image'])
            
            return transformed_view1, transformed_view2


def get_BreakHis_trainset_loader(train_path, training_method=None, transform = None,target_transform = None, augmentation_strategy = None, image_pair=[], pair_sampling_method = "OP", batch_size = 14, num_workers = 2):

    dataset = BreakHis_Dataset_SSL(train_path, training_method, transform, target_transform, augmentation_strategy, image_pair, pair_sampling_method)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last = True)
    return train_loader