import os
import PIL
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import bc_config


class BreakHis_Dataset(Dataset):
    
    def _populate_image_dict(self, path, image_dict, binary_label, multi_label):
        for image_name in os.listdir(path):
            image_seq = image_name.split('.')[0].split('-')[4]
            image_key = f"{self.patient_uid}_{image_seq}"
            image_dict[image_key] = os.path.join(path, image_name)
            self.label_binary_dict[image_key] = binary_label
            self.label_multi_dict[image_key] = multi_label
    
    def __init__(self, train_path, transform = None, augmentation_strategy = None, pre_processing = [], image_type_list = []):

        # Standard setting for - dataset path, augmentation, trnformations, preprocessing, etc. 
        self.train_path = train_path
        self.transform = transform
        self.pre_processing = pre_processing
        self.augmentation_strategy = augmentation_strategy
        self.image_type_list = image_type_list

        #key pairing for image examples
        self.image_dict_40x = {}
        self.image_dict_100x = {}
        self.image_dict_200x = {}
        self.image_dict_400x = {}

        #key pairing for labels
        self.label_binary_dict = {}
        self.label_multi_dict = {}
        
        if not os.path.exists(train_path):
            raise ValueError(f"Provided train_path {train_path} doesn't exist.")

        #print(os.listdir(train_path))
        for patient_dir_name in os.listdir(train_path):
            self.patient_uid = patient_dir_name.split('-')[1]
            binary_label = patient_dir_name.split('_')[1]
            multi_label = patient_dir_name.split('_')[2]
            
            magnifications = ["40X", "100X", "200X", "400X"]
            dicts = [self.image_dict_40x, self.image_dict_100x, self.image_dict_200x, self.image_dict_400x]

            for mag, d in zip(magnifications, dicts):
                path = os.path.join(train_path, patient_dir_name, mag)
                self._populate_image_dict(path, d, binary_label, multi_label)

        # Mapping of magnification types to their lists
        self.dict_magnification_list = {
        bc_config.X40: list(self.image_dict_40x.keys()),
        bc_config.X100: list(self.image_dict_100x.keys()),
        bc_config.X200: list(self.image_dict_200x.keys()),
        bc_config.X400: list(self.image_dict_400x.keys())
        }
        
        # Compute the intersection of image sets across magnification levels
        img_list = set(self.dict_magnification_list[self.image_type_list[0]])
        for magnification_level in self.image_type_list[1:]:
            img_list.intersection_update(self.dict_magnification_list[magnification_level])
        self.image_list = list(img_list)
        
        
        # Print lengths for debugging
        print(len(self.dict_magnification_list[bc_config.X40]), 
            len(self.dict_magnification_list[bc_config.X100]), 
            len(self.dict_magnification_list[bc_config.X200]), 
            len(self.dict_magnification_list[bc_config.X400]))
        print(len(self.image_list))
        
                        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        item_dict = {}
        patient_id = self.image_list[index].split('_')[0]
        
        magnifications = [bc_config.X400, bc_config.X200, bc_config.X100, bc_config.X40]
        image_dicts = [self.image_dict_400x, self.image_dict_200x, self.image_dict_100x, self.image_dict_40x]

        for mag, d in zip(magnifications, image_dicts):
            if mag in self.image_type_list:
                item_dict[mag] = PIL.Image.open(d[self.image_list[index]])
        
        state = torch.get_rng_state()
        if self.augmentation_strategy:
            for mg_level, img in item_dict.items():
                torch.set_rng_state(state)
                item_dict[mg_level] = self.augmentation_strategy(image=np.array(img))

        if self.transform:
            for mg_level, img in item_dict.items():
                if not self.augmentation_strategy and not self.pre_processing:
                    item_dict[mg_level] = self.transform(np.array(img))
                elif self.augmentation_strategy:
                    item_dict[mg_level] = self.transform(img['image'])
                else:
                    item_dict[mg_level] = self.transform(img)

        return (patient_id, 
                list(item_dict.keys())[0], 
                item_dict, 
                bc_config.binary_label_dict[self.label_binary_dict[self.image_list[index]]], 
                bc_config.multi_label_dict[self.label_multi_dict[self.image_list[index]]])

class BreakHis_Dataset_400x(Dataset):
    
    def _populate_image_dict(self, path, image_dict, binary_label, multi_label):
        for image_name in os.listdir(path):
            image_seq = image_name.split('.')[0].split('-')[4]
            image_key = f"{self.patient_uid}_{image_seq}"
            image_dict[image_key] = os.path.join(path, image_name)
            self.label_binary_dict[image_key] = binary_label
            self.label_multi_dict[image_key] = multi_label
    
    def __init__(self, train_path, transform=None, augmentation_strategy=None, pre_processing=[]):

        # Standard settings
        self.train_path = train_path
        self.transform = transform
        self.pre_processing = pre_processing
        self.augmentation_strategy = augmentation_strategy

        self.image_dict_400x = {}
        self.label_binary_dict = {}
        self.label_multi_dict = {}

        if not os.path.exists(train_path):
            raise ValueError(f"Provided train_path {train_path} doesn't exist.")

        for patient_dir_name in os.listdir(train_path):
            self.patient_uid = patient_dir_name.split('-')[1]
            binary_label = patient_dir_name.split('_')[1]
            multi_label = patient_dir_name.split('_')[2]

            path = os.path.join(train_path, patient_dir_name, "400X")
            self._populate_image_dict(path, self.image_dict_400x, binary_label, multi_label)

        self.image_list = list(self.image_dict_400x.keys())
        print(len(self.image_list))
        
                        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        img_path = self.image_dict_400x[self.image_list[index]]
        img = PIL.Image.open(img_path)

        state = torch.get_rng_state()
        if self.augmentation_strategy:
            torch.set_rng_state(state)
            augmented = self.augmentation_strategy(image=np.array(img))
            img = augmented['image']

        if self.transform:
            img = self.transform(img)

        binary_label = bc_config.binary_label_dict[self.label_binary_dict[self.image_list[index]]]
        multi_label = bc_config.multi_label_dict[self.label_multi_dict[self.image_list[index]]]

        return img, binary_label


def get_breakhis_data_loader(dataset_path, transform=None, augmentation_strategy=None, pre_processing=None, image_type_list=None, batch_size=32, num_workers=2, is_test=False):
    """
    Returns a DataLoader for the BreakHis dataset.
    
    Parameters:
    - dataset_path (str): Path to the dataset.
    - transform: Transformations to be applied to the images.
    - augmentation_strategy: Data augmentation methods for training.
    - pre_processing: Pre-processing steps to apply to the images.
    - image_type_list (list, optional): List of image types to use. Defaults to an empty list.
    - batch_size (int): Size of the batches.
    - num_workers (int): Number of workers for data loading.
    - is_test (bool): If True, returns a test data loader, otherwise returns a training data loader.

    Returns:
    DataLoader: DataLoader object for the dataset.
    """
    
    if image_type_list is None:
        image_type_list = []

    if is_test:
        augmentation_strategy = None
        shuffle = False
    else:
        shuffle = True

    dataset = BreakHis_Dataset(train_path=dataset_path, transform=transform, augmentation_strategy=augmentation_strategy, pre_processing=pre_processing, image_type_list=image_type_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return loader

def get_400_data_loader(dataset_path, transform=None, augmentation_strategy=None, pre_processing=None,  batch_size=10, num_workers=2, is_test=False):
    """
    Returns a DataLoader for the BreakHis dataset.
    
    Parameters:
    - dataset_path (str): Path to the dataset.
    - transform: Transformations to be applied to the images.
    - augmentation_strategy: Data augmentation methods for training.
    - pre_processing: Pre-processing steps to apply to the images.
    - image_type_list (list, optional): List of image types to use. Defaults to an empty list.
    - batch_size (int): Size of the batches.
    - num_workers (int): Number of workers for data loading.
    - is_test (bool): If True, returns a test data loader, otherwise returns a training data loader.

    Returns:
    DataLoader: DataLoader object for the dataset.
    """
    
    

    if is_test:
        augmentation_strategy = None
        shuffle = False
    else:
        shuffle = True

    dataset = BreakHis_Dataset_400x(train_path=dataset_path, transform=transform, augmentation_strategy=augmentation_strategy, pre_processing=pre_processing)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return loader
