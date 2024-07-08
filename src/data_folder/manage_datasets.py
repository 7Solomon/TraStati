import torch
import os
from src.neural_network_stuff.custome_dataset import CustomImageDataset

from src.data_folder.get_system_image.get_data import getData
from src.data_folder.resize_image import resize
from src.data_folder.noise_image import randomize_image
from src.data_folder.rotate_image import rotate_image


import random

def create_random_image():  
    id = ''.join(random.choices('0123456789abcdef', k=12))

    img, label = getData()

    img, label = resize(img, label)
    img, label = rotate_image(img, label)
    img, label = randomize_image(img, label)

    return img, id, label
    
   
def load_datasets(dataset_name):
    training_set = torch.load(f'src/data_folder/datasets/{dataset_name}/train.pt')
    val_set = torch.load(f'src/data_folder/datasets/{dataset_name}/val.pt')
    return training_set, val_set

        
def add_to_datasets(d_1,d_2):
    d_1.merge_dataset(d_2)   
    return d_1 

def clear_label_files():
    with open('src/data_folder/test_dataloader/train/label.txt','w') as label_file:
        label_file.write('')
    with open('src/data_folder/test_dataloader/val/label.txt','w') as label_file:
        label_file.write('')






