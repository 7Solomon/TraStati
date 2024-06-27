import argparse
from src.data_folder.get_system_image.get_data import getData,clean_folder
from src.data_folder.resize_image import resize_all_images
from src.data_folder.noise_image import randomize_all_images
from src.data_folder.rotate_image import rotate_all_images

import os
import shutil


def split_data(end_path,output_path):
    label_data_val, label_data_train = '', ''
    
    with open(f'{end_path}/label.txt','r') as file:
        data = file.read().split('\n')

        half = int(len(data)/2)
        for i, line in enumerate(data):
            id = line.split(':')[0]
            if id != '': 
                if i < half:
                    shutil.copyfile(f'{end_path}/{id}.jpg', f'{output_path}/train/{id}.jpg')
                    label_data_train = label_data_train + line + '\n'
                else:
                    shutil.copyfile(f'{end_path}/{id}.jpg', f'{output_path}/val/{id}.jpg')
                    label_data_val = label_data_val + line + '\n'
    with open(f'{output_path}/train/label.txt','a') as file:
        file.write(label_data_train)
    with open(f'{output_path}/val/label.txt','a') as file:
        file.write(label_data_val)
def create_valTrain_folder(path,n=10, randomize=False):  
    clean_folder('src/data_folder/get_system_image/img')
    clean_folder('src/data_folder/cut_images')
    clean_folder('src/data_folder/rotated_images')
    clean_folder('src/data_folder/noised_images')
    
    getData(n,randomize)
    resize_all_images()
    rotate_all_images('src/data_folder/cut_images','src/data_folder/rotated_images')
    randomize_all_images('src/data_folder/rotated_images','src/data_folder/noised_images')

    split_data('src/data_folder/noised_images',path)
    
    


if __name__ == '__main__':
    clean_folder('src/data_folder/test_dataloader/train')
    clean_folder('src/data_folder/test_dataloader/val')
    create_valTrain_folder('src/data_folder/test_dataloader')


    