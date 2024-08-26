import ast
import os
import numpy as np

import cv2

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from src.visualize.draw_graph import get_degree_lines, draw_stuff_on_image_and_save, get_points_from_label
from src.visualize.visualize_image import visualize_image

import random



transform = transforms.Compose([
        transforms.ToTensor(),  # Konvertiert das Bild in einen Tensor
        transforms.Resize((840, 960)),  # Ändert die Größe des Bildes
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisiert die Pixelwerte
    ])

transform_reverse = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Resize((840, 960)),
])


def target_transform(label:dict):
    items_data = []
    class_data = []
    
    for item in label.values():
        items_data.append([*item['koordinaten'],item['rotation']])
        class_data.append(item['type'])
    
    # Change format
    tensor_data = torch.tensor([[x/840, y/960, degree/360] for x,y,degree in items_data], dtype=torch.float32)
    assert len(tensor_data) == len(class_data) 
    
    # Get Transform data 
    if len(tensor_data) > 20:
        print('Du hast eine System mit mehr als 20 knoten, passe auf nur die ersten 20 werden genommen')
        transformed_data = torch.stack(tensor_data[:20])
        transformed_class_data = torch.stack(class_data[:20])
        
    else:
        transformed_data = torch.stack([*tensor_data, *[torch.tensor([0, 0, 0], dtype=torch.float32) for _ in range(20 - len(tensor_data))]])
        transformed_class_data = torch.tensor([*class_data, *[0 for _ in range(20 - len(tensor_data))]])
    return {'classes': transformed_class_data,'data':transformed_data}



class CustomImageDataset(Dataset):
    def __init__(self):
        self.transform = transform
        self.target_transform = target_transform

        self.id_list = [] 
        self.label_dic = {}
        self.image_dic = {}

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]
        
        image = self.transform(self.image_dic[id])
        label = self.target_transform(self.label_dic[id])
        return image, label
    
    def merge_dataset(self,dataset):
        self.id_list.extend(dataset.id_list)
        self.label_dic.update(dataset.label_dic)
        self.image_dic.update(dataset.image_dic)

    def add_new_img(self, img, id, label_data):
        self.image_dic[id] = img
        self.label_dic[id] = label_data

        self.id_list.append(id)

    def display_data(self, idx:int , save:bool = False):
        id = self.id_list[idx]
        img = self.image_dic[id]

        points,degrees = get_points_from_label(self.label_dic[id])
        degree_lines = get_degree_lines(points, degrees)

        img_array = np.array(img)        
        img = draw_stuff_on_image_and_save(img_array,points,degree_lines)

        if save:
            cv2.imwrite('assets/test_output_image.jpg', img)

        visualize_image(img, f'Image nr. {idx}')
    

    def save_to_file(self, dataset_name):
        """
        Save the dataset to files, automatically splitting into train and val sets.
        
        :param dataset_name: Name of the dataset (used in the file path)
        """
        # Create the directory if it doesn't exist
        save_dir = os.path.join('src', 'data_folder', 'datasets', dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # Determine the split
        total_images = len(self.id_list)
        if total_images < 100:
            val_size = max(5, int(0.2 * total_images))
        elif total_images < 1000:
            val_size = int(0.1 * total_images)
        else:
            val_size = int(0.05 * total_images)
        train_size = total_images - val_size

        # Randomly shuffle the id_list
        shuffled_ids = random.sample(self.id_list, len(self.id_list))

        # Create train and val datasets
        train_dataset = CustomImageDataset()
        val_dataset = CustomImageDataset()

        # Populate train dataset
        for id in shuffled_ids[:train_size]:
            train_dataset.add_new_img(self.image_dic[id], id, self.label_dic[id])

        # Populate val dataset
        for id in shuffled_ids[train_size:]:
            val_dataset.add_new_img(self.image_dic[id], id, self.label_dic[id])

        # Save the datasets
        torch.save(train_dataset, os.path.join(save_dir, 'train.pt'))
        torch.save(val_dataset, os.path.join(save_dir, 'val.pt'))

        print(f"Datensatz gesaven in {save_dir}")
        print(f"Train set size: {len(train_dataset)}")
        print(f"Val set size: {len(val_dataset)}")



if __name__ == '__main__':

    dataset = CustomImageDataset('/home/johannes/Dokumente/programme/transfomer_model/data/test_dataloader/train/label.txt','/home/johannes/Dokumente/programme/transfomer_model/data/test_dataloader/train')
    dataset.__getitem__(0)
    batch_size = 64  # Adjust as needed
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #print(data_loader)
