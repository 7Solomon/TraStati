import ast
import os
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from visualize.draw_graph import get_degree_lines, draw_stuff_on_image_and_save, get_points_from_label

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust the size as needed
    transforms.ToTensor(),
])

def target_transform(label):
    parsed_data = label
    tensor_data = []
    class_data = []
    
    for i, item in enumerate(parsed_data):
            
        condensed_item = [item[0][0],item[0][1],*item[2:]]
        tensor_item = torch.tensor(condensed_item, dtype=torch.float32)  
        tensor_data.append(tensor_item)
        class_data.append(item[1])

    # Muss gelten
    assert len(tensor_data) == len(class_data)

    if len(tensor_data) > 20:
        print('Du hast eine System mit mehr als 20 knoten, passe auf nur die ersten 20 werden genommen')
        transformed_data = torch.stack(tensor_data[:20])
        transformed_class_data = torch.stack(class_data[:20])
    else:
        transformed_data = torch.stack([*tensor_data, *[torch.tensor([0, 0, 0], dtype=torch.float32) for _ in range(20 - len(tensor_data))]])
        transformed_class_data = torch.tensor([*class_data, *[0 for _ in range(20 - len(tensor_data))]])
    
    return {'classes': transformed_class_data,'data':transformed_data}



class CustomImageDataset(Dataset):
    def __init__(self, path, transform=transform,target_transform=target_transform):
        self.transform = transform
        self.target_transform = target_transform

        self.id_list = [] 
        self.label_dic = {}
        self.image_dic = {}

        self.get_data(path)

    def get_images(self, img_ids, path):
        for id in img_ids:
            with Image.open(f'{path}/{id}.jpg').convert("RGB") as img:
                self.image_dic[id] = img
    def get_data(self,path):
        new_images_to_add = []
        with open(f'{path}/label.txt') as label_file:
            label_list = label_file.read().split('\n')
            for object in label_list:
                split_object = object.split(':')
                if len(split_object) == 2:
                    key = split_object[0]
                    value = split_object[1].split('|')
                    value = [ast.literal_eval(e) for e in value]
                    self.label_dic[key] = value

                    self.id_list.append(key)
                    new_images_to_add.append(key)
        self.get_images(new_images_to_add,path)   

        

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

    def add_new_data(self, new_path):
        self.get_data(new_path)

    def display_data(self,idx=0):
        id = self.id_list[idx]
        img = self.image_dic[id]

        points,degrees = get_points_from_label(self.label_dic[id])
        degree_lines = get_degree_lines(points, degrees)
        draw_stuff_on_image_and_save(img,points,degree_lines)



if __name__ == '__main__':

    dataset = CustomImageDataset('/home/johannes/Dokumente/programme/transfomer_model/data/test_dataloader/train/label.txt','/home/johannes/Dokumente/programme/transfomer_model/data/test_dataloader/train')
    dataset.__getitem__(0)
    batch_size = 64  # Adjust as needed
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #print(data_loader)
