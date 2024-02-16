import sys, math
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from neural_network_stuff.custome_DETR.detr import build
from neural_network_stuff.custome_DETR.misc_stuff import nested_tensor_from_tensor_list, collate_fn
from neural_network_stuff.custome_DETR import misc_stuff

from functions import load_datasets

transform = transforms.Compose([
        transforms.ToTensor(),  # Konvertiert das Bild in einen Tensor
        transforms.Resize((840, 960)),  # Ändert die Größe des Bildes
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisiert die Pixelwerte
    ])



def test():

    train, val = load_datasets('test')
    t1 = train.__getitem__(0)

    IMG_URL = 'images/WhatsApp Image 2023-08-12 at 14.00.07(6).jpg'
    with Image.open(IMG_URL) as img:

        # get image
        img_tensor = transform(img)
        nested_img_tensor = nested_tensor_from_tensor_list([img_tensor])      

       

    



if __name__ == '__main__':
    test()
