import torch
import cv2
import numpy as np

from torchvision import transforms

from src.data_folder.manage_datasets import load_datasets
from src.neural_network_stuff.custome_DETR.misc_stuff import nested_tensor_from_tensor_list
from src.visualize.draw_graph import draw_stuff_on_image_and_save, get_degree_lines
from src.visualize.visualize_attention_map import attention_map
from src.neural_network_stuff.custome_DETR.detr import build

transform_reverse = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Resize((840, 960)),
])


def visualize_output(train_set, model_name, idx):
    image = train_set.image_dic[train_set.id_list[idx]]
    image_array = np.array(image)


    model, criterion = build()
    model.load_state_dict(torch.load(f'src/neural_network_stuff/models/{model_name}'))
    

    # Muss besser gelöst werden
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    # Output generieren
    item_img, item_label = train_set.__getitem__(idx)

    samples = nested_tensor_from_tensor_list([item_img])          # Das eine Batch kommt []
    samples = samples.to(device)
    
    output = model(samples)

    # Punkte und Winkel des Outputs
    points = [(e[0].item(),e[1].item()) for e in output['output_center_degree_points'][0]]
    degrees = [e[2].item() for e in output['output_center_degree_points'][0]]
 
    points = [(int(840*x), int(960*y)) for x,y in points]
    degrees = [int(360/64*deg) for deg in degrees]

    #print(f'points: {points}')
    #print(f'degrees: {degrees}')

    # Zeichnen der Punkte und Winkel
    degree_lines = get_degree_lines(points, degrees)
    drawn_on_image = draw_stuff_on_image_and_save(image_array,points,degree_lines)
    
    # For Attention map und display der punkte
    attention_map(output["attention_weights"], drawn_on_image)
    
    target = [{k: v.to(device) for k, v in item_label.items()}]   # [] für Batch
    loss = criterion(output, target)
    print(f'loss: {loss}')