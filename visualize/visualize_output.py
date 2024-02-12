import torch

from data_folder.manage_datasets import load_datasets
from visualize.draw_graph import draw_stuff_on_image_and_save, get_degree_lines
from neural_network_stuff.custome_DETR.detr import build

def visualize_output(train_set, model_name, idx):
    image = train_set.image_dic[train_set.id_list[int(idx)]]
    image_size = train_set.image_dic[train_set.id_list[int(idx)]].size

    model, criterion = build()
    model.load_state_dict(torch.load(f'neural_network_stuff/models/{model_name}'))
    

    item_img, item_label = train_set.__getitem__(int(idx))
    item_label['data'], item_label['classes'] = item_label['data'].unsqueeze(0), item_label['classes'].unsqueeze(0)
    
    output = model(item_img.unsqueeze(0))
    points, degrees = [(int(e[0]),int(e[1])) for e in output['data'][0]], [int(e[2]) for e in output['data'][0]]
    degree_lines = get_degree_lines(points, degrees)
    draw_stuff_on_image_and_save(image,points,degree_lines)

    loss = criterion(output,item_label)
    #print(points)
    #print(degrees)
    #print(loss)