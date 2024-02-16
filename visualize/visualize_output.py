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

    #print(output['output_center_degree_points'][0][0])
    #print(output['output_center_degree_points'][0][0])
    points = [(e[0].item(),e[1].item()) for e in output['output_center_degree_points'][0]]
    degrees = [e[2].item() for e in output['output_center_degree_points'][0]]

    points = [(int(840*x), int(960*y)) for x,y in points]
    degrees = [int(360/64*deg) for deg in degrees]
    #print(f'points: {points}')

    degree_lines = get_degree_lines(points, degrees)
    draw_stuff_on_image_and_save(image,points,degree_lines)

    loss = criterion(output,item_label)
    #print(points)
    #print(degrees)
    #print(loss)