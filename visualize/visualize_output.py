import torch

from data_folder.manage_datasets import load_datasets
from neural_network_stuff.custome_DETR.misc_stuff import nested_tensor_from_tensor_list
from visualize.draw_graph import draw_stuff_on_image_and_save, get_degree_lines
from neural_network_stuff.custome_DETR.detr import build

def visualize_output(train_set, model_name, idx):
    image = train_set.image_dic[train_set.id_list[int(idx)]]
    image_size = train_set.image_dic[train_set.id_list[int(idx)]].size

    model, criterion = build()
    model.load_state_dict(torch.load(f'neural_network_stuff/models/{model_name}'))
    

    # Muss besser gelöst werden
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    item_img, item_label = train_set.__getitem__(int(idx))
    samples = nested_tensor_from_tensor_list([item_img])          # Das eine Batch kommt []
    samples = samples.to(device)
    
    output = model(samples)

    #print(output['output_center_degree_points'][0][0])
    #print(output['output_center_degree_points'][0][0])
    points = [(e[0].item(),e[1].item()) for e in output['output_center_degree_points'][0]]
    degrees = [e[2].item() for e in output['output_center_degree_points'][0]]

    points = [(int(840*x), int(960*y)) for x,y in points]
    degrees = [int(360/64*deg) for deg in degrees]
    #print(f'points: {points}')

    degree_lines = get_degree_lines(points, degrees)
    draw_stuff_on_image_and_save(image,points,degree_lines)

    target = [{k: v.to(device) for k, v in item_label.items()}]   # [] für Batch
    loss = criterion(output, target)
    #print(points)
    #print(degrees)
    print(f'loss: {loss}')