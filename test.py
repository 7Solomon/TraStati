import argparse
from data_folder.manage_datasets import create_datasets, loop_iteration_for_datasets
from data_folder.manage_datasets import load_datasets    
from data_folder.create_data_folder import create_valTrain_folder          

from visualize.draw_graph import draw_stuff_on_image_and_save, get_points_from_label, get_degree_lines,draw_some_item
from visualize.visualize_dataset import draw_elemet_of_dataset, load_dataset_and_ask_for_idx, some_testS
from visualize.visualize_output import visualize_output
from neural_network_stuff.test_detr_model import testDetr
from neural_network_stuff.custome_loss import CustomeClassLoss
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust the size as needed
    transforms.ToTensor(),
])

if __name__ == '__main__':
    #load_dataset_and_ask_for_idx('v5_big_test')

    t,v = load_datasets('littl_and_littler_dataset_test')
    img, label = t.__getitem__(0)

    model = testDetr()
    criterion = CustomeClassLoss()
    model.load_state_dict(torch.load('neural_network_stuff/models/v_2.pth'))

    model.eval()
    output = model(img.unsqueeze(0))
    loss = criterion(output, label)



    #with Image.open('data_folder/test_dataloader/train/708b50143867.jpg') as img:
    #    img_in = transform(img)
    #    img_in = img_in.unsqueeze(0) # Add batch dimension
    #    output = model(img_in)
    #    #visualize_output(img,output)
    #    loss = criterion(output, output['data'])


    #parser = argparse.ArgumentParser(description='A script with a --number argument.')
    #parser.add_argument('--number', type=int, help='An integer argument.')
    #args = parser.parse_args()

    #if args.number is not None:
    #    number = args.number
    #else:
    #    number = 0

    #draw_some_item(number)
    #t,v = load_datasets('v5_big_test')
    #t.display_data(number)
   