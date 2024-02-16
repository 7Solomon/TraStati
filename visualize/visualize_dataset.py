from data_folder.manage_datasets import load_datasets
from visualize.draw_graph import get_degree_lines, draw_stuff_on_image_and_save
from torchvision import transforms
from PIL import Image

transform_reverse = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Resize((840, 960)),
    #transforms.ToPILImage(),  
])



def get_points(data):
    points, degrees = [], []
    for element in data:
        point = (int(element[0]),int(element[1]))
        degree = int(element[2])
        if point != (0,0):
            points.append(point)
            degrees.append(degree)

    degree_lines = get_degree_lines(points, degrees)
    return points,degree_lines

def draw_elemet_of_dataset(dataset,idx=0):
    item = dataset.__getitem__(idx)
    class_label, data_label =item[1]['classes'],item[1]['data']
    #img = inverse_transform(item[0])
    points,degree_lines = get_points(data_label)
    print(points,degree_lines)
    draw_stuff_on_image_and_save(item[0],points,degree_lines)
    

def load_dataset_and_ask_for_idx(name):
    repeat_q = True

    t,v = load_datasets(name) 
    print(f'Len : {t.__len__()-1}')    
    while repeat_q:
        idx = input('Idx: ')
        try:
            int(idx)
            t.display_data(int(idx))
        except :
            if idx == 'cap' or idx == 'stop' or idx == 'du ehrenloser':
                repeat_q = False
            else:
                print('thats not an int')
                continue
        
        


def some_testS():
    with Image.open('data_folder/noised_images/ae518e35590b.jpg')as img:
        print(img.size)