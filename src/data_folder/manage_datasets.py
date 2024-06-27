import torch
from src.neural_network_stuff.custome_dataset import CustomImageDataset
from src.data_folder.create_data_folder import create_valTrain_folder
from src.data_folder.get_system_image.get_data import clean_folder

def create_datasets(dataset_name):
    training_set= CustomImageDataset('src/data_folder/test_dataloader/train')
    val_set = CustomImageDataset('src/data_folder/test_dataloader/val')

    torch.save(training_set, f'src/data_folder/datasets/{dataset_name}_train.pt')
    torch.save(val_set, f'src/data_folder/datasets/{dataset_name}_val.pt')

def load_datasets(dataset_name):
    training_set = torch.load(f'src/data_folder/datasets/{dataset_name}_train.pt')
    val_set = torch.load(f'src/data_folder/datasets/{dataset_name}_val.pt')
    return training_set, val_set

def add_data_to_dataset(dataset_name, path_train='src/data_folder/test_dataloader/train', path_val='src/data_folder/test_dataloader/val'):
    training_set, val_set = load_datasets(dataset_name)
    print(f'Length train before: {training_set.__len__()}')
    
    training_set.add_new_data(path_train)
    val_set.add_new_data(path_val)
    
    print('--------: Added Data to Datasets :--------')
    print(f'Length train after: {training_set.__len__()}')

    torch.save(training_set, f'src/data_folder/datasets/{dataset_name}_train.pt')
    torch.save(val_set, f'src/data_folder/datasets/{dataset_name}_val.pt')

def loop_iteration_for_datasets(dataset_name,iteration_length,n=100, randomize=True):

    for _ in range(iteration_length):
        clean_folder('src/data_folder/test_dataloader/train')
        clean_folder('src/data_folder/test_dataloader/val')
        create_valTrain_folder('src/data_folder/test_dataloader',n=n, randomize=randomize)
        add_data_to_dataset(dataset_name)
        

def add_to_datasets(d_1,d_2):
    d_1.merge_dataset(d_2)   
    return d_1 

def clear_label_files():
    with open('src/data_folder/test_dataloader/train/label.txt','w') as label_file:
        label_file.write('')
    with open('src/data_folder/test_dataloader/val/label.txt','w') as label_file:
        label_file.write('')






