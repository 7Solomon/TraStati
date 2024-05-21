from funcsigs import *

import torch

import argparse, os
from neural_network_stuff.custome_DETR.detr import build
from neural_network_stuff.train import train_net
from data_folder.create_data_folder import create_valTrain_folder
from data_folder.manage_datasets import create_datasets, load_datasets, add_to_datasets

from functions import *

def data_creation_for_loop(base_train=None, base_val=None):
    if base_train == None and base_val == None:
        create_valTrain_folder('data_folder/test_dataloader',n=10, randomize=True)
        create_datasets('base_dataset')
        base_train, base_val = load_datasets('base_dataset')
    

    create_valTrain_folder('data_folder/test_dataloader',n=40, randomize=True)
    create_datasets('extended_dataset')

    extended_train, extended_val = load_datasets('extended_dataset')
    base_train = add_to_datasets(base_train, extended_train)
    base_val = add_to_datasets(base_val, extended_val)
    return base_train, base_val

def train_model_on_loop_dataset(train, val, modell=None, criterion=None):
    assert modell == None and criterion == None or modell != None and criterion != None, 'Also ka aber das sollte halt schon sein so'
    if modell == None and criterion == None:
        modell, criterion = build()
    
    model_state = train_net(modell, criterion, train, val, num_epochs=10, load_model=None, save_as='endless_loop_model')
    modell.load_state_dict(model_state)
    return modell
    
def endless_loop():
    print('Wähle das Base Dataset')
    name = ask_for_dataset(new_create_bool=False)

    for iteration in range(1,50):
        new_name = f'big_loop_{iteration}'
        
        shutil.copytree(f'data_folder/datasets/{name}', f'data_folder/datasets/{new_name}')
        if iteration > 2:
            os.rmdir(f'data_folder/datasets/big_loop_{iteration - 2}')

        name = new_name

        num_img = 10
        num_loop = 10

        loop_iteration_for_datasets(name,num_loop,num_img,randomize=True)



def loop_for_optimal_model():
    print('noch nicht implementiert')
    name_of_dataset = ask_for_dataset(new_create_bool=False)
    train, val = load_datasets(name_of_dataset)

    model, criterion = build()
    #model = train_net(model, criterion, train, val, num_epochs=20, save_as=f'neural_network_stuff/iterated_model_parameter/{model_name}', load_model=None)
    return model