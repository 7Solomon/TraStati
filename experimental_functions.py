from funcsigs import *

import torch

import argparse, os
from neural_network_stuff.custome_DETR.detr import build
from neural_network_stuff.train import train_net
from data_folder.create_data_folder import create_valTrain_folder
from data_folder.manage_datasets import create_datasets, load_datasets, add_to_datasets


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
    # Einlesesn der Anzahl an Loops
    try:
        n = int(input('Wie lange willst du Loopen?'))
    except:
        print('Du hast kein Int angegeben')
        return
    
    # Erstellen des ersten Datensatzes
    base_train, base_val = data_creation_for_loop()
    model, criterion = build()
    # Laden des Endlosen Modells
    if torch.load('neural_network_stuff/models/endless_loop_model') == None:
        model = train_model_on_loop_dataset(base_train, base_val, modell=None, criterion=criterion)
    else:
        model.load_state_dict(torch.load('neural_network_stuff/models/endless_loop_model'))
        model = train_model_on_loop_dataset(base_train, base_val, modell=model, criterion=criterion)
    
    for _ in range(n):
        base_train, base_val = data_creation_for_loop(base_train, base_val)
        model = train_model_on_loop_dataset(base_train, base_val, modell=model, criterion=criterion)
        print('---------')
    
