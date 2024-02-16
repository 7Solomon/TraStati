import torch

import argparse, os
from neural_network_stuff.custome_DETR.detr import build
from neural_network_stuff.train import train_net
from data_folder.create_data_folder import create_valTrain_folder
from data_folder.manage_datasets import loop_iteration_for_datasets, create_datasets, load_datasets, add_to_datasets
from visualize.visualize_dataset import load_dataset_and_ask_for_idx
from visualize.visualize_output import visualize_output


def create_folders():
    create_valTrain_folder('data_folder/test_dataloader')    # Random und n können hier rein


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
    

    



def look_trough_dataset():
    datasets = list(set(["_".join(e.split('_')[:-1]) for e in os.listdir('data_folder/datasets/')]))

    if len(datasets) != 0:        # Wenn es Datasets gibt
        # Auflisten der Datensätze
        for i, dataset in enumerate(datasets):
            print(f'{i}: {dataset}')
        print('---------')
        idx_set = input('What datasets do you want? ')

        dataset_name = datasets[int(idx_set)]
        load_dataset_and_ask_for_idx(dataset_name)
    else:
        print('Keine Datasets vorhanden')
    
def test_and_visualize_model():
    models = os.listdir('neural_network_stuff/models/')
    datasets =  list(set(["_".join(e.split('_')[:-1]) for e in os.listdir('data_folder/datasets/')]))
    
    # Auflisten der Datensätze
    if len(datasets) != 0:                  # Wenn es Datasets gibt
        for i, dataset in enumerate(datasets):
            print(f'{i}: {dataset}')
        print('---------')
        idx_set = input('What datasets do you want? ')
        dataset_name = datasets[int(idx_set)]


        # Laden des Datensatzes
        train_set, val_set = load_datasets(dataset_name)
        print('---------')
        print(f'Anzahl an Images:{train_set.__len__()}')


    else:
        print('Keine Datenätze vorhanden')

    
    # Auflisten der Modelle
    if len(models) != 0:
        for i, model in enumerate(models):
            print(f'{i}: {model}')
        print('---------')
        idx_modell = input('What model do you want? ')
        model_name = models[int(idx_modell)]

        go_loop = True
        while go_loop:
            idx = input('Welches Bild willst du Checken? ')
            if idx == 'cap' or idx == 'stop' or idx == 'halt':
                go_loop = False
            else:
                visualize_output(train_set, model_name, idx)
    else:
        print('Keine Modelle vorhanden')
        
    
    



    


def data():
    datasets = [*list(set(["_".join(e.split('_')[:-1]) for e in os.listdir('data_folder/datasets/')])), 'Willst du ein neues Dataset?']

    for i, dataset in enumerate(datasets):
        print(f'{i}: {dataset}')
    print('---------')
    idx_set = input('Zu welchem willst adden?')
    try:
        if int(idx_set) == len(datasets)-1:
            name = input('Welchen namen willst du für Datenset? ')
            create_datasets(name)
        else:
            name = datasets[int(idx_set)]

        num_img = int(input('Anzahl an Bildern pro Loop: '))
        num_loop = int(input('Anzahl an loops: '))
        print('Mit Randomized Systems')  
    except:
        print ('Du hast kein Int angegeben')
    loop_iteration_for_datasets(name,num_loop,num_img,randomize=True)

    

def train():
    models = [*os.listdir('neural_network_stuff/models/'), 'Willst du ein neues generieren?']
    datasets =  list(set(["_".join(e.split('_')[:-1]) for e in os.listdir('data_folder/datasets/')]))
    
    # Auflisten der Datensätze
    if len(datasets) != 0:                  # Wenn es Datasets gibt
        for i, dataset in enumerate(datasets):
            print(f'{i}: {dataset}')
        print('---------')
        idx_set = input('What datasets do you want? ')
        dataset_name = datasets[int(idx_set)]

        # Laden des Datensatzes
        train_set, val_set = load_datasets(dataset_name)
    else:                              
        print('Keine Datenätze vorhanden')
        return

    # Auflisten der Modelle
    for i, model in enumerate(models):
        print(f'{i}: {model}')
    print('---------')
    idx_modell = input('What model do you want? ')
    model_name = models[int(idx_modell)]

    print('---------')
    num_eppochs = input('Anzahl der Epochen? ')

    
    image_size = train_set.image_dic[train_set.id_list[0]].size
    model, criterion = build()
    if idx_modell == str(len(models)-1):
        model_save_name = input('Wie willst du das neue Modell speichern?')
        train_net(model, criterion, train_set, val_set, num_epochs=int(num_eppochs), load_model=None, save_as=f'neural_network_stuff/models/{model_name}')
    else:
        save = input('Willst du das Modell Überschreiben? [Y/n]: ').strip().lower() or 'y'
        if save == 'y':
            train_net(model, criterion, train_set, val_set, num_epochs=int(num_eppochs), load_model=f'neural_network_stuff/models/{model_name}', save_as=f'neural_network_stuff/models/{model_name}')
        elif save == 'n':
            model_save_name = input('Wie willst du das neue Modell speichern?')
            train_net(model, criterion, train_set, val_set, num_epochs=int(num_eppochs), load_model=f'neural_network_stuff/models/{model_name}', save_as=f'neural_network_stuff/models/{model_name}')
       
        else:
            print('Not Valid')


    

if __name__ == "__main__":
    print('Du bist in der falschen Datei')

