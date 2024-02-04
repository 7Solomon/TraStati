import torch

import argparse, os
from neural_network_stuff.custome_loss import CustomeComLoss
from neural_network_stuff.train import train_net
from neural_network_stuff.test_detr_model import testDetr
from data_folder.create_data_folder import create_valTrain_folder
from data_folder.manage_datasets import loop_iteration_for_datasets, create_datasets, load_datasets
from visualize.visualize_dataset import load_dataset_and_ask_for_idx
from visualize.visualize_output import visualize_output


def create_folders():
    create_valTrain_folder('data_folder/test_dataloader')    # Random und n können hier rein




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
    model = testDetr(image_size=image_size)
    #print(idx_set,str(len(models)-1))
    if idx_set == str(len(models)-1):
        model_save_name = input('Wie willst du das neue Modell speichern?')
        train_net(model, train_set, val_set, num_epochs=int(num_eppochs), load_model=None, save_as=model_save_name)
    else:
        save = input('Willst du das Modell Überschreiben? [Y/n]: ').strip().lower() or 'y'
        if save == 'y':
            train_net(model, train_set, val_set, num_epochs=int(num_eppochs), load_model=f'neural_network_stuff/models/{model_name}', save_as=f'neural_network_stuff/models/{model_name}')
        elif save == 'n':
            model_save_name = input('Wie willst du das neue Modell speichern?')
            train_net(model, train_set, val_set, num_epochs=int(num_eppochs), load_model=f'neural_network_stuff/models/{model_name}', save_as=model_save_name)
       
        else:
            print('Not Valid')


    

if __name__ == "__main__":
    train()

