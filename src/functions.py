import torch

import argparse, os
from src.neural_network_stuff.custome_DETR.detr import build
from src.neural_network_stuff.train import train_net
from src.data_folder.create_data_folder import create_valTrain_folder
from src.data_folder.manage_datasets import loop_iteration_for_datasets, create_datasets, load_datasets, add_to_datasets, clear_label_files
from src.visualize.visualize_dataset import load_dataset_and_ask_for_idx
from src.visualize.visualize_output import visualize_output


def create_folders():
    create_valTrain_folder('src/data_folder/test_dataloader')    # Random und n können hier rein



def ask_for_dataset(new_create_bool: bool = False):
    if os.path.exists('src/data_folder/datasets/'):
        datasets = list(set(["_".join(e.split('_')[:-1]) for e in os.listdir('src/data_folder/datasets/')]))
    else:
        os.mkdir('src/data_folder/datasets/')
        datasets = list(set(["_".join(e.split('_')[:-1]) for e in os.listdir('src/data_folder/datasets/')]))
    print('---------Datensätze---------')
    for i, dataset in enumerate(datasets):
        print(f'{i}: {dataset}')
    if new_create_bool:
        print(f'{len(datasets)}: Willst du ein neues Dataset?')
    print('----------------------------')
    idx_set = input('What datasets do you want? ')
    
    if idx_set == str(len(datasets)) and new_create_bool:
        name = input('Welchen namen willst du für Datenset? ')
        
        # Experimanetal BE carefull, maybe not good this is
        clear_label_files()
        create_datasets(name)
        return name
    else:
        # Schauen ob int
        try:
            idx_set = int(idx_set)
            return datasets[idx_set]
        except:
            print('Du hast kein Int angegeben, oder er war auserhalb des Bereichs')
            ask_for_dataset()

def ask_for_model(new_create_bool:  bool = False):
    if os.path.exists('src/neural_network_stuff/models/'):
        models = os.listdir('src/neural_network_stuff/models/')
    else:
        os.mkdir('src/neural_network_stuff/models/')
        models = os.listdir('src/neural_network_stuff/models/')
    print('---------Modelle---------')
    for i, model in enumerate(models):
        print(f'{i}: {model}')
    if new_create_bool:
        print(f'{len(models)}: Willst du ein neues Modell?')
    print('-------------------------')

    idx_modell = input('Welches Modell willst du? ')
    if idx_modell == str(len(models)) and new_create_bool:
        model_name = input('Wie willst du das neue Modell nennen? ')
        return model_name, True

    else:
        try:
            idx_modell = int(idx_modell)
            return models[idx_modell], False
        except:
            print('Du hast kein Int angegeben, oder er war auserhalb des Bereichs')
            ask_for_model()
    



def look_trough_dataset():

    dataset_name = ask_for_dataset()
    load_dataset_and_ask_for_idx(dataset_name)
    
    

def test_and_visualize_model():
    dataset_name = ask_for_dataset(new_create_bool=False)
    model_name, did_create_new_model = ask_for_model(new_create_bool=False)

    # Laden des Datensatzes
    train_set, val_set = load_datasets(dataset_name)
    print('---------')
    print(f'Anzahl an Images:{train_set.__len__()}')
 

    go_loop = True
    while go_loop:
        idx = input('Welches Bild willst du Checken? ')

        try :
            idx = int(idx)
        except:
            if idx == 'cap' or idx == 'stop' or idx == 'halt':
                go_loop = False
            else:
                print('thats not an int, und kein richtiger stop command')
            continue
        visualize_output(train_set, model_name, idx)


                    

def data():
    name = ask_for_dataset(new_create_bool=True)
    print('---------')
    try:
        num_img = int(input('Anzahl an Bildern pro Loop: '))
        num_loop = int(input('Anzahl an loops: '))
        print('Mit Randomized Systems')  
    except:
        print('Du hast kein Int angegeben')
        return

    loop_iteration_for_datasets(name,num_loop,num_img,randomize=True)

    

def train():
    dataset_name = ask_for_dataset(new_create_bool=False)
    train_set, val_set = load_datasets(dataset_name)


    model_name, did_create_new_model = ask_for_model(new_create_bool=True)
    print('---------')
    try:
        num_eppochs = int(input('Anzahl der Epochen? '))
    except:
        print('Du hast kein Int angegeben')
        return

    model, criterion = build()

    if not did_create_new_model:
        save = input('Willst du das Modell Überschreiben? [Y/n]: ').strip().lower() or 'y'
        if save == 'y':
            train_net(model, criterion, train_set, val_set, num_epochs=int(num_eppochs), load_model=f'src/neural_network_stuff/models/{model_name}', save_as=f'src/neural_network_stuff/models/{model_name}')
        elif save == 'n':
            model_save_name = input('Wie willst du das neue Modell speichern?')
            train_net(model, criterion, train_set, val_set, num_epochs=int(num_eppochs), load_model=f'src/neural_network_stuff/models/{model_name}', save_as=f'src/neural_network_stuff/models/{model_save_name}')
    else:
        # Falls ein neues Modell erstellt wurde
        train_net(model, criterion, train_set, val_set, num_epochs=int(num_eppochs), save_as=f'src/neural_network_stuff/models/{model_name}')

if __name__ == "__main__":
    print('Du bist in der falschen Datei')

