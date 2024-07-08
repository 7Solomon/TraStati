import torch

import argparse, os
from src.neural_network_stuff.custome_DETR.detr import build
from src.neural_network_stuff.train import train_net
from src.data_folder.manage_datasets import load_datasets, add_to_datasets, clear_label_files, create_random_image
from src.visualize.visualize_dataset import load_dataset_and_ask_for_idx
from src.visualize.visualize_output import visualize_output

from src.neural_network_stuff.custome_dataset import CustomImageDataset


def ask_for_dataset():
    if not os.path.exists('src/data_folder/datasets/'):
        os.mkdir('src/data_folder/datasets/')

    datasets = os.listdir('src/data_folder/datasets')
    
    print('---------Datensätze---------')
    print('----------------------------')
    for i, dataset in enumerate(datasets):
        print(f'{i}: {dataset}')
    
    print('-----')
    print(f'{len(datasets)}: Willst du ein neues? ')
    
    print('----------------------------')
    print('----------------------------')
    idx_set = input('Welches datasets willst du? ') 


    # erstelle neues Dataset
    if idx_set == str(len(datasets)):
       dataset_name = input('Wie willst du das neue Dataset nennen? ')
       dataset = CustomImageDataset()
       dataset.save_to_file(dataset_name)
       return dataset_name


    # Get Dataset with id
    try:
        idx_set = int(idx_set)
        return datasets[idx_set]
    except:
        print('Du hast kein Int angegeben, oder er war auserhalb des Bereichs')
        ask_for_dataset()
    

def ask_for_model(new_create_bool:  bool = False):
    if not os.path.exists('src/neural_network_stuff/models/'):
        os.mkdir('src/neural_network_stuff/models/')

    models = os.listdir('src/neural_network_stuff/models/')
    
    print('---------Modelle---------')
    print('----------------------------')
    for i, model in enumerate(models):
        print(f'{i}: {model}')

    print('-----')
    print(f'{len(models)}: Willst du ein neues')

    print('----------------------------')
    print('----------------------------')

    idx_modell = input('Welches Modell willst du? ')

    if idx_modell == str(len(models)):
        model_name = input('Wie willst du das neue Modell nennen? ')
        return model_name

    
    try:
        idx_modell = int(idx_modell)
        return models[idx_modell]
    except:
        print('Du hast kein Int angegeben, oder er war auserhalb des Bereichs')
        ask_for_model()
    



def look_trough_dataset():
    """
    Loads dataset and displays the image with the ask idx
    """
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
    name = ask_for_dataset()
    print('---------')
    try:
        num_img = int(input('Anzahl an Bildern: '))
        print('Mit Randomized Systems!')  
    except:
        print('Du hast kein Int angegeben')
        #data()
    dataset = CustomImageDataset()

    for i in range(num_img):
        img, id, label = create_random_image()
        dataset.add_new_img(img, id, label)

        if i % 10 == 0 or i == 0:
            dataset.save_to_file(name)


    

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

