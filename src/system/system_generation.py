from src.system.connection_map import *
from src. system.length_of_system import *

from src.functions import ask_for_dataset, load_datasets

def generate_system():
    dataset_name = ask_for_dataset()
    train_set, val_set = load_datasets(dataset_name)

    debug = get_standart_length_of_system(train_set)


    #connection_map()