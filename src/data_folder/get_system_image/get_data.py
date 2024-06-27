import random,os

from src.data_folder.get_system_image.convert_sytem_as_png import convertSystem

def getData(num,randomize=False):
    for i in range(num):
        random_string = ''.join(random.choices('0123456789abcdef', k=12))
        convertSystem(random_string,randomize)


def clean_folder(path = 'src/data_folder/get_system_image/img'):
    open(f'{path}/label.txt', 'w').close()

    list_dir = os.listdir(path)
    for element_str in list_dir:
        if element_str != 'label.txt':
            os.remove(f'{path}/{element_str}')

if __name__ == '__main__':
    #clean_folder()
    getData(10)