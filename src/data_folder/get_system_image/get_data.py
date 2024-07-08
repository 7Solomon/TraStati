import os

from src.data_folder.get_system_image.convert_sytem_as_png import convert_system

def getData():
    img, label = convert_system()
    return img, label


if __name__ == '__main__':
    #clean_folder()
    getData(10)