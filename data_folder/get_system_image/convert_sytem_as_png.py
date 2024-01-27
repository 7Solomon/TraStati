import subprocess
import os

from latex import build_pdf

from data_folder.get_system_image.save_system_to_tex_file import getSystemAndSave

def convertSystem(id,randomize=False, output_folder='data_folder/get_system_image/img'):

    label_list = getSystemAndSave(randomize)

    output_path = os.path.join(output_folder, f'{id}.jpg')
    subprocess.run(['pdflatex', 'data.tex'])
    # PNG-Bild in JPG umwandeln
    subprocess.run(['convert', '-density', '300', 'data.pdf', '-quality', '90', output_path])

    # Optional: Aufräumen, entferne temporäre Dateien
    os.remove('data.tex')
    os.remove('data.aux')
    os.remove('data.log')
    os.remove('data.pdf')

    write_label_file(label_list, id, output_folder)    


def write_label_file(label_list, id , output_folder):
    output_file = os.path.join(output_folder, 'label.txt')
    
    with open(output_file, 'a') as file:
        file.write(f'{id}:')
        file.write('|'.join(str(item) for item in label_list))
        file.write('\n')
        
if __name__ == '__main__':
    convertSystem('0')

    