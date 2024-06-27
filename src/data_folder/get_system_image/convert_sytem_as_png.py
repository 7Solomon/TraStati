import subprocess
import os

from latex import build_pdf

from src.data_folder.get_system_image.save_system_to_tex_file import getSystemAndSave

def convertSystem(id,randomize=False, output_folder='src/data_folder/get_system_image/img'):

    label_list = getSystemAndSave(randomize)

    output_path = os.path.join(output_folder, f'{id}.jpg')
    subprocess.run(['pdflatex', '-output-directory=src/data_folder/get_system_image', 'src/data_folder/get_system_image/data.tex'])
    # PNG-Bild in JPG umwandeln
    subprocess.run(['convert', '-density', '300', 'src/data_folder/get_system_image/data.pdf', '-quality', '90', output_path])

    # Optional: Aufräumen, entferne temporäre Dateien
    if os.path.exists('src/data_folder/get_system_image/data.tex'):
      os.remove('src/data_folder/get_system_image/data.tex')
    if os.path.exists('src/data_folder/get_system_image/data.aux'):
      os.remove('src/data_folder/get_system_image/data.aux')
    if os.path.exists('src/data_folder/get_system_image/data.log'):
      os.remove('src/data_folder/get_system_image/data.log')
    if os.path.exists('src/data_folder/get_system_image/data.pdf'):
      os.remove('src/data_folder/get_system_image/data.pdf')

    write_label_file(label_list, id, output_folder)    


def write_label_file(label_list, id , output_folder):
    output_file = os.path.join(output_folder, 'label.txt')
    
    with open(output_file, 'a') as file:
        file.write(f'{id}:')
        file.write('|'.join(str(item) for item in label_list))
        file.write('\n')
        
if __name__ == '__main__':
    convertSystem('0')

    