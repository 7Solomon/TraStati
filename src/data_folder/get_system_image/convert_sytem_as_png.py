import subprocess
import os
from latex import build_pdf
from src.data_folder.get_system_image.save_system_to_tex_file import getSystemAndSave

def convert_system(id, randomize=False):
    label_list = getSystemAndSave(randomize)
    output_folder = os.path.join("src", "data_folder", "get_system_image", "img")
    output_path = os.path.join(output_folder, f'{id}.jpg')
    
    # Run pdflatex
    subprocess.run(['pdflatex', '-output-directory=src/data_folder/get_system_image', 'src/data_folder/get_system_image/data.tex'])
    
    # Convert PDF to JPG
    pdf_to_jpg(output_path)
    
    # Clean up temporary files
    cleanup_temp_files()
    
    # Write label file  
    write_label_file(label_list, id, output_folder)

def pdf_to_jpg(output_path):
    pdf_path = os.path.join('src', 'data_folder', 'get_system_image', 'data.pdf')
    if os.name == 'nt':  # Windows
        command = ['magick', 'convert', '-density', '300', pdf_path, '-quality', '90', output_path]
    else:  # Unix/Linux
        command = ['convert', '-density', '300', pdf_path, '-quality', '90', output_path]
    
    subprocess.run(command)

def cleanup_temp_files():
    temp_files = ['data.tex', 'data.aux', 'data.log', 'data.pdf']
    for file in temp_files:
        file_path = os.path.join('src' ,'data_folder', 'get_system_image', file)
        if os.path.exists(file_path):
            os.remove(file_path)

def write_label_file(label_list, id, output_folder):
    output_file = os.path.join(output_folder, 'label.txt')
    with open(output_file, 'a') as file:
        file.write(f'{id}:')
        file.write('|'.join(str(item) for item in label_list))
        file.write('\n')

if __name__ == '__main__':
    convert_system('0')