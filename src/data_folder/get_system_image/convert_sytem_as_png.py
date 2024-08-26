import subprocess
import os
import io

from PIL import Image
from latex import build_pdf

from src.data_folder.get_system_image.save_system_to_tex_file import getSystemAndSave

def convert_system():
    label_data = getSystemAndSave()

    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory=src/data_folder/get_system_image', 'src/data_folder/get_system_image/data.tex'], capture_output=True, text=True)
    #print(result.stdout)
    # Print stderr if there were any errors
    if result.stderr:
        print("Errors:", result.stderr)


    # Convert PDF to JPG
    image = pdf_to_pil_image()
    
    # Clean up temporary files
    ### Dees thorugh Win Error
    #cleanup_temp_files()


    return image, label_data

def pdf_to_pil_image():
    pdf_path = os.path.join('src', 'data_folder', 'get_system_image', 'data.pdf')
    if os.name == 'nt':  # Windows
        command = ['magick', 'convert', '-density', '300', pdf_path, '-quality', '90', 'jpg:-']
    else:  # Unix/Linux
        command = ['convert', '-density', '300', pdf_path, '-quality', '90', 'jpg:-']
    
    result = subprocess.run(command, capture_output=True, check=True)
    image_bytes = io.BytesIO(result.stdout)
    return Image.open(image_bytes)

def cleanup_temp_files():
    temp_files = ['data.aux', 'data.log']   #   'data.tex' 'data.pdf'
    for file in temp_files:
        file_path = os.path.join('src' ,'data_folder', 'get_system_image', file)
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    convert_system('0')