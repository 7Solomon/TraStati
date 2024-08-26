import random
import argparse
import os
import string

from src.data_folder.get_system_image.grid import generate_a_connected_grid, get_lengths, create_fachwerk
import src.configure as configure 
positionABCMap = {}
intToABC = dict(enumerate(string.ascii_lowercase))


### Can be added to document_start to remove debug output
#data += '\\batchmode'
#data += '\n'
document_start = """
\\documentclass[12pt,letterpaper]{article}
\n
\\usepackage{styles/tikz}
\n
\\usepackage{styles/stanli}
\n
\\begin{document}
\n
\\begin{figure}
\n
\\centering
\n
\\begin{tikzpicture}
\n
\\scaling{.1}
\n
"""

null_punkt = """
\\fill[red] (0, 0) circle (1pt);
\n
\\end{tikzpicture}
\n
\\end{figure}
\n
\\end{document}
\n
"""
    

def get_type_and_rotation(grid,i,j):
    arround_list = []
    switch = [[i+1,j],[i,j+1],[i-1,j],[i,j-1]]
    for id, element in enumerate(switch):
        if 0 <= element[0] < len(grid) and 0 <= element[1] < len(grid[0]):
            if grid[element[0]][element[1]] == 1:
                arround_list.append(id)
    if len(arround_list) >= 3:
        type = 0
    else:
        type = random.randint(1,4)


    if j == 0 and not j == len(grid[0])-1:
        rotation = 270
    elif i == 0:
        rotation = 180
    elif j == len(grid[0])-1:
        rotation = 90
    
    
    
    # Rotation wegen nachbar
    elif not arround_list.__contains__(0):
        rotation =  270

    else:
        rotation = 0

    #print(f'tr:{type,rotation}')
    return type, rotation


def write_lager(data:dict) -> str: 
    """
    returns: TeX data of a lager as str
    """

    # Get Out of data
    lager_number = data['index']
    position = data['koordinaten']
    type = data['type']
    rotation = data['rotation']

    # Create Base Fo lager
    base_line = rf'\point{{{intToABC[lager_number]}}}{{{position[0]}}}{{{position[1]}}}'
   
   # Define auflager for type
    if type != 0:
        # Type per number
        support_line = rf'\support{{{type}}}{{{intToABC[lager_number]}}}{rotation}'
    else:
        # Gelenk 
        support_line = rf'\hinge{{1}}{{{intToABC[lager_number]}}}'
   
   # Combine 
    return f"{base_line}\n{support_line}\n"

def connect_lager(staebe_list:list, point_data:dict) -> str:
    connect_tex_data = ''
    #for i,j in point_data.items():
    #    print(i, j)
    for staeb in staebe_list:
        # Get index of Lagers out of data
        first_lager_index, second_lager_index = point_data[staeb[0]]['index'], point_data[staeb[1]]['index']

        # Write Beam to Tex
        line = '\\beam' +  '{' + '4' + '}' +  '{' + intToABC[first_lager_index] + '}' + '{' + intToABC[second_lager_index] + '}' + '\n'
        connect_tex_data += line
    return connect_tex_data



def loopSystem():
    point_data, lager_tex_data = {}, []

    # Get System    
    fw = create_fachwerk()
    grid, connector_list = fw['data'], fw['staebe']

    #print(grid)
    #print(connector_list)

    # Für die Randomization der lengths
    ### PROBLEM
    lager_liste = get_lengths(grid)
    """print(grid)
    for ((i,j),(n,m)) in connector_list:
        print(f'{(i,j)}: {grid[i,j]}')
        print(f'{(n,m)}: {grid[n,m]}')
        print('---')
    print(lager_liste)"""



    # Loop über Grid um Tex Data zu bekommen
    for n_i, lager in enumerate(lager_liste):
        (i,j), (x,y) = lager['index'], lager['koordinaten']
        type, rotation = get_type_and_rotation(grid,i,j)
    
        point_data[(i,j)] = {   
                                'index': n_i,
                                'koordinaten': (x,y),
                                'type': type,
                                'rotation': rotation
                            }
    
    # Write Tex data
    for data in point_data.values(): 
        tex_data_of_lager = write_lager(data)
        lager_tex_data.append(tex_data_of_lager)
    
    # Get the connection balken between the points
    connect_tex_data = connect_lager(connector_list, point_data)
    
    
    label = {
        'points':point_data,
        'connections': connector_list
    }
    tex_data =  {
        'points' : "".join(lager_tex_data),
        'connections': connect_tex_data
    }
    return label, tex_data



def getSystemAndSave():
    """
    returns:
    """
    global document_start, null_punkt
    tex_output_path = os.path.join("src","data_folder", "get_system_image", "data.tex")

    # ensures the file exists, and is empty
    open(tex_output_path, "w").close()

    # Get the data 
    label, tex_data = loopSystem()
    # Combines Data for Tex file 
    data = document_start + tex_data['points'] + tex_data['connections'] +  null_punkt  

    #print(data)
    # Save data to Tex file
    f = open(tex_output_path, "w")
    f.write(data)
    f.close()
    

    return label['points']



