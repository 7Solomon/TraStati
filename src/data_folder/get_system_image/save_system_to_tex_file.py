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

    print(f'tr:{type,rotation}')
    return type, rotation


def writeALager(n_i,position_x,position_y, type, rotation, i, j): 
    global positionABCMap
   
    base_line = rf'\point{{{intToABC[n_i]}}}{{{position_x}}}{{{position_y}}}'
   
    if type != 0:
         support_line = rf'\support{{{type}}}{{{intToABC[n_i]}}}{rotation}'
    else:
        support_line = rf'\hinge{{1}}{{{intToABC[n_i]}}}'
   
    
    tex_data = f"{base_line}\n{support_line}\n"

    positionABCMap[(i,j)] = intToABC[n_i] 
    return tex_data

def connect_lager(connector_list, lager_liste, positionABCMap_passed):
    connect_tex_data = ''
    allready_connected = []

    for balken in connector_list:
        # Look for the lager that are conencted
        first_lager, second_lager = positionABCMap_passed[balken[0]], positionABCMap_passed[balken[1]]

        # Dont take doubles
        if (first_lager,second_lager) in allready_connected or (second_lager,first_lager) in allready_connected:
            continue
        allready_connected.append((first_lager,second_lager))

        # Write Beam to Tex
        line = '\\beam' +  '{' + '4' + '}' +  '{' + first_lager + '}' + '{' + second_lager + '}' + '\n'
        connect_tex_data += line
    
    return connect_tex_data






def loopSystem():
    point_data, lager_tex_data = [], []
    fw = create_fachwerk()
    
    grid, connector_list = fw['data'], fw['staebe']

    # Für die Randomization der lengths
    lager_liste = get_lengths(grid)

    # Loop über Grid um Tex Data zu bekommen
    for n_i, lager in enumerate(lager_liste):
        (i,j), (x,y) = lager['index'], lager['koordinaten']
        type, rotation = get_type_and_rotation(grid,i,j)
    
        point_data.append([(x,y),type,rotation])
        lager_tex_data.append(writeALager(n_i,x,y,type, rotation,i,j))
    
    # Get the connection balken between the points
    connect_tex_data = connect_lager(connector_list, lager_liste, positionABCMap)
    
    
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



