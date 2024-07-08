import random
import argparse
import os

from src.data_folder.get_system_image.grid import generate_a_connected_grid, get_lengths
import src.configure as configure 
data = ''
positionABCMap = {}
intToABC = {'0':'a', '1':'b', '2':'c', '3':'d', '4':'e', '5':'f', '6':'g', '7':'h', '8':'i', '9':'j', '10':'k', '11':'l', '12':'m', '13':'n', '14':'o', '15':'p', '16':'q', '17':'r', '18':'s', '19':'t', '20':'u', '21':'v', '22':'w', '23':'x', '24':'y', '25':'z'}

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


    return type, rotation

def loopSystem(randomize=False):


    label_list = []
    grid, connector_list = generate_a_connected_grid(3, 3, PROB=0.3)
    #print(connector_list)
    
    #VISUALIZE
    #print('ACHTUNG: visualisierung ist nicht ganz richtig')  jetzt ist es richtig
    #for line in grid:
    #    print(line)
 
    lager_liste = get_lengths(grid, randomize=randomize)
    
    for n_i, lager in enumerate(lager_liste):
        (i,j), (x,y) = lager[0], lager[1]
        type, rotation = get_type_and_rotation(grid,i,j)
        
        label_list.append([(x,y),type,rotation])


        writeALager(n_i,x,y,type, rotation,i,j)
    
    connect_lager(connector_list, lager_liste, positionABCMap)
    return label_list


def writeALager(n_i,position_x,position_y, type, rotation, i, j): 
    global data, positionABCMap
    
    
    line_1 = '\point' + '{' + intToABC[str(n_i)] + '}' + '{' + str(position_x) + '}' + '{' + str(position_y) + '}' + '\n'
    

    if type != 0:
        line_2 = '\support' + '{' + str(type)+ '}' + '{' + intToABC[str(n_i)] + '}' + '[' + str(rotation) + ']' + '\n'
    else:
        line_2 = '\hinge' + '{' + '1' + '}' + '{' + intToABC[str(n_i)] + '}' + '\n'
   
    
    data += line_1
    data += line_2

    positionABCMap[(i,j)] = intToABC[str(n_i)] 

def connect_lager(connector_list, lager_liste, positionABCMap_passed):
    global data
    allready_connected = []
    #print(lager_liste)
    #print(positionABCMap_passed)

    for balken in connector_list:
        first_lager, second_lager = positionABCMap_passed[balken[0]], positionABCMap_passed[balken[1]]
        #print((first_lager,second_lager))

        if (first_lager,second_lager) in allready_connected or (second_lager,first_lager) in allready_connected:
            continue

        allready_connected.append((first_lager,second_lager))

        line_1 = '\\beam' +  '{' + '4' + '}' +  '{' + first_lager + '}' + '{' + second_lager + '}' + '\n'
        data += line_1
        """
        (i,j), (x,y) = lager[0], lager[1]
        found_idx =  []
        for idx, sublist in enumerate(connector_list):
            if (i,j) in sublist:
                found_idx.append(idx)
        print(lager)
        print(found_idx)
        #print(f'{(i,j)}: {[connector_list[e] for e in found_idx]}')
        for element in found_idx:
            first_lager, secondLager = connector_list[element]
            
            buchstabe_1 = positionABCList_passed[[e[0] for e in positionABCList_passed].index(first_lager)][1]
            buchstabe_2 = positionABCList_passed[[e[0] for e in positionABCList_passed].index(secondLager)][1]
            print(f'{positionABCList_passed[[e[0] for e in positionABCList_passed].index(first_lager)][1]}: {[e[0] for e in positionABCList_passed].index(first_lager)}')
            print(f'{positionABCList_passed[[e[0] for e in positionABCList_passed].index(secondLager)][1]}: {[e[0] for e in positionABCList_passed].index(secondLager)}')
            print(f'{connector_list[element]}: {buchstabe_1}, {buchstabe_2}')

            if (buchstabe_1, buchstabe_2) in allready_connected or (buchstabe_2, buchstabe_1) in allready_connected:
                continue
            allready_connected.append((buchstabe_1, buchstabe_2))
            """
            #line_1 = '\\beam' +  '{' + '4' + '}' +  '{' + buchstabe_1 + '}' + '{' + buchstabe_2 + '}' + '\n'
            
            #data += line_1


def getSystemAndSave():
    global data
    tex_output_path = os.path.join("src","data_folder", "get_system_image", "data.tex")
    randomize = configure.randomize_images

    open(tex_output_path, "w").close()
    #data += '\\batchmode'
    #data += '\n'

    data += '\\documentclass[12pt,letterpaper]{article}'
    data += '\n'

    data += '\\usepackage{styles/tikz}'
    data += '\n'
    data += '\\usepackage{styles/stanli}'
    data += '\n'


    data += '\\begin{document}'
    data += '\n'
    data += '\\begin{figure}'
    data += '\n'
    data += '\\centering'
    data += '\n'
    data += '\\begin{tikzpicture}'
    data += '\n'
    data += '\\scaling{.1}'
    data += '\n'

    label_list = loopSystem(randomize)

    # Nullpunkt definieren
    data +=  '\\fill[red] (0, 0) circle (1pt);'
    data += '\n'


    data += '\\end{tikzpicture}'
    data += '\n'
    data += '\\end{figure}'
    data += '\n'
    data += '\\end{document}'
    data += '\n'
    
    f = open(tex_output_path, "w")

    f.write(data)
    f.close()
    
    data = ''
    positionABCList = ''

    return label_list





if __name__ == '__main__':
    label_list = getSystemAndSave()
    print(f'label_list: {label_list}')
   
