import os
from xml.dom import minidom
from unidades import UnidadErratica, UnidadPredecible
from pandas import DataFrame
import numpy as np
import math


def espacios_recorridos(mapa):
    recorridos = 0
    for x in mapa:
        for y in x:
            if(y == 1):
                recorridos += 1
    return recorridos



# se asume que todos los niveles seran ordenados secuencialmente por lo que solo se considera la cantidad
def load_levels():
    dir_path = './models/levels/'
    levels = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            levels.append(path)
    return len(levels)


def get_area_mapa(new_mapa):
    area = 0
    fila, columna = new_mapa.shape
    posX = 0
    posY = 0
    for x in new_mapa:
        for y in x:
            if (y == 0):
                north = False
                east = False
                west = False
                south = False
                posObjetoX = posX
                posObjetoY = posY
                while(posObjetoX!=0 and new_mapa[posObjetoX][posObjetoY] != -8):
                    posObjetoX -= 1
                    if(new_mapa[posObjetoX][posObjetoY] == -8):
                        north = True
                posObjetoX = posX
                posObjetoY = posY
                while(posObjetoX!=fila-1 and new_mapa[posObjetoX][posObjetoY] != -8):
                    posObjetoX += 1
                    if(new_mapa[posObjetoX][posObjetoY] == -8):
                        south = True
                posObjetoX = posX
                posObjetoY = posY
                while(posObjetoY!=0 and new_mapa[posObjetoX][posObjetoY] != -8):
                    posObjetoY -= 1
                    if(new_mapa[posObjetoX][posObjetoY] == -8):
                        west = True
                posObjetoX = posX
                posObjetoY = posY
                while(posObjetoY!=columna-1 and new_mapa[posObjetoX][posObjetoY] != -8):
                    posObjetoY += 1
                    if(new_mapa[posObjetoX][posObjetoY] == -8):
                        east = True
                if(north == True and east == True and west == True and south == True):
                    area += 1
            posY += 1
        posX += 1
        posY = 0
    return area

def get_new_map(map_path):
    new_mapa = np.zeros((15,20),dtype=int)
    # return copy.deepcopy(mapa_inicial)
    file = minidom.parse(map_path)
    muros = file.getElementsByTagName('geom')
    for muro in muros:
        if("side" in muro.attributes['name'].value or "hallway" in muro.attributes['name'].value):
            #print(muro.attributes['name'].value)
            size = muro.attributes['size'].value
            sizeX = math.trunc(float(size.split(" ")[0]))
            sizeY = math.trunc(float(size.split(" ")[1]))
            pos = muro.attributes['pos'].value
            x = float(pos.split(" ")[0])
            y = float(pos.split(" ")[1])
            if x >= 0 and y >= 0:
                # Primer Cuadrante
                fila = 3 - math.trunc(y)
                columna = 4 + math.trunc(x)
                if(sizeY>0):
                    for largo in range(sizeY):
                        new_mapa[fila+(largo+1),columna]=-8
                        new_mapa[fila-largo,columna]=-8
            elif x < 0 and y >= 0:
                #Segundo cuadrante
                fila = 3 - math.trunc(y) 
                columna = 3 + math.trunc(x)
                if(sizeY>0):
                    for largo in range(sizeY):
                        new_mapa[fila+(largo+1),columna]=-8
                        new_mapa[fila-largo,columna]=-8
            elif x < 0 and y < 0:
                # Tercer Cuadrante
                fila = 4 - math.trunc(y)
                columna = 3 + math.trunc(x)
                if(sizeY>0):
                    for largo in range(sizeY):
                        new_mapa[fila+largo,columna]=-8
                        new_mapa[fila-(largo+1),columna]=-8
            elif x >= 0 and y < 0:
                # Cuarto Cuadrante
                fila = 4 - math.trunc(y)
                columna = 4 + math.trunc(x)
                if(sizeY>0):
                    for largo in range(sizeY):
                        new_mapa[fila+largo,columna]=-8
                        new_mapa[fila-(largo+1),columna]=-8
            if(sizeX>0):
                for largo in range(sizeX):
                    new_mapa[fila,columna+largo]=-8
                    new_mapa[fila,columna-(largo+1)]=-8
        #print(muro.attributes['size'].value)
    #mostrar_mapa(new_mapa)
    return new_mapa

def mostrar_mapa(mapa, area_mapa):
    recorridos = espacios_recorridos(mapa)
    print(f'el area del mapa es {area_mapa}')
    print(DataFrame(mapa))
    print("Espacios Recorridos: ", espacios_recorridos(mapa))
    print("Porcentaje Recorrido: ", recorridos*100/area_mapa, "%")

def get_unidades_dict(map_path):
    unidades = []
    qpos = 14
    dom = minidom.parse(map_path)
    elements = dom.getElementsByTagName('body')
    for element in elements:
        pos = element.attributes['pos'].value.split()
        x = pos[0]
        y = pos[1]
        if "randomMovingObject" in element.attributes['name'].value:
            tipo= "Erratico"
        else:
            tipo = "Predecible"
        unidad = {"tipo": tipo, "x": x, "y": y, "qpos": qpos}
        unidades.append(unidad)
        qpos += 7

    return unidades
