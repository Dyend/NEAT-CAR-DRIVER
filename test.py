import json
import math
from pandas import DataFrame

def get_map():
    f = open('mapa.json')
    data = json.load(f)
    f.close()
    return data["mapa"]

mapa = get_map()

def test():

    x = -2.8
    y = 2.8

    if x >= 0 and y >= 0:
        # Primer Cuadrante
        fila = 3 - math.trunc(y)
        columna = 4 + math.trunc(x)
    elif x < 0 and y >= 0:
        #Segundo cuadrante
        fila = 3 - math.trunc(y) 
        columna = 3 + math.trunc(x)
    elif x < 0 and y < 0:
        # Tercer Cuadrante
        fila = 4 - math.trunc(y)
        columna = 3 + math.trunc(x)
    elif x >= 0 and y < 0:
        # Cuarto Cuadrante
        fila = 4 - math.trunc(y)
        columna = 4 + math.trunc(x)
    print(x)
    print(y)
    print(fila)
    print(columna)
    mapa[fila][columna] = 1
    print(DataFrame(mapa))
test()