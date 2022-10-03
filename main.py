import math
import os
import random
import glfw
import multiprocessing
import neat
import visualize
import json
import argparse
from xml.dom import minidom
import numpy as np
from pandas import DataFrame
from mujoco_py import MjSim, MjViewer, load_model_from_path
from neat import nn, parallel
from car_driver_neat import CarPopulation, CarConfig
from unidades import Auto, UnidadErratica, UnidadPredecible

#model = load_model_from_xml(MODEL_XML)
xml_path = './models/autito.xml'
model = load_model_from_path(xml_path)

cores = multiprocessing.cpu_count()
render = False
area_mapa = 0


parser = argparse.ArgumentParser(description='CarDriver NEAT')
parser.add_argument('--max-steps', dest='steps', type=int, default=30000,
                    help='The max number of steps to take per genome (timeout)')
parser.add_argument('--episodes', type=int, default=3,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--generations', type=int, default=200,
                    help="The number of generations to evolve the network")
parser.add_argument('--checkpoint', type=str,
                    help="Uses a checkpoint to start the simulation")
parser.add_argument('--training', type=bool, default=False,
                    help="continue training from that generation")                   

args = parser.parse_args()
steps = args.steps
episodes = args.episodes
generations = args.generations

args = parser.parse_args()
def get_map():
    f = open('mapa.json')
    data = json.load(f)
    f.close()
    return data["mapa"]

def espacios_recorridos(mapa):
    recorridos = 0
    for x in mapa:
        for y in x:
            if(y == 1):
                recorridos += 1
    return recorridos

def mostrar_mapa(mapa):
    recorridos = espacios_recorridos(mapa)
    print(f'el area del mapa es {area_mapa}')
    print(DataFrame(mapa))
    print("Espacios Recorridos: ", espacios_recorridos(mapa))
    print("Porcentaje Recorrido: ", recorridos*100/area_mapa, "%")

def get_new_map():
    new_mapa = np.zeros((10,20),dtype=int)
    # return copy.deepcopy(mapa_inicial)
    file = minidom.parse('models/autito.xml')
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



# Retorna el SQM vacio mas cercano y marca donde esta parado el vehiculo
def get_direccion(mapa, posicion_actual):

    x = float(posicion_actual[0])
    y = float(posicion_actual[1])

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
    
    if x < -3 or y > 3 or y < -3: #or x > 3
        print('x : ', x)
        print('y : ', y)
        print('Fuera del mapa')

    mapa[fila][columna] = 1


    return 

def worker_evaluate_genome(g, config):
    net = nn.FeedForwardNetwork.create(g, config)
    fitness = 0
    seed = config.config_information["seed"]
    for e in range(1, episodes + 1):
        #print(f'seed {seed} episidio {e} genoma {g.key}')
        fitness += simular_genoma(net, steps, render, seed)
        random.seed(config.config_information["seed"] * (e + 1))
        seed = random.random()
    fitness = fitness / 3
    print(f'El genoma {g.key} tuvo un Fitness de {fitness}')
    return fitness

def close_render(viewer):
    glfw.destroy_window(viewer.window)



def ejecutar_movimientos(unidades, step):
    for unidad in unidades:
        unidad.movimiento(step)

def simular_genoma(net, steps, render, seed):
    sim = MjSim(model)
    sim.reset()
    mapa = get_new_map()
    auto = Auto(0, 0, sim, "", qpos=0, nn=net,velocidad=0.1, render=render)
    objeto_erratico = UnidadErratica(10, 0 ,sim, seed,qpos = 21, nombre="randomMovingObject", render=render, velocidad=0.005)
    objeto_predecible = UnidadPredecible(2.5, 0 , sim, nombre="movingObject", qpos=14, render=render)
    unidades = [auto, objeto_erratico, objeto_predecible]
    if render:
        viewer = MjViewer(sim)
    else:
        sim.forward()
    for step in range(steps):
        sim.step()
        if render:
            viewer.render()
        ejecutar_movimientos(unidades, step)
        get_direccion(mapa, auto.posicion_vehiculo)
        # Terminacion de simulacion si cumple alguno de estos  criterios
        if auto.terminacion:
            break
    auto.ajustar_fitness(espacios_recorridos(mapa)/area_mapa)
    if render:
        close_render(viewer)
        mostrar_mapa(mapa)
    return auto.fitness


def evaluate_genome(g, config):
    net = neat.nn.FeedForwardNetwork.create(g, config)
    return simular_genoma(net, steps, render)

def eval_fitness(genomes):
    for g in genomes:
        fitness = evaluate_genome(g)
        g.fitness = fitness

def eval_genomes(genomes, config):
    #Simular cada genoma? 
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = simular_genoma(net, steps, render)

mapa_inicial = get_new_map() 
area_mapa = get_area_mapa(mapa_inicial)

# Simulation
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config2')
config = CarConfig(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
pop = CarPopulation(config)

if args.checkpoint:
    pop = neat.Checkpointer.restore_checkpoint(f'./checkpoints/neat-checkpoint-{args.checkpoint}')
    

stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.StdOutReporter(True))
pop.add_reporter(neat.Checkpointer(1, filename_prefix='./checkpoints/neat-checkpoint-'))
# Start simulation


if args.training == True:
    input("Presione enter para iniciar entrenamiento...")
    pe = parallel.ParallelEvaluator(cores, worker_evaluate_genome)
    winner = pop.run(pe.evaluate, generations)
elif not args.checkpoint:
    print('No se han ingresado parametros para iniciar correctamente')
    exit()


#print('Number of evaluations: {0}'.format(winner))

#visualization.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
#visualization.plot_species(stats, view=True, filename="feedforward-speciation.svg")

input("Presione enter para ejecutar el mejor genoma...")

if args.training == False:
    pe = parallel.ParallelEvaluator(cores, worker_evaluate_genome)
    winner = pop.run(pe.evaluate, 1)
else:
    visualize.plot_stats(stats, ylog=False, view=True, generations=generations)
    visualize.plot_species(stats, view=True)

# Comentar esto si se añaden mas nodos o debe ajustarse
node_names = {
    0: 'direccion',
    1: 'velocidad',
    -1: 'Sensor Proximidad frontal',
    -2: 'Sensor Proximidad frontal derecho',
    -3: 'Sensor Proximidad frontal izquierdo',
    -4: 'Sensor Proximidad derecho',
    -5: 'Sensor Proximidad izquierdo',
    -6: 'Sensor Proximidad trasero',
    -7: 'Sensor Proximidad trasero derecho',
    -8: 'Sensor Proximidad trasero izquierdo',
    }
visualize.draw_net(config, winner, False, node_names=node_names)

print('\nBest genome:\n{!s}'.format(winner))


winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
fitness = simular_genoma(winner_net, 100000, render=True, seed=random.random())
print(f'Fitness = {fitness}')