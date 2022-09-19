import math
import os
import random
import glfw
import multiprocessing
import neat
import visualize
import json
import copy
from pandas import DataFrame
from mujoco_py import MjSim, MjViewer, load_model_from_path
from neat import nn, parallel
from car_driver_neat import CarPopulation, CarConfig
from unidades import Auto, UnidadErratica, UnidadPredecible

#model = load_model_from_xml(MODEL_XML)
xml_path = './models/autito.xml'
model = load_model_from_path(xml_path)

cores = multiprocessing.cpu_count()
steps = 30000
render = False
generations = 1000
training = True
checkpoint = not training
_print = False

def get_map():
    f = open('mapa.json')
    data = json.load(f)
    f.close()
    return data["mapa"]

mapa_inicial = get_map() 

def get_new_map():
    return copy.deepcopy(mapa_inicial)

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
    
    if x < -3 or x > 3 or y > 3 or y < -3:
        print('x : ', x)
        print('y : ', y)
        print('Fuera del mapa')

    mapa[fila][columna] = 1


    return 

def worker_evaluate_genome(g, config):
    net = nn.FeedForwardNetwork.create(g, config)
    return simular_genoma(net, steps, render, config.config_information["seed"])

def close_render(viewer):
    glfw.destroy_window(viewer.window)


def mostrar_mapa(mapa):
    print(DataFrame(mapa))

def ejecutar_movimientos(unidades, step):
    for unidad in unidades:
        unidad.movimiento(step)

def simular_genoma(net, steps, render, seed):
    sim = MjSim(model)
    sim.reset()
    mapa = get_new_map()
    auto = Auto(0, 0, sim, "", qpos=0, nn=net,velocidad=0.1, render=render)
    objeto_erratico = UnidadErratica(10, 0 ,sim, seed,qpos = 21, nombre="randomMovingObject", render=render)
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
        # Terminacion de simulacion si cumple alguno de estos  criterios
        if auto.terminacion:
            break
    if render:
        close_render(viewer)
        mostrar_mapa(mapa)
    if _print:
        print(f"Fitness {auto.fitness}")
        print(objeto_erratico.valor_inicial_semilla)
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


# Simulation
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config2')
config = CarConfig(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
pop = CarPopulation(config)
if checkpoint:
    pop = neat.Checkpointer.restore_checkpoint('./checkpoints/neat-checkpoint-0')
    

stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.StdOutReporter(True))
pop.add_reporter(neat.Checkpointer(1, filename_prefix='./checkpoints/neat-checkpoint-'))
# Start simulation


if training:
    pe = parallel.ParallelEvaluator(cores, worker_evaluate_genome)
    winner = pop.run(pe.evaluate, generations)


#print('Number of evaluations: {0}'.format(winner))

#visualization.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
#visualization.plot_species(stats, view=True, filename="feedforward-speciation.svg")

input("Press Enter to run the best genome...")

_print = True
if checkpoint:
    pe = parallel.ParallelEvaluator(cores, worker_evaluate_genome)
    winner = pop.run(pe.evaluate, 1)
else:
    visualize.plot_stats(stats, ylog=False, view=True)
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
simular_genoma(winner_net, 100000, render=True, seed=random.random())
