import math
import os
import gc
import numpy as np
import glfw
import multiprocessing
import neat
import visualize
import json
import copy
from pandas import DataFrame
from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjViewerBasic, load_model_from_path, MjRenderContextOffscreen
from neat import nn, population, statistics, parallel

#model = load_model_from_xml(MODEL_XML)
xml_path = './models/autito.xml'
model = load_model_from_path(xml_path)

cores = multiprocessing.cpu_count()
steps = 30000
render = False
generations = 1000
training = True
checkpoint = False
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
    return simular_genoma(net, steps, render)

def close_render(viewer):
    glfw.destroy_window(viewer.window)


def mostrar_mapa(mapa):
    print(DataFrame(mapa))


def simular_genoma(net, steps, render):
    choque = 0
    sim = MjSim(model)
    sim.reset()
    mapa = get_new_map()
    visitado = []
    fitness = 0
    if render:
        viewer = MjViewer(sim)
    else:
        sim.forward()
    for step in range(steps):
        sim.step()
        if render:
            viewer.render()
        sensor_proximidad_frontal = sim.data.sensordata[9]
        sensor_proximidad_frontal_derecho = sim.data.sensordata[10]
        sensor_proximidad_frontal_izquierdo = sim.data.sensordata[11]
        sensor_proximidad_derecho = sim.data.sensordata[12]
        sensor_proximidad_izquierdo = sim.data.sensordata[13]
        sensor_proximidad_trasero = sim.data.sensordata[14]
        sensor_proximidad_trasero_derecho = sim.data.sensordata[15]
        sensor_proximidad_trasero_izquierdo = sim.data.sensordata[16]
        sensor_giroscopio = sim.data.sensordata[3:6]
        sensor_velocimetro = sim.data.sensordata[6:9]
        sensor_acelerometro = sim.data.sensordata[0:3]
        posicion_vehiculo = (format(sim.data.qpos[0],".1f"),format(sim.data.qpos[1],".1f"))
        # Obtiene la direccion del SQM vacio mas cercano
        direccion = get_direccion(mapa, posicion_vehiculo)
        #Entradas
        sensor_id = [sim.model.sensor_name2id('front-rangefinder'),
        sim.model.sensor_name2id('gyro'),
        sim.model.sensor_name2id('velocimeter'),
        sim.model.sensor_name2id('accelerometer'),

        ]
        input = [#sensor_giroscopio[0],
        #sensor_giroscopio[1],
        #sensor_giroscopio[2],
        #sensor_acelerometro[0],
        #sensor_acelerometro[1],
        #ensor_acelerometro[2],
        #sensor_velocimetro[0],
        #sensor_velocimetro[1],
        #sensor_velocimetro[2],
        sensor_proximidad_frontal,
        sensor_proximidad_frontal_derecho,
        sensor_proximidad_frontal_izquierdo,
        sensor_proximidad_derecho,
        sensor_proximidad_izquierdo,
        sensor_proximidad_trasero,
        sensor_proximidad_trasero_derecho,
        sensor_proximidad_trasero_izquierdo,
                ]#, direccion]
        # Salidas
        output = net.activate(input)
        #for i in range(len(output)):
        #   sim.data.ctrl[i] = output[i]

        acelerar = output[0]
        direccion = output[1]

        if acelerar > 0.5 and sim.data.ctrl[1] < 1:
            sim.data.ctrl[1] += 0.1
        elif sim.data.ctrl[1] > -1:
            sim.data.ctrl[1] -= 0.1

        if direccion > 0.5 and sim.data.ctrl[0] < 1:
            sim.data.ctrl[0] += 0.01
        elif sim.data.ctrl[0] > -1:
            sim.data.ctrl[0] -= 0.01

        # Movimiento del cubo
        sim.data.qpos[15] += (math.sin(step*0.01)-math.sin((step-1)*0.01))
        criterio, visitado, choque = evaluar(visitado, sim, choque)
        # Terminacion de simulacion si cumple alguno de estos  criterios
        if step == 1000 and fitness <= 10:
            break
        if (choque == 1):
            break
        fitness += criterio
    if render:
        close_render(viewer)
        mostrar_mapa(mapa)
    if _print:
        print(f"Fitness {fitness}")

    return fitness


def evaluar(visitado, sim, choque):
    criterio = 0
    velocidad = sim.data.ctrl[1]
    datos_colision = sim.data.ncon
    if (detectar_colision(datos_colision, sim)):
        #Retornar valores negativos si se desea descontar puntaje por cada frame que se está haciendo colisión.
        choque = 1
        return criterio, visitado, choque
    posicion_vehiculo = (format(sim.data.qpos[0],".0f"),format(sim.data.qpos[1],".0f")) #verigicar bien si corresponde al centro del vehiculo y no a una rueda
    if (posicion_vehiculo in visitado):
        # print("Este espacio ya fué visitado")
        return criterio, visitado, choque
    visitado.append(posicion_vehiculo)
    #retornamos como maximo el valor 1 correspondiende a la velocidad.
    #se descuenta puntaje si este está retrosediendo (velocidad negativa) pues se considera que estaria pasando por un lugar que ya visitó.
    criterio += velocidad + len(visitado)
    return criterio, visitado, choque

def detectar_colision(datos_colision, sim):
    for i in range(datos_colision):
        contact = sim.data.contact[i]
        if(sim.model.geom_id2name(contact.geom1) == None or sim.model.geom_id2name(contact.geom2) == None):
            if(sim.model.geom_id2name(contact.geom1) != "floor" and sim.model.geom_id2name(contact.geom2) != "floor"):
                return True
    return False

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
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
pop = population.Population(config)
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
simular_genoma(winner_net, 100000, render=True)
