import math
import os
import random
import glfw
import multiprocessing
import neat
import argparse
from mujoco_py import MjSim, MjViewer, load_model_from_path
from neat import nn, parallel
from car_driver_neat import CarPopulation, CarConfig
from unidades import Auto
from map import get_new_map, mostrar_mapa, espacios_recorridos, get_area_mapa, load_levels, get_unidades_dict
from unidades import create_unidades

cores = multiprocessing.cpu_count()
render = False
area_mapa = 0


parser = argparse.ArgumentParser(description='CarDriver NEAT')
parser.add_argument('--max-steps', dest='steps', type=int, default=50000,
                    help='The max number of steps to take per genome (timeout)')
parser.add_argument('--episodes', type=int, default=3,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--generations', type=int, default=200,
                    help="The number of generations to evolve the network")
parser.add_argument('--checkpoint', type=str,
                    help="Uses a checkpoint to start the simulation")
parser.add_argument('--training', type=bool, default=False,
                    help="continue training from that generation")
parser.add_argument('--seed', type=float,
                    help="seed to test the checkpoint")
parser.add_argument('--show_seed', type=int, default=False,
                    help="print seed for specific genome")
parser.add_argument('--map', type=int, default=load_levels(),
                    help="map to be used when loading checkpoint")

args = parser.parse_args()
steps = args.steps
episodes = args.episodes
generations = args.generations

args = parser.parse_args()


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
    
    if x < -3 or y > 3 or y < -9: #or x > 3
        print('x : ', x)
        print('y : ', y)
        print('Fuera del mapa')

    mapa[fila][columna] = 1


    return 

def worker_evaluate_genome(g, config):
    net = nn.FeedForwardNetwork.create(g, config)
    total_fitness = 0
    seed = config.config_information["seed"]
    map_area = config.config_information["map_area"]
    map_path = config.config_information["map_path"]
    unidades_dict = config.config_information["unidades_dict"]
    for e in range(1, episodes + 1):
        fitness = simular_genoma(net, steps, render, seed, map_path, map_area, unidades_dict)
        total_fitness += fitness
        if args.show_seed and args.show_seed == g.key:
            print(f'seed {seed} episidio {e} genoma {g.key} fitness {fitness} en mapa {map_path}')
        random.seed(config.config_information["seed"] * (e + 1))
        seed = random.random()
    total_fitness = total_fitness / episodes
    if args.show_seed:
        if args.show_seed == g.key:
            print(f'El genoma {g.key} tuvo un Fitness de {total_fitness}')
    else:
        print(f'El genoma {g.key} tuvo un Fitness de {total_fitness}')
    return total_fitness

def close_render(viewer):
    glfw.destroy_window(viewer.window)



def ejecutar_movimientos(unidades, step):
    for unidad in unidades:
        unidad.movimiento(step)

def simular_genoma(net, steps, render, seed, map_path, area_mapa, unidades_dict):
    model = load_model_from_path(map_path)
    sim = MjSim(model)
    sim.reset()
    mapa = get_new_map(map_path)
    auto = Auto(0, 0, sim, "", qpos=0, nn=net,velocidad=0.1, render=render)
    unidades = [auto] + create_unidades(unidades_dict= unidades_dict, sim=sim, render=render, seed=seed)
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
        mostrar_mapa(mapa, area_mapa)
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


print('\nBest genome:\n{!s}'.format(winner))


winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
if args.seed:
    seed = args.seed
else:
    seed = random.random()

map_path = f'./models/levels/{args.map}.xml'
unidades = get_unidades_dict(map_path)
mapa = get_new_map(map_path)
area_mapa = get_area_mapa(mapa)
fitness = simular_genoma(winner_net, 100000, render=True, seed=seed, map_path=map_path, area_mapa=area_mapa, unidades_dict=unidades)
print(f'Fitness = {fitness}')