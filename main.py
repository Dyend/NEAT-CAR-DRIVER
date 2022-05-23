import math
import os
import numpy as np
import glfw
import multiprocessing
import neat
from mujoco_py import load_model_from_xml, MjSim, MjViewerBasic, load_model_from_path
from neat import nn, population, statistics, parallel

#model = load_model_from_xml(MODEL_XML)
xml_path = './models/autito.xml'
model = load_model_from_path(xml_path)

cores = multiprocessing.cpu_count()
episodios = 2
steps = 6000
render = False
generations = 100
training = True
checkpoint = False

def worker_evaluate_genome(g, config):
    net = nn.FeedForwardNetwork.create(g, config)
    return simular(net, episodios, steps, render)

def close_render(viewer):
    glfw.destroy_window(viewer.window)

def simular_genoma(net, sim, steps, render):
    sim.reset()
    t = 0
    visitado = []
    fitness = 0
    if render:
        viewer = MjViewerBasic(sim)
    for step in range(steps):
        sim.step()
        if render:
            viewer.render()
        sensor_proximidad = sim.data.get_sensor("rangefinder")
        sensor_giroscopio = sim.data.get_sensor("gyro")
        sensor_velocimetro = sim.data.get_sensor("velocimeter")
        sensor_acelerometro = sim.data.get_sensor("accelerometer")
        #Entradas
        input = [sensor_proximidad, sensor_giroscopio, sensor_acelerometro, sensor_velocimetro]
        # Salidas
        output = net.activate(input)
        for i in range(len(output)):
            sim.data.ctrl[i] = output[i]

        sim.data.qpos[15] = 0.5 + math.cos(t*0.01)
        t += 1
        criterio, visitado = evaluar(visitado, sim)
        fitness += criterio
    if render:
        close_render(viewer)
    print(f"Fitness {fitness}")
    return fitness

def simular(net, episodes, steps, render=False):
    episodes = 1
    steps = 3000
    fitnesses = []
    sim = MjSim(model)
    for e in range(episodes):
        fitnesses.append(simular_genoma(net, sim , steps, render))
    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    return fitness


def evaluar(visitado, sim):
    criterio = 0
    velocidad = sim.data.ctrl[1]
    datos_colision = sim.data.ncon
    if((format(sim.data.qpos[0],".1f"),format(sim.data.qpos[1],".1f")) in visitado):
        # print("Este espacio ya fué visitado")
        return criterio, visitado
    visitado.append((format(sim.data.qpos[0],".1f"),format(sim.data.qpos[1],".1f")))
    if (detectar_colision(datos_colision, sim)):
        #Retornar valores negativos si se desea descontar puntaje por cada frame que se está haciendo colisión.
        return criterio, visitado
    #retornamos como maximo el valor 1 correspondiende a la velocidad.
    #se descuenta puntaje si este está retrosediendo (velocidad negativa) pues se considera que estaria pasando por un lugar que ya visitó.
    criterio += velocidad
    return criterio, visitado

def detectar_colision(datos_colision, sim):
    for i in range(datos_colision):
        contact = sim.data.contact[i]
        if(sim.model.geom_id2name(contact.geom1) == None or sim.model.geom_id2name(contact.geom2) == None):
            if(sim.model.geom_id2name(contact.geom1) != "floor" or sim.model.geom_id2name(contact.geom2) == "floor"):
                return True
    return False

def evaluate_genome(g, config):
    net = neat.nn.FeedForwardNetwork.create(g, config)
    return simular(net, episodios, steps, render)

def eval_fitness(genomes):
    for g in genomes:
        fitness = evaluate_genome(g)
        g.fitness = fitness

def eval_genomes(genomes, config):
    sim = MjSim(model)
    #Simular cada genoma? 
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = simular_genoma(net, sim, steps, render)


# Simulation
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config2')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

pop = population.Population(config)
if checkpoint:
    pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-99')
    

stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.StdOutReporter(True))
pop.add_reporter(neat.Checkpointer(generations/2))
# Start simulation


if training:
    pe = parallel.ParallelEvaluator(cores, worker_evaluate_genome)
    winner = pop.run(pe.evaluate, generations)


#print('Number of evaluations: {0}'.format(winner))

#visualization.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
#visualization.plot_species(stats, view=True, filename="feedforward-speciation.svg")

input("Press Enter to run the best genome...")

if checkpoint:
    winner = pop.run(eval_genomes, 1)

# Save best network
import pickle
with open('winner.pkl', 'wb') as output:
   pickle.dump(winner, output, 1)

print('\nBest genome:\n{!s}'.format(winner))


winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
simular(winner_net, 1, 10000, render=True)
