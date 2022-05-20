from mujoco_py import load_model_from_xml, MjSim, MjViewer, load_model_from_path
import math
import os

#model = load_model_from_xml(MODEL_XML)
xml_path = './models/autito.xml'
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

def simular(steps, render=False):
    t = 0
    fitness = 0
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
        output = [0, 0]
        for i in range(output):
            sim.data.ctrl[i] = output[i]

        sim.data.qpos[15] = 0.5 + math.cos(t*0.01)
        t += 1
        estado = sim.get_state()
        fitness += evaluar(estado)

    return fitness

def entrenamiento():
    episodes = 1
    steps = 3000
    for e in range(episodes):
        resultado = simular(steps, render=True)

def evaluar(estado):
    return 1

entrenamiento()

