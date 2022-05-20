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
        output = [-1, 1]
        for i in range(len(output)):
            sim.data.ctrl[i] = output[i]

        sim.data.qpos[15] = 0.5 + math.cos(t*0.01)
        t += 1
        fitness += evaluar(sim.data.ctrl[1], sim.data.ncon)

    return fitness

def entrenamiento():
    episodes = 1
    steps = 3000
    for e in range(episodes):
        resultado = simular(steps, render=True)
        print(f'fitness {resultado}')
def evaluar(velocidad, datos_colision):
    if (detectar_colision(datos_colision)):
        #Retornar valores negativos si se desea descontar puntaje por cada frame que se está haciendo colisión.
        return 0
    #retornamos como maximo el valor 1 correspondiende a la velocidad.
    #se descuenta puntaje si este está retrosediendo (velocidad negativa) pues se considera que estaria pasando por un lugar que ya visitó.
    return velocidad

def detectar_colision(datos_colision):
    for i in range(datos_colision):
        contact = sim.data.contact[i]
        if(sim.model.geom_id2name(contact.geom1) == None or sim.model.geom_id2name(contact.geom2) == None):
            if(sim.model.geom_id2name(contact.geom1) != "floor" or sim.model.geom_id2name(contact.geom2) == "floor"):
                return True
    return False
entrenamiento()

