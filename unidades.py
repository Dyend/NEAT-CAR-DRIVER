import random
import math
from util import detectar_colision

# En este archivo se encuentran definido como objetos las unidades que estan en la simulacion.
class Unidad:
    def __init__(self, x, y, sim, nombre, qpos, velocidad=0.01, cant_movimientos=200, render=False):
        self.x = x
        self.y = y
        self.sim = sim
        self.velocidad = velocidad
        self.render = render
        # el attributo movimiento sirve para cambiar la direccion aleatoriamente cada tantos cant_movimientos
        self.cant_movimiento = cant_movimientos
        self.direccion_x = 0
        self.direccion_y = 0
        self.qpos_x = qpos
        self.qpos_y = qpos + 1
        self.qpos_z = qpos + 2
        self.nombre = nombre

    def movimiento(self, step):
        pass


class UnidadErratica(Unidad):
    def __init__(self, x, y, sim, seed, nombre, qpos, velocidad=0.01,cant_movimientos=200,  render=False):
        super().__init__(x, y , sim , nombre, qpos, velocidad, cant_movimientos, render)
        random.seed(seed)
        # eliminar despues solo para pruebas
        self.valor_inicial_semilla = []

    def direccion_aleatoria(self):
        valor_aleatorio = random.random()
        # eliminar solo para pruebas
        if len(self.valor_inicial_semilla) < 5:
            self.valor_inicial_semilla.append(valor_aleatorio)
        if valor_aleatorio > 0.5:
            return 1
        return -1

    def movimiento(self, step):
        if step % self.cant_movimiento == 0:
            self.direccion_x = self.direccion_aleatoria()
            self.direccion_y = self.direccion_aleatoria()

        if not detectar_colision(self.sim, nombre=self.nombre):
            self.x = self.sim.data.qpos[self.qpos_x]
            self.y = self.sim.data.qpos[self.qpos_y]
            self.sim.data.qpos[self.qpos_x] += self.velocidad * self.direccion_x
            self.sim.data.qpos[self.qpos_y] += self.velocidad * self.direccion_y
        else:
            self.direccion_x = self.direccion_x * -1
            self.direccion_y = self.direccion_y * -1

class UnidadPredecible(Unidad):

    def __init__(self, x, y, sim, nombre, qpos, velocidad=0.01, cant_movimientos=200, render=False):
        super().__init__(x, y, sim, nombre, qpos, velocidad, cant_movimientos, render)

    def movimiento(self, step):
        self.sim.data.qpos[self.qpos_y]+= (math.sin(step*0.01)-math.sin((step-1)*0.01))

class Auto(Unidad):

    def __init__(self, x, y, sim, nombre, qpos, nn, velocidad=0.01, cant_movimientos=200, render=False):
        super().__init__(x, y, sim, nombre, qpos, velocidad, cant_movimientos, render)
        self.neural_network = nn
        self.fitness = 0
        self.terminacion = False
        self.visitados = []
        self.posicion_vehiculo = []

    def get_inputs(self):
        sensor_proximidad_frontal = self.sim.data.sensordata[9]
        sensor_proximidad_frontal_derecho = self.sim.data.sensordata[10]
        sensor_proximidad_frontal_izquierdo = self.sim.data.sensordata[11]
        sensor_proximidad_derecho = self.sim.data.sensordata[12]
        sensor_proximidad_izquierdo = self.sim.data.sensordata[13]
        sensor_proximidad_trasero = self.sim.data.sensordata[14]
        sensor_proximidad_trasero_derecho = self.sim.data.sensordata[15]
        sensor_proximidad_trasero_izquierdo = self.sim.data.sensordata[16]
        sensor_giroscopio = self.sim.data.sensordata[3:6]
        sensor_velocimetro = self.sim.data.sensordata[6:9]
        sensor_acelerometro = self.sim.data.sensordata[0:3]
        _inputs = [#sensor_giroscopio[0],
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
        ]
        return _inputs

    def movimiento(self, step):
        # obtener inputs de la nn
        inputs = self.get_inputs()
        output = self.neural_network.activate(inputs)
        acelerar = output[0]
        direccion = output[1]

        if acelerar > 0.5 and self.sim.data.ctrl[1] < 1:
            self.sim.data.ctrl[1] += self.velocidad
        elif self.sim.data.ctrl[1] > -1:
            self.sim.data.ctrl[1] -= self.velocidad

        if direccion > 0.5 and self.sim.data.ctrl[0] < 1:
            self.sim.data.ctrl[0] += 0.01
        elif self.sim.data.ctrl[0] > -1:
            self.sim.data.ctrl[0] -= 0.01

        self.posicion_vehiculo = [format(self.sim.data.qpos[self.qpos_x],".1f"),format(self.sim.data.qpos[self.qpos_y],".1f")]
        # Obtiene la direccion del SQM vacio mas cercano
        #direccion = get_direccion(mapa, posicion_vehiculo)
        self.evaluar()
        # Terminacion de simulacion si cumple alguno de estos  criterios
        if step == 1000 and self.fitness <= 10:
            self.terminacion = True

    def evaluar(self):
        
        velocidad = self.sim.data.ctrl[1]
        if (detectar_colision(self.sim)):
            self.terminacion = True
            return
        posicion_vehiculo = (format(self.sim.data.qpos[self.qpos_x],".0f"),format(self.sim.data.qpos[self.qpos_y],".0f")) #verificar bien si corresponde al centro del vehiculo y no a una rueda
        if not (posicion_vehiculo in self.visitados):
            # este espacio no ha sido visitado
            self.visitados.append(posicion_vehiculo)
            #retornamos como maximo el valor 1 correspondiende a la velocidad.
            #se descuenta puntaje si este está retrosediendo (velocidad negativa) pues se considera que estaria pasando por un lugar que ya visitó.
            self.fitness += velocidad + len(self.visitados)

    def ajustar_fitness(self, porcentaje_area_recorrida):
        self.fitness = self.fitness*porcentaje_area_recorrida