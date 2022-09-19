import random
from util import detectar_colision

# En este archivo se encuentran definido como objetos las unidades que estan en la simulacion.

class ObjetoErratico:
    def __init__(self, x, y, sim, seed, nombre, qpos, velocidad=0.01,cant_movimientos=200,  render=False):
        self.x = x
        self.y = y
        self.step = 0
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

class ObjetoPredecible:
    pass

class Auto:
    pass