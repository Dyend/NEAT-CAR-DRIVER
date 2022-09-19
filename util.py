
def mantener_rango(rango1, rango2, valor_actual, suma, tamanio):
    nuevo_valor = suma + valor_actual
    if nuevo_valor > (rango1 + tamanio) and nuevo_valor < (rango2 - tamanio):
        return nuevo_valor
    return valor_actual

def detectar_colision(sim, nombre=None):
    datos_colision = sim.data.ncon
    for i in range(datos_colision):
        contact = sim.data.contact[i]
        if(sim.model.geom_id2name(contact.geom1) == nombre or sim.model.geom_id2name(contact.geom2) == nombre):
            if(sim.model.geom_id2name(contact.geom1) != "floor" and sim.model.geom_id2name(contact.geom2) != "floor"):
                return True
    return False