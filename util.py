
def mantener_rango(rango1, rango2, valor_actual, suma, tamanio):
    nuevo_valor = suma + valor_actual
    if nuevo_valor > (rango1 + tamanio) and nuevo_valor < (rango2 - tamanio):
        return nuevo_valor
    return valor_actual

