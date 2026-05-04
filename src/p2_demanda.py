"""
Persona 2 – Demanda y Evaluación
=================================
Calcula flujos de viaje entre nodos según densidad poblacional y define la
función de costo de una configuración de estaciones.

Convenio de matrices (heredado de p1_ciudad.py):
    Las matrices de población tienen shape (C, R):
        pob[c, r] = población en fila r, columna c
    Coincide con la salida de generar_escenarios() de p1.

Estaciones:
    Se representan como array-like de tuplas (fila, col),
    con fila ∈ [0, R) y col ∈ [0, C).

Distancia:
    Se usa la distancia Manhattan (L1), natural para desplazamiento en cuadrícula.
"""

import numpy as np
from config import R, C


# ---------------------------------------------------------------------------
# Distancias a la estación más cercana
# ---------------------------------------------------------------------------

def distancias_minimas(estaciones):
    """
    Distancia Manhattan de cada nodo de la cuadrícula a la estación más cercana.

    Parámetros
    ----------
    estaciones : array-like, shape (N_est, 2)
        Cada fila es (fila, col) de una estación.

    Retorna
    -------
    d_min : ndarray, shape (C, R)
        d_min[c, r] = distancia al nodo (fila=r, col=c) desde la estación más cercana.
    """
    estaciones = np.array(estaciones)
    if estaciones.ndim == 1:
        estaciones = estaciones[np.newaxis, :]

    # Grilla con la misma convención que p1: ii[c,r]=r  jj[c,r]=c
    ii, jj = np.meshgrid(np.arange(R), np.arange(C))  # shape (C, R)

    d_min = np.full((C, R), np.inf)
    for est in estaciones:
        fila_est, col_est = int(est[0]), int(est[1])
        dist = np.abs(ii - fila_est) + np.abs(jj - col_est)
        d_min = np.minimum(d_min, dist)

    return d_min


# ---------------------------------------------------------------------------
# Costo de una configuración de estaciones para un escenario
# ---------------------------------------------------------------------------

def calcular_costo_escenario(pob, estaciones):
    """
    Costo de viaje promedio ponderado por población para un único escenario.

    Modelo de viaje
    ---------------
    Un usuario parte del nodo o y va al nodo d.
    - P(origen = o)       ∝ pop(o)
    - P(destino = d | o)  ∝ pop(d),   d ≠ o

    Costo del viaje: dist_min(o) + dist_min(d)
        donde dist_min(x) = distancia Manhattan de x a la estación más cercana.

    Desarrollando la esperanza (ver derivación en el notebook):
        E[costo] = Σ_o  p(o) · [d_min(o)  +  E[d_min(d) | o]]
    con
        E[d_min(d) | o] = (Σ_d pop(d)·d_min(d) − pop(o)·d_min(o))
                          / (total_pop − pop(o))

    Parámetros
    ----------
    pob        : ndarray, shape (C, R)  –  salida de generar_escenarios() de p1.
    estaciones : array-like, shape (N_est, 2)  –  tuplas (fila, col).

    Retorna
    -------
    float : costo promedio ponderado (unidades: celdas de la cuadrícula).
    """
    d_min = distancias_minimas(estaciones)          # (C, R)

    pop_flat = pob.flatten().astype(float)           # (C·R,)
    d_flat   = d_min.flatten()                       # (C·R,)

    total_pop = pop_flat.sum()
    if total_pop == 0:
        return 0.0

    p_origen = pop_flat / total_pop                  # P(origen = o)

    # Σ_d pop(d)·d_min(d)
    suma_pop_d = np.dot(pop_flat, d_flat)

    # E[d_min(dest) | origen = o]
    denominadores = total_pop - pop_flat
    numeradores   = suma_pop_d - pop_flat * d_flat
    e_dest_dado_o = np.where(denominadores > 0,
                             numeradores / denominadores,
                             0.0)

    # E[costo] = Σ_o p(o) · (d_min(o) + E[d_dest | o])
    costo = np.dot(p_origen, d_flat + e_dest_dado_o)
    return float(costo)


# ---------------------------------------------------------------------------
# Costo promedio sobre múltiples escenarios
# ---------------------------------------------------------------------------

def calcular_costo_promedio(escenarios, estaciones):
    """
    Costo promedio ponderado sobre una lista de escenarios de población.

    Parámetros
    ----------
    escenarios : list de ndarray (C, R)  –  salida de generar_escenarios() de p1.
    estaciones : array-like, shape (N_est, 2)  –  tuplas (fila, col).

    Retorna
    -------
    float : media del costo sobre todos los escenarios.
            Es la métrica que la Persona 3 debe minimizar.
    """
    costos = [calcular_costo_escenario(pob, estaciones) for pob in escenarios]
    return float(np.mean(costos))


# ---------------------------------------------------------------------------
# Utilidad: costo por nodo (para visualización)
# ---------------------------------------------------------------------------

def costo_por_nodo(pob, estaciones):
    """
    Contribución al costo de cada nodo como origen (útil para mapas de calor).

    Retorna
    -------
    contrib : ndarray, shape (C, R)
        contrib[c, r] = p(o=(r,c)) · (d_min(o) + E[d_dest | o])
        La suma de contrib es igual al costo total del escenario.
    """
    d_min = distancias_minimas(estaciones)

    pop_flat = pob.flatten().astype(float)
    d_flat   = d_min.flatten()
    total_pop = pop_flat.sum()

    if total_pop == 0:
        return np.zeros((C, R))

    p_origen     = pop_flat / total_pop
    suma_pop_d   = np.dot(pop_flat, d_flat)
    denominadores = total_pop - pop_flat
    numeradores   = suma_pop_d - pop_flat * d_flat
    e_dest_dado_o = np.where(denominadores > 0, numeradores / denominadores, 0.0)

    contrib_flat = p_origen * (d_flat + e_dest_dado_o)
    return contrib_flat.reshape(C, R)
