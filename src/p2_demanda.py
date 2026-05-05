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

Interfaz con Persona 3:
    costo = evaluar_configuracion(escenarios, estaciones)
    → float a minimizar
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import R, C


# ---------------------------------------------------------------------------
# 1. Distancias a la estación más cercana
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
        d_min[c, r] = distancia del nodo (fila=r, col=c) a la estación más cercana.
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
# 2. Probabilidades destino dado origen
# ---------------------------------------------------------------------------

def probabilidades_destino(pop_flat, origen_idx):
    """
    Distribución de probabilidad del destino dado un origen.

    P(destino=d | origen=o) = pop(d) / (pop_total − pop(o)),  d ≠ o

    Parámetros
    ----------
    pop_flat   : ndarray 1D de shape (N,) — población aplanada del escenario.
    origen_idx : int — índice del nodo origen en pop_flat.

    Retorna
    -------
    p_dest : ndarray 1D de shape (N,)
        Probabilidades que suman 1; p_dest[origen_idx] = 0.
    """
    pop_flat = np.asarray(pop_flat, dtype=float)
    denom = pop_flat.sum() - pop_flat[origen_idx]
    if denom <= 0:
        return np.zeros_like(pop_flat)
    p_dest = pop_flat.copy()
    p_dest[origen_idx] = 0.0
    p_dest /= denom
    return p_dest


# ---------------------------------------------------------------------------
# 3. Matriz de flujos OD
# ---------------------------------------------------------------------------

def calcular_flujos(pob):
    """
    Matriz de flujos origen-destino para un escenario.

    flujo(o, d) = pop(o) * pop(d),   o ≠ d  (diagonal = 0)

    El flujo es proporcional al número esperado de viajes entre cada par.
    La normalización (dividir por flujo_total) se aplica en evaluar_configuracion.

    Parámetros
    ----------
    pob : ndarray, shape (C, R) — salida de generar_escenarios() de p1.

    Retorna
    -------
    flujos : ndarray, shape (N, N)  donde N = C*R
        flujos[o_flat, d_flat] = pop_flat[o_flat] * pop_flat[d_flat]
        La diagonal es 0 (sin auto-viajes).
    flujo_total : float
        Suma de todos los flujos = Σ_{o≠d} pop(o)*pop(d)
                                 = pop_total² − Σ_o pop(o)²
    """
    pop_flat = pob.flatten().astype(float)              # (N,)
    flujos = np.outer(pop_flat, pop_flat)               # (N, N)
    np.fill_diagonal(flujos, 0.0)
    flujo_total = float(flujos.sum())
    return flujos, flujo_total


# ---------------------------------------------------------------------------
# 4. Costo de una configuración para un escenario (O(N) vectorizado)
# ---------------------------------------------------------------------------

def calcular_costo_escenario(pob, estaciones):
    """
    Costo de viaje promedio ponderado por flujo para un único escenario.

    Modelo de viaje
    ---------------
    - flujo(o, d) ∝ pop(o) * pop(d),   o ≠ d
    - costo(o, d) = d_min(o) + d_min(d)   (tren despreciable)

    Fórmula O(N) equivalente a la suma completa sobre todos los pares OD:
        E[costo] = Σ_o p(o) · [d_min(o) + E[d_min(d) | o]]
    con
        E[d_min(d) | o] = (Σ_d pop(d)·d_min(d) − pop(o)·d_min(o))
                          / (pop_total − pop(o))

    Parámetros
    ----------
    pob        : ndarray, shape (C, R)
    estaciones : array-like, shape (N_est, 2)

    Retorna
    -------
    float : costo promedio ponderado en celdas de la cuadrícula.
    """
    d_min = distancias_minimas(estaciones)              # (C, R)

    pop_flat = pob.flatten().astype(float)              # (N,)
    d_flat   = d_min.flatten()                          # (N,)

    total_pop = pop_flat.sum()
    if total_pop == 0:
        return 0.0

    p_origen    = pop_flat / total_pop                  # P(origen = o)
    suma_pop_d  = np.dot(pop_flat, d_flat)              # Σ_d pop(d)·d_min(d)

    denominadores = total_pop - pop_flat                # pop_total − pop(o)
    numeradores   = suma_pop_d - pop_flat * d_flat      # Σ_{d≠o} pop(d)·d_min(d)
    e_dest_dado_o = np.where(denominadores > 0,
                             numeradores / denominadores,
                             0.0)

    return float(np.dot(p_origen, d_flat + e_dest_dado_o))


# ---------------------------------------------------------------------------
# 5. Función principal: evaluar_configuracion
# ---------------------------------------------------------------------------

def evaluar_configuracion(escenarios, estaciones):
    """
    Costo promedio ponderado por flujo sobre todos los escenarios.

    Fórmula
    -------
        C̄(S) = Σ_e  w_e · costo_e(S)
                ─────────────────────
                Σ_e  w_e

    donde w_e = Σ_{o≠d} pop_e(o)·pop_e(d)  = pop_total_e² − Σ_o pop_e(o)²
    es el flujo total del escenario e, y costo_e(S) es el costo promedio
    ponderado dentro del escenario (calculado en O(N) por calcular_costo_escenario).

    Escenarios con mayor población pesan más, lo que es consistente con que
    el denominador sea la suma total de flujos y no el número de escenarios.

    Parámetros
    ----------
    escenarios : list de ndarray (C, R) — salida de generar_escenarios() de p1.
    estaciones : array-like, shape (N_est, 2) — tuplas (fila, col).

    Retorna
    -------
    float : costo promedio ponderado (celdas). Minimizar para Persona 3.
    """
    costos  = np.empty(len(escenarios))
    pesos   = np.empty(len(escenarios))

    for k, pob in enumerate(escenarios):
        pop_flat = pob.flatten().astype(float)
        # w_e = Σ_{o≠d} pop(o)*pop(d) = pop_total² − ||pop||²
        pesos[k]  = pop_flat.sum() ** 2 - np.dot(pop_flat, pop_flat)
        costos[k] = calcular_costo_escenario(pob, estaciones)

    if pesos.sum() == 0:
        return 0.0

    return float(np.average(costos, weights=pesos))


# Alias por compatibilidad con código que use el nombre anterior
calcular_costo_promedio = evaluar_configuracion


# ---------------------------------------------------------------------------
# 6. Utilidad: contribución al costo por nodo (para visualización)
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
    d_min    = distancias_minimas(estaciones)
    pop_flat = pob.flatten().astype(float)
    d_flat   = d_min.flatten()
    total_pop = pop_flat.sum()

    if total_pop == 0:
        return np.zeros((C, R))

    p_origen      = pop_flat / total_pop
    suma_pop_d    = np.dot(pop_flat, d_flat)
    denominadores = total_pop - pop_flat
    numeradores   = suma_pop_d - pop_flat * d_flat
    e_dest_dado_o = np.where(denominadores > 0, numeradores / denominadores, 0.0)

    return (p_origen * (d_flat + e_dest_dado_o)).reshape(C, R)


# ---------------------------------------------------------------------------
# Tests básicos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(0)

    # --- Población sintética uniforme ---
    pob_uniforme = np.ones((C, R))

    # 1. probabilidades_destino suma 1 y excluye origen
    pop_flat = pob_uniforme.flatten()
    p = probabilidades_destino(pop_flat, origen_idx=0)
    assert abs(p.sum() - 1.0) < 1e-10, "probabilidades_destino no suma 1"
    assert p[0] == 0.0, "probabilidades_destino incluye el origen"
    print("OK  probabilidades_destino: suma 1, excluye origen")

    # 2. calcular_flujos: diagonal cero, flujo_total correcto
    flujos, flujo_total = calcular_flujos(pob_uniforme)
    N = C * R
    assert flujos.shape == (N, N), "flujos tiene shape incorrecto"
    assert np.all(np.diag(flujos) == 0), "diagonal de flujos no es 0"
    expected_total = float(N) * (N - 1)   # pop=1 en todos los nodos
    assert abs(flujo_total - expected_total) < 1e-8, "flujo_total incorrecto"
    print(f"OK  calcular_flujos: shape {flujos.shape}, flujo_total={flujo_total:.0f}")

    # 3. calcular_costo_escenario: valor razonable, > 0
    estaciones_test = np.array([[5, 5], [5, 15], [15, 5], [15, 15], [10, 10], [0, 10]])
    costo = calcular_costo_escenario(pob_uniforme, estaciones_test)
    assert costo > 0, "costo debe ser positivo"
    print(f"OK  calcular_costo_escenario: {costo:.4f} celdas")

    # 4. evaluar_configuracion: escalar float
    escenarios_test = [np.random.poisson(5, (C, R)) for _ in range(10)]
    resultado = evaluar_configuracion(escenarios_test, estaciones_test)
    assert isinstance(resultado, float), "evaluar_configuracion no retorna float"
    assert resultado > 0, "evaluar_configuracion debe ser positivo"
    print(f"OK  evaluar_configuracion: {resultado:.4f} celdas")

    # 5. Más estaciones → costo menor o igual
    estaciones_pocas  = np.array([[10, 10]])
    estaciones_muchas = estaciones_test
    costo_pocas  = evaluar_configuracion(escenarios_test, estaciones_pocas)
    costo_muchas = evaluar_configuracion(escenarios_test, estaciones_muchas)
    assert costo_muchas <= costo_pocas, "más estaciones deben reducir el costo"
    print(f"OK  mas estaciones reducen costo: {costo_pocas:.4f} -> {costo_muchas:.4f}")

    # 6. Alias funciona
    assert calcular_costo_promedio is evaluar_configuracion
    print("OK  alias calcular_costo_promedio apunta a evaluar_configuracion")

    print("\nTodos los tests pasaron.")
