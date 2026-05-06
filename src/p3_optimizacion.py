"""
Encuentra la configuración óptima de hasta N estaciones que minimiza el costo
promedio de viaje, y genera visualizaciones de los resultados.

Estrategias implementadas:
    1. Greedy: construcción incremental eligiendo la estación que más reduce el costo.
    2. Exploración estocástica del espacio de soluciones.

Interfaz con Persona 2:
    evaluar_fn(escenarios, estaciones) → float (costo a minimizar)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from config import R, C


# 1. ALGORITMO GREEDY

def optimizar_greedy(evaluar_fn, escenarios, N, verbose=True):
    """
    Construcción greedy: agrega estaciones una a una eligiendo la que más reduce el costo.
    
    Estrategia
    ----------
    1. Comienza sin estaciones (o con una estación inicial).
    2. En cada iteración, prueba agregar cada nodo candidato de la cuadrícula.
    3. Selecciona el nodo que produce el menor costo.
    4. Repite hasta tener N estaciones.
    
    Parámetros
    ----------
    evaluar_fn : callable
        Función evaluar_configuracion(escenarios, estaciones) → float
    escenarios : list de ndarray
        Escenarios de población generados por Persona 1
    N : int
        Número máximo de estaciones
    verbose : bool
        Mostrar progreso
    
    Retorna
    -------
    mejor_config : ndarray, shape (N, 2)
        Coordenadas (fila, col) de las N estaciones
    mejor_costo : float
        Costo de la configuración final
    historial : list de float
        Costo en cada paso (longitud N+1, incluyendo inicial)
    """
    # Candidatos: todos los nodos de la cuadrícula
    candidatos = [(i, j) for i in range(R) for j in range(C)]
    
    # Inicialización: comenzar con el nodo central
    estaciones_actuales = []
    
    # Calcular costo inicial (sin estaciones = infinito, usamos una estación inicial)
    # Alternativa: comenzar con la estación que individualmente da menor costo

    mejor_costo_individual = np.inf
    mejor_inicial = None
    
    if verbose:
        print("Buscando estación inicial óptima...")
    
    # Muestreo para encontrar buen punto inicial (probamos todos los nodos para garantizar el óptimo)
    for nodo in candidatos:
        costo = evaluar_fn(escenarios, np.array([nodo]))
        if costo < mejor_costo_individual:
            mejor_costo_individual = costo
            mejor_inicial = nodo
    
    estaciones_actuales.append(mejor_inicial)
    historial = [mejor_costo_individual]
    
    if verbose:
        print(f"Estación inicial: {mejor_inicial}, costo: {mejor_costo_individual:.4f}")
    
    # Construcción greedy
    for k in range(1, N):
        mejor_costo_paso = np.inf
        mejor_nodo = None
        
        # Probar agregar cada candidato
        for nodo in candidatos:
            # Evitar duplicados
            if nodo in estaciones_actuales:
                continue
            
            # Evaluar configuración temporal
            config_temp = estaciones_actuales + [nodo]
            costo = evaluar_fn(escenarios, np.array(config_temp))
            
            if costo < mejor_costo_paso:
                mejor_costo_paso = costo
                mejor_nodo = nodo
        
        # Agregar la mejor estación encontrada
        estaciones_actuales.append(mejor_nodo)
        historial.append(mejor_costo_paso)
        
        if verbose:
            mejora = historial[-2] - historial[-1]
            print(f"Estación {k+1}/{N}: {mejor_nodo}, costo: {mejor_costo_paso:.4f}, "
                  f"mejora: {mejora:.4f}")
    
    mejor_config = np.array(estaciones_actuales)
    mejor_costo = historial[-1]
    
    return mejor_config, mejor_costo, historial


# 2. ALGORITMO DE EXPLORACIÓN ESTOCÁSTICA

def generar_vecino(estaciones, tipo='reemplazar'):
    """
    Genera una configuración vecina modificando ligeramente las estaciones.
    
    Tipos de movimiento:
        - 'reemplazar': reemplaza una estación por un nodo aleatorio
        - 'mover': mueve una estación a una posición adyacente
        - 'swap': intercambia dos estaciones
    """

    estaciones_nuevas = estaciones.copy()
    N = len(estaciones)
    
    if tipo == 'reemplazar' or np.random.rand() < 0.5:
        # Reemplazar una estación aleatoria
        idx = np.random.randint(N)
        estaciones_nuevas[idx] = [np.random.randint(R), np.random.randint(C)]
    else:
        # Mover una estación a posición adyacente
        idx = np.random.randint(N)
        delta_i = np.random.randint(-2, 3)
        delta_j = np.random.randint(-2, 3)
        nueva_i = np.clip(estaciones_nuevas[idx, 0] + delta_i, 0, R - 1)
        nueva_j = np.clip(estaciones_nuevas[idx, 1] + delta_j, 0, C - 1)
        estaciones_nuevas[idx] = [nueva_i, nueva_j]
    
    return estaciones_nuevas


def optimizar_recocido(evaluar_fn, escenarios, N, 
                       T_inicial=10.0, T_final=0.01, alpha=0.95,
                       max_iter=1000, verbose=True):
    """
    Optimización por recocido simulado
    
    Estrategia
    ----------
    1. Comienza con una configuración aleatoria o greedy.
    2. En cada iteración:
       - Genera una configuración vecina (pequeña modificación)
       - Si mejora, se acepta
       - Si empeora, se acepta con probabilidad exp(-ΔC/T)
    3. La temperatura T disminuye gradualmente (enfriamiento)
    4. Al final, converge a un óptimo local (idealmente global)
    
    Parámetros
    ----------
    evaluar_fn : callable
        Función de evaluación
    escenarios : list de ndarray
        Escenarios de población
    N : int
        Número de estaciones
    T_inicial : float
        Temperatura inicial (alta = más exploración)
    T_final : float
        Temperatura final
    alpha : float
        Factor de enfriamiento (T_new = alpha * T_old)
    max_iter : int
        Número máximo de iteraciones
    verbose : bool
        Mostrar progreso
    
    Retorna
    -------
    mejor_config : ndarray, shape (N, 2)
    mejor_costo : float
    historial : list de float
        Costo de la mejor solución en cada iteración
    """

    # Inicialización: configuración aleatoria
    estaciones_actual = np.array([[np.random.randint(R), np.random.randint(C)] 
                                   for _ in range(N)])
    costo_actual = evaluar_fn(escenarios, estaciones_actual)
    
    # Mejor solución encontrada
    mejor_config = estaciones_actual.copy()
    mejor_costo = costo_actual
    
    # Historial para visualización
    historial = [mejor_costo]
    
    # Temperatura
    T = T_inicial
    
    if verbose:
        print(f"Recocido Simulado: T_inicial={T_inicial}, T_final={T_final}, "
              f"alpha={alpha}, max_iter={max_iter}")
        print(f"Costo inicial: {costo_actual:.4f}")
    
    aceptadas = 0
    rechazadas = 0
    
    for iteracion in range(max_iter):
        # Generar vecino
        estaciones_vecino = generar_vecino(estaciones_actual)
        costo_vecino = evaluar_fn(escenarios, estaciones_vecino)
        
        # Diferencia de costo
        delta_C = costo_vecino - costo_actual
        
        # Criterio de aceptación
        if delta_C < 0:
            # Mejora: aceptar siempre
            aceptar = True
            aceptadas += 1
        else:
            # Empeora: aceptar con probabilidad exp(-ΔC/T)
            prob_aceptacion = np.exp(-delta_C / T)
            aceptar = np.random.rand() < prob_aceptacion
            if aceptar:
                aceptadas += 1
            else:
                rechazadas += 1
        
        if aceptar:
            estaciones_actual = estaciones_vecino
            costo_actual = costo_vecino
            
            # Actualizar mejor solución
            if costo_actual < mejor_costo:
                mejor_config = estaciones_actual.copy()
                mejor_costo = costo_actual
        
        # Guardar historial
        historial.append(mejor_costo)
        
        # Enfriamiento
        T = T * alpha
        
        # Mostrar progreso
        if verbose and (iteracion + 1) % 100 == 0:
            tasa_aceptacion = aceptadas / (aceptadas + rechazadas) if (aceptadas + rechazadas) > 0 else 0
            print(f"Iter {iteracion+1}/{max_iter}: T={T:.4f}, "
                  f"mejor_costo={mejor_costo:.4f}, "
                  f"tasa_aceptación={tasa_aceptacion:.2%}")
            aceptadas = rechazadas = 0
        
        # Terminar si temperatura es muy baja
        if T < T_final:
            if verbose:
                print(f"Temperatura final alcanzada en iteración {iteracion+1}")
            break
    
    if verbose:
        print(f"\nOptimización completada. Mejor costo: {mejor_costo:.4f}")
    
    return mejor_config, mejor_costo, historial

def ordenar_estaciones_tsp(estaciones):
    """
    Ordena las estaciones usando un algoritmo de vecino más cercano (Nearest Neighbor)
    para crear un ciclo cerrado de longitud mínima aproximada.
    """
    estaciones = np.array(estaciones)
    if len(estaciones) <= 1:
        return estaciones

    unvisited = list(range(1, len(estaciones)))
    ordered = [0]
    current_idx = 0

    while unvisited:
        # Encontrar el no visitado más cercano usando distancia Manhattan
        best_dist = np.inf
        best_idx = -1

        curr_pos = estaciones[current_idx]
        for candidate_idx in unvisited:
            cand_pos = estaciones[candidate_idx]
            dist = np.abs(curr_pos[0] - cand_pos[0]) + np.abs(curr_pos[1] - cand_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_idx = candidate_idx

        ordered.append(best_idx)
        unvisited.remove(best_idx)
        current_idx = best_idx

    return estaciones[ordered]


# 3. FUNCIÓN PRINCIPAL DE OPTIMIZACIÓN

def optimizar_ruta(evaluar_fn, escenarios, N, metodo='greedy', **kwargs):
    """
    Interfaz unificada para optimización de ruta.
    
    Parámetros
    ----------
    evaluar_fn : callable
        evaluar_configuracion de Persona 2
    escenarios : list de ndarray
        Escenarios de población de Persona 1
    N : int
        Número máximo de estaciones
    metodo : str
        'greedy' o 'recocido'
    **kwargs
        Parámetros adicionales para el método específico
    
    Retorna
    -------
    dict con claves:
        'estaciones': ndarray (N, 2)
        'costo': float
        'historial': list
        'metodo': str
    """
    if metodo == 'greedy':
        estaciones, costo, historial = optimizar_greedy(
            evaluar_fn, escenarios, N,
            verbose=kwargs.get('verbose', True)
        )
    elif metodo == 'recocido':
        estaciones, costo, historial = optimizar_recocido(
            evaluar_fn, escenarios, N,
            T_inicial=kwargs.get('T_inicial', 10.0),
            T_final=kwargs.get('T_final', 0.01),
            alpha=kwargs.get('alpha', 0.95),
            max_iter=kwargs.get('max_iter', 1000),
            verbose=kwargs.get('verbose', True)
        )
    else:
        raise ValueError(f"Método '{metodo}' no reconocido. Use 'greedy' o 'recocido'.")

    # Ordenar estaciones en un ciclo cerrado usando TSP Nearest Neighbor
    estaciones = ordenar_estaciones_tsp(estaciones)

    return {
        'estaciones': estaciones,
        'costo': costo,
        'historial': historial,
        'metodo': metodo
    }


# 4. VISUALIZACIONES

def visualizar_ruta_sobre_poblacion(escenario, estaciones, titulo="Ruta del tren"):
    """
    Superpone la ruta del tren sobre el mapa de densidad poblacional.
    
    Parámetros
    ----------
    escenario : ndarray (C, R)
        Mapa de población (un escenario)
    estaciones : ndarray (N, 2)
        Coordenadas de las estaciones
    titulo : str
    """
    ii, jj = np.meshgrid(np.arange(R), np.arange(C))
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Mapa de densidad
    cf = ax.contourf(ii, jj, escenario, 50, cmap='YlOrRd', alpha=0.8)
    plt.colorbar(cf, ax=ax, label='Población')
    
    # Estaciones
    ax.scatter(estaciones[:, 0], estaciones[:, 1], 
               c='blue', s=200, zorder=5, 
               edgecolors='white', linewidths=2,
               marker='s', label='Estaciones')
    
    # Numerar estaciones
    for k, (i, j) in enumerate(estaciones):
        ax.text(i, j, str(k+1), color='white', 
                ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Conectar estaciones (ruta)
    if len(estaciones) > 1:
        # Ruta lineal
        for k in range(len(estaciones) - 1):
            ax.plot([estaciones[k, 0], estaciones[k+1, 0]],
                   [estaciones[k, 1], estaciones[k+1, 1]],
                   'b-', linewidth=2, alpha=0.6)
        
        # Cerrar el ciclo (ruta cerrada)
        ax.plot([estaciones[-1, 0], estaciones[0, 0]],
               [estaciones[-1, 1], estaciones[0, 1]],
               'b--', linewidth=2, alpha=0.4, label='Cierre de ciclo')
    
    ax.set_xlabel('Fila (i)', fontsize=11)
    ax.set_ylabel('Columna (j)', fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def visualizar_convergencia(historial, titulo="Convergencia del algoritmo"):
    """
    Muestra cómo varía el costo a medida que se agregan estaciones o avanzan las iteraciones.
    
    Parámetros
    ----------
    historial : list de float
        Costo en cada paso
    titulo : str
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(historial, marker='o', linewidth=2, markersize=6, color='steelblue')
    ax.set_xlabel('Iteración / Número de estaciones', fontsize=11)
    ax.set_ylabel('Costo promedio (celdas)', fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.4)
    
    # Marcar el mejor costo
    mejor_idx = np.argmin(historial)
    mejor_costo = historial[mejor_idx]
    ax.axhline(mejor_costo, color='red', linestyle='--', alpha=0.7, 
               label=f'Mejor: {mejor_costo:.4f} (paso {mejor_idx})')
    ax.legend()
    
    plt.tight_layout()
    return fig, ax


def comparar_metodos(resultados_dict, escenario_ref):
    """
    Compara visualmente los resultados de diferentes métodos de optimización.
    
    Parámetros
    ----------
    resultados_dict : dict
        {nombre_metodo: resultado_optimizacion}
    escenario_ref : ndarray (C, R)
        Escenario de referencia para visualización
    """
    n_metodos = len(resultados_dict)
    fig, axes = plt.subplots(1, n_metodos, figsize=(7*n_metodos, 6))
    
    if n_metodos == 1:
        axes = [axes]
    
    ii, jj = np.meshgrid(np.arange(R), np.arange(C))
    
    for ax, (nombre, resultado) in zip(axes, resultados_dict.items()):
        estaciones = resultado['estaciones']
        costo = resultado['costo']
        
        # Mapa de población
        cf = ax.contourf(ii, jj, escenario_ref, 50, cmap='YlOrRd', alpha=0.7)
        plt.colorbar(cf, ax=ax)
        
        # Estaciones
        ax.scatter(estaciones[:, 0], estaciones[:, 1],
                  c='blue', s=200, zorder=5,
                  edgecolors='white', linewidths=2, marker='s')
        
        # Ruta
        if len(estaciones) > 1:
            for k in range(len(estaciones) - 1):
                ax.plot([estaciones[k, 0], estaciones[k+1, 0]],
                       [estaciones[k, 1], estaciones[k+1, 1]],
                       'b-', linewidth=2, alpha=0.6)
            ax.plot([estaciones[-1, 0], estaciones[0, 0]],
                   [estaciones[-1, 1], estaciones[0, 1]],
                   'b--', linewidth=2, alpha=0.4)
        
        ax.set_title(f'{nombre}\nCosto = {costo:.4f} celdas', fontweight='bold')
        ax.set_xlabel('Fila (i)')
        ax.set_ylabel('Columna (j)')
    
    plt.suptitle('Comparación de métodos de optimización', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, axes


# Tests básicos

if __name__ == '__main__':
    print("Módulo p3_optimizacion cargado correctamente.")
    print("Funciones disponibles:")
    print("  - optimizar_greedy()")
    print("  - optimizar_recocido()")
    print("  - optimizar_ruta()")
    print("  - visualizar_ruta_sobre_poblacion()")
    print("  - visualizar_convergencia()")
    print("  - comparar_metodos()")