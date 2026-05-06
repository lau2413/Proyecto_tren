"""
Ejecuta ambos algoritmos (greedy y recocido simulado), compara resultados
y genera visualizaciones.

Uso:
    python scripts/run_p3.py [--metodo greedy|recocido|ambos] [--n_estaciones N]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import argparse

from config import R, C, N_ESTACIONES, T, NUM_ESCENARIOS, SEMILLA
from src.p1_ciudad import generar_escenarios
from src.p2_demanda import evaluar_configuracion
from src.p3_optimizacion import (
    optimizar_ruta,
    visualizar_ruta_sobre_poblacion,
    visualizar_convergencia,
    comparar_metodos
)


def main():
    parser = argparse.ArgumentParser(description='Optimización de ruta del tren')
    parser.add_argument('--metodo', type=str, default='ambos',
                       choices=['greedy', 'recocido', 'ambos'],
                       help='Método de optimización a usar')
    parser.add_argument('--n_estaciones', type=int, default=N_ESTACIONES,
                       help=f'Número de estaciones (default: {N_ESTACIONES})')
    parser.add_argument('--num_escenarios', type=int, default=NUM_ESCENARIOS,
                       help=f'Número de escenarios (default: {NUM_ESCENARIOS})')
    parser.add_argument('--semilla', type=int, default=SEMILLA,
                       help=f'Semilla aleatoria (default: {SEMILLA})')
    parser.add_argument('--no-plot', action='store_true',
                       help='No mostrar gráficas (solo guardar)')
    
    args = parser.parse_args()
    
    # Configurar semilla
    np.random.seed(args.semilla)
    
    print('='*70)
    print('OPTIMIZACIÓN DE RUTA DEL TREN')
    print('='*70)
    print(f'Cuadrícula:      {R}×{C}')
    print(f'Estaciones:      {args.n_estaciones}')
    print(f'Escenarios:      {args.num_escenarios}')
    print(f'Semilla:         {args.semilla}')
    print(f'Método:          {args.metodo}')
    print('='*70)
    
    # 1. Generar escenarios
    print('\n[1/4] GENERANDO ESCENARIOS DE POBLACIÓN...')
    escenarios = generar_escenarios(T, args.num_escenarios)
    print(f'       {len(escenarios)} escenarios generados')
    print(f'       Población total (escenario 0): {escenarios[0].sum():,}')
    
    # 2. Optimización
    resultados = {}
    
    if args.metodo in ['greedy', 'ambos']:
        print('\n[2/4] EJECUTANDO OPTIMIZACIÓN GREEDY...')
        print('-'*70)
        resultado_greedy = optimizar_ruta(
            evaluar_configuracion,
            escenarios,
            args.n_estaciones,
            metodo='greedy',
            verbose=True
        )
        resultados['Greedy'] = resultado_greedy
        print(f'\n       Costo final: {resultado_greedy["costo"]:.4f} celdas')
    
    if args.metodo in ['recocido', 'ambos']:
        print('\n[3/4] EJECUTANDO OPTIMIZACIÓN RECOCIDO SIMULADO...')
        print('-'*70)
        resultado_recocido = optimizar_ruta(
            evaluar_configuracion,
            escenarios,
            args.n_estaciones,
            metodo='recocido',
            T_inicial=10.0,
            T_final=0.01,
            alpha=0.95,
            max_iter=1000,
            verbose=True
        )
        resultados['Recocido Simulado'] = resultado_recocido
        print(f'\n       Costo final: {resultado_recocido["costo"]:.4f} celdas')
    
    # 3. Comparación y reporte
    print('\n[4/4] GENERANDO VISUALIZACIONES Y REPORTE')
    print('-'*70)
    
    if len(resultados) > 1:
        # Comparar métodos
        print('\nComparación de resultados:')
        print(f'{"Método":<25} {"Costo (celdas)":<15} {"Diferencia"}')
        print('-'*60)
        
        costos = [r['costo'] for r in resultados.values()]
        mejor_costo = min(costos)
        
        for nombre, res in resultados.items():
            diff = res['costo'] - mejor_costo
            diff_str = f'{diff:+.4f}' if diff != 0 else '(mejor)'
            print(f'{nombre:<25} {res["costo"]:<15.4f} {diff_str}')
        
        # Visualización comparativa
        fig, axes = comparar_metodos(resultados, escenarios[0])
        if not args.no_plot:
            plt.show()
        plt.savefig('resultados_comparacion.png', dpi=150, bbox_inches='tight')
        print('\n       Gráfica guardada: resultados_comparacion.png')
    
    # Elegir mejor resultado
    mejor_nombre = min(resultados.items(), key=lambda x: x[1]['costo'])[0]
    mejor_resultado = resultados[mejor_nombre]
    
    print(f'\n Mejor método: {mejor_nombre}')
    print(f'   Costo óptimo: {mejor_resultado["costo"]:.4f} celdas')
    print(f'\n   Estaciones:')
    for k, (i, j) in enumerate(mejor_resultado['estaciones']):
        print(f'     {k+1}. (fila={i:2d}, col={j:2d})')
    
    # Visualización de la mejor solución
    fig, ax = visualizar_ruta_sobre_poblacion(
        escenarios[0],
        mejor_resultado['estaciones'],
        titulo=f'Ruta Óptima ({mejor_nombre}) — Costo: {mejor_resultado["costo"]:.4f}'
    )
    if not args.no_plot:
        plt.show()
    plt.savefig('ruta_optima.png', dpi=150, bbox_inches='tight')
    print('\n       Gráfica guardada: ruta_optima.png')
    
    # Convergencia
    fig, ax = visualizar_convergencia(
        mejor_resultado['historial'],
        titulo=f'Convergencia ({mejor_nombre})'
    )
    if not args.no_plot:
        plt.show()
    plt.savefig('convergencia.png', dpi=150, bbox_inches='tight')
    print('\n       Gráfica guardada: convergencia.png')
    
    # Guardar resultados
    os.makedirs('data', exist_ok=True)
    np.save('data/estaciones_optimas.npy', mejor_resultado['estaciones'])
    print('\n       Estaciones guardadas: data/estaciones_optimas.npy')
    
    # Reporte de texto
    with open('data/reporte_optimizacion.txt', 'w', encoding='utf-8') as f:
        f.write('REPORTE DE OPTIMIZACIÓN DE RUTA DEL TREN\n')
        f.write('='*70 + '\n\n')
        f.write(f'CONFIGURACIÓN:\n')
        f.write(f'  Cuadrícula:      {R}×{C} ({R*C} nodos)\n')
        f.write(f'  Estaciones:      {args.n_estaciones}\n')
        f.write(f'  Escenarios:      {args.num_escenarios}\n')
        f.write(f'  Horizonte:       {T} años\n')
        f.write(f'  Semilla:         {args.semilla}\n\n')
        
        f.write(f'RESULTADOS:\n')
        for nombre, res in resultados.items():
            f.write(f'\n  {nombre}:\n')
            f.write(f'    Costo: {res["costo"]:.4f} celdas\n')
        
        f.write(f'\n  Mejor método: {mejor_nombre}\n')
        f.write(f'  Costo óptimo: {mejor_resultado["costo"]:.4f} celdas\n\n')
        
        f.write(f'ESTACIONES ÓPTIMAS (fila, columna):\n')
        for k, (i, j) in enumerate(mejor_resultado['estaciones']):
            f.write(f'  {k+1}. ({i:2d}, {j:2d})\n')
    
    print('       Reporte guardado: data/reporte_optimizacion.txt')
    
    print('\n' + '='*70)
    print('Optimización completada exitosamente')
    print('='*70)


if __name__ == '__main__':
    main()