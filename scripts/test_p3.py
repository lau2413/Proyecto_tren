"""
Test rápido de la implementación de Persona 3

Verifica que todos los componentes funcionen correctamente
con un caso pequeño (ejecución rápida).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("="*70)
print("TEST RÁPIDO - Persona 3")
print("="*70)

# Test 1: Importaciones
print("\n[1/5] VERIFICANDO IMPORTACIONES")
try:
    from config import R, C, N_ESTACIONES
    from src.p1_ciudad import generar_escenarios
    from src.p2_demanda import evaluar_configuracion
    from src.p3_optimizacion import (
        optimizar_greedy,
        optimizar_recocido,
        optimizar_ruta,
        visualizar_ruta_sobre_poblacion,
        visualizar_convergencia,
        comparar_metodos
    )
    print("       Todas las importaciones exitosas")
except Exception as e:
    print(f"       Error en importaciones: {e}")
    sys.exit(1)

# Test 2: Generación de escenarios
print("\n[2/5] GENERANDO ESCENARIOS DE PRUEBA")
try:
    np.random.seed(42)
    escenarios_test = generar_escenarios(t=20, num_escenarios=5)
    assert len(escenarios_test) == 5
    assert escenarios_test[0].shape == (C, R)
    print(f"       5 escenarios generados correctamente")
    print(f"       Shape: {escenarios_test[0].shape}")
except Exception as e:
    print(f"       Error: {e}")
    sys.exit(1)

# Test 3: Optimización Greedy
print("\n[3/5] PROBANDO ALGORITMO GREEDY")
try:
    resultado_greedy = optimizar_ruta(
        evaluar_configuracion,
        escenarios_test,
        N=4,  # Menos estaciones para test rápido
        metodo='greedy',
        verbose=False
    )
    
    assert 'estaciones' in resultado_greedy
    assert 'costo' in resultado_greedy
    assert 'historial' in resultado_greedy
    assert resultado_greedy['estaciones'].shape == (4, 2)
    assert isinstance(resultado_greedy['costo'], float)
    assert resultado_greedy['costo'] > 0
    
    print(f"       Greedy ejecutado correctamente")
    print(f"       Costo: {resultado_greedy['costo']:.4f} celdas")
    print(f"       Estaciones encontradas: {len(resultado_greedy['estaciones'])}")
except Exception as e:
    print(f"       Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Optimización Recocido Simulado
print("\n[4/5] PROBANDO ALGORITMO RECOCIDO SIMULADO")
try:
    resultado_recocido = optimizar_ruta(
        evaluar_configuracion,
        escenarios_test,
        N=4,
        metodo='recocido',
        max_iter=100,  # Pocas iteraciones para test rápido
        verbose=False
    )
    
    assert 'estaciones' in resultado_recocido
    assert 'costo' in resultado_recocido
    assert 'historial' in resultado_recocido
    assert resultado_recocido['estaciones'].shape == (4, 2)
    assert isinstance(resultado_recocido['costo'], float)
    assert resultado_recocido['costo'] > 0
    
    print(f"       Recocido Simulado ejecutado correctamente")
    print(f"       Costo: {resultado_recocido['costo']:.4f} celdas")
    print(f"       Iteraciones: {len(resultado_recocido['historial'])}")
except Exception as e:
    print(f"       Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Comparación de resultados
print("\n[5/5] COMPARANDO RESULTADOS")
try:
    print(f"      * Greedy:            {resultado_greedy['costo']:.4f} celdas")
    print(f"      * Recocido Simulado: {resultado_recocido['costo']:.4f} celdas")
    
    diferencia = abs(resultado_greedy['costo'] - resultado_recocido['costo'])
    print(f"      * Diferencia:        {diferencia:.4f} celdas")
    
    mejor = "Greedy" if resultado_greedy['costo'] < resultado_recocido['costo'] else "Recocido"
    print(f"       Mejor método en este test: {mejor}")
    
    # Verificar que ambos mejoran sobre una solución trivial
    # (una sola estación en el centro)

    estacion_centro = np.array([[R//2, C//2]])
    costo_trivial = evaluar_configuracion(escenarios_test, estacion_centro)
    
    assert resultado_greedy['costo'] < costo_trivial, "Greedy debe mejorar sobre solución trivial"
    assert resultado_recocido['costo'] < costo_trivial, "Recocido debe mejorar sobre solución trivial"
    
    print(f"       Ambos métodos mejoran sobre solución trivial ({costo_trivial:.4f})")
    
except Exception as e:
    print(f"       Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Resumen final
print("\n" + "="*70)
print(" TODOS LOS TESTS PASARON EXITOSAMENTE")
print("="*70)
print("\nPróximos pasos:")
print("  1. Ejecutar con configuración completa:")
print("     python scripts/run_p3.py")
print("\n  2. O explorar el notebook interactivo:")
print("     jupyter notebook notebooks/p3_optimizacion_visualizacion.ipynb")
print("\n" + "="*70)