import numpy as np
import matplotlib.pyplot as plt
from config import T, NUM_ESCENARIOS, N_ESTACIONES, SEMILLA
from src.p1_ciudad import generar_escenarios
from src.p2_demanda import evaluar_configuracion
from src.p3_optimizacion import optimizar_ruta, visualizar_ruta_sobre_poblacion

def main():
    # Semilla de reproducibilidad
    np.random.seed(SEMILLA)
    print(f"Starting Route Optimization Pipeline (Seed: {SEMILLA})")
    print(f"Config: {NUM_ESCENARIOS} scenarios, {N_ESTACIONES} stations\n")

    # 1. Generación de escenarios 
    print("Step 1: Generating population scenarios...")
    escenarios = generar_escenarios(T, NUM_ESCENARIOS)
    print(f"Generated {len(escenarios)} scenarios.\n")

    # 2. Optimización de la ruta 
    # Pasamos evaluar_configuracion como función de evaluación a optimizar_ruta
    print("Step 2: Optimizing route (this may take a moment)...")
    resultado = optimizar_ruta(
        evaluar_fn=evaluar_configuracion,
        escenarios=escenarios,
        N=N_ESTACIONES,
        metodo='greedy',
        verbose=True
    )

    # Imprimir resultados de optimización
    print("\n" + "="*30)
    print("OPTIMIZATION COMPLETE")
    print(f"Method: {resultado['metodo']}")
    print(f"Optimal Cost: {resultado['costo']:.4f} cells")
    print("Optimal Station Locations:")
    for k, pos in enumerate(resultado['estaciones']):
        print(f"  Station {k+1}: Row {pos[0]}, Col {pos[1]}")
    print("="*30 + "\n")

    # 3. Visualización de resultados
    print("Step 3: Visualizing results...")
    # Visualize on the first scenario as a representative example
    fig, ax = visualizar_ruta_sobre_poblacion(
        escenario=escenarios[0],
        estaciones=resultado['estaciones'],
        titulo=f"Ruta Óptima ({resultado['metodo'].capitalize()})"
    )

    plt.savefig('main_result.png')
    print("Result saved to main_result.png")
    plt.show()

if __name__ == "__main__":
    main()
