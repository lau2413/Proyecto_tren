# Proyecto_tren - guía de integración

##  Estructura del Proyecto

```
proyecto_tren/
│
├── config.py                    # Configuración global
├── requirements.txt             # Dependencias Python
│
├── src/                         # Módulos principales
│   ├── p1_ciudad.py            # Persona 1: Generación de escenarios
│   ├── p2_demanda.py           # Persona 2: Cálculo de demanda y evaluación
│   └── p3_optimizacion.py      # Persona 3: Optimización y visualización
│
├── mocks/                       # Mocks para desarrollo
│   └── mock_p1.py
│
├── scripts/                     # Scripts ejecutables
│   ├── run_p1.py               # Ejecutar persona 1
│   ├── run_p3.py               # Ejecutar persona 3
│   └── test_p3.py              # Test rápido persona 3
│
├── notebooks/                   # Notebooks de demostración
│   ├── informacion_base_modelo_de_prescriptiva.ipynb
│   ├── p2_demanda_evaluacion.ipynb
│
├── scripts/data/                        # Resultados (generado al ejecutar)
│   ├── estaciones_optimas.npy
│   └── reporte_optimizacion.txt
│
└── README.md          # Documentación proyecto
```

## Flujo de datos

```
┌─────────────────┐
│   PERSONA 1     │
│  p1_ciudad.py   │
│                 │
│ generar_        │
│ escenarios()    │
└────────┬────────┘
         │
         │ escenarios: list[ndarray(C,R)]
         │
         ▼
┌─────────────────┐
│   PERSONA 2     │
│  p2_demanda.py  │
│                 │
│ evaluar_        │
│ configuracion() │
└────────┬────────┘
         │
         │ función: (escenarios, estaciones) → costo
         │
         ▼
┌─────────────────┐
│   PERSONA 3     │
│p3_optimizacion.py│
│                 │
│ optimizar_ruta()│
│                 │
│ visualizar_*()  │
└─────────────────┘
         │
         ▼
    Resultados:
    * estaciones_optimas.npy
    * gráficas PNG
    * reporte TXT
```

## Cómo ejecutar el proyecto completo

### Paso 1: Preparar el entorno

```bash
# Instalar dependencias
pip install -r requirements.txt

# Verificar estructura
python scripts/test_p3.py
```

### Paso 2: Ejecutar optimización

```bash
# Opción A: Script completo
python scripts/run_p3.py

# Opción B: Solo greedy
python scripts/run_p3.py --metodo greedy

# Opción C: Personalizado
python scripts/run_p3.py --n_estaciones 8 --num_escenarios 50
```

### Paso 3: Analizar resultados

Los resultados se guardan automáticamente en:
- `data/estaciones_optimas.npy` - Coordenadas de estaciones
- `data/reporte_optimizacion.txt` - Reporte textual
- `ruta_optima.png` - Visualización de la ruta
- `convergencia.png` - Curva de convergencia
- `resultados_comparacion.png` - Comparación de métodos


### Ejemplo completo de integración

```python
import numpy as np
from config import R, C, N_ESTACIONES, T, NUM_ESCENARIOS, SEMILLA
from src.p1_ciudad import generar_escenarios
from src.p2_demanda import evaluar_configuracion
from src.p3_optimizacion import optimizar_ruta, visualizar_ruta_sobre_poblacion

# Configurar semilla
np.random.seed(SEMILLA)

# PERSONA 1: Generar escenarios de población
print("Generando escenarios...")
escenarios = generar_escenarios(T, NUM_ESCENARIOS)
print(f" {len(escenarios)} escenarios generados")

# PERSONA 2: La función de evaluación ya está lista
# evaluar_configuracion(escenarios, estaciones) → costo

# PERSONA 3: Optimizar la ruta
print("\nOptimizando ruta...")
resultado = optimizar_ruta(
    evaluar_fn=evaluar_configuracion,
    escenarios=escenarios,
    N=N_ESTACIONES,
    metodo='greedy',  # o 'recocido'
    verbose=True
)

print(f"\n Optimización completada")
print(f"  Costo óptimo: {resultado['costo']:.4f} celdas")
print(f"  Estaciones:")
for k, (i, j) in enumerate(resultado['estaciones']):
    print(f"    {k+1}. (fila={i}, col={j})")

# Visualizar
import matplotlib.pyplot as plt
visualizar_ruta_sobre_poblacion(escenarios[0], resultado['estaciones'])
plt.savefig('mi_ruta.png')
plt.show()
```

##  API persona 3

### Función principal

```python
optimizar_ruta(evaluar_fn, escenarios, N, metodo='greedy', **kwargs)
```

**Parámetros:**
- `evaluar_fn`: función de Persona 2
- `escenarios`: lista de escenarios de Persona 1
- `N`: número de estaciones
- `metodo`: 'greedy' o 'recocido'
- `**kwargs`: parámetros adicionales (verbose, T_inicial, etc.)

**Retorna:**
```python
{
    'estaciones': ndarray(N, 2),  # Coordenadas (fila, col)
    'costo': float,                # Costo promedio en celdas
    'historial': list,             # Evolución del costo
    'metodo': str                  # Método utilizado
}
```

### Funciones de visualización

```python
# 1. Ruta sobre mapa de población
visualizar_ruta_sobre_poblacion(escenario, estaciones, titulo="...")

# 2. Curva de convergencia
visualizar_convergencia(historial, titulo="...")

# 3. Comparación de métodos
comparar_metodos(resultados_dict, escenario_ref)
```

### Integración conjunta

| Persona | Produce | Formato | Consume en P3 |
|---------|---------|---------|---------------|
| 1 | Escenarios de población | `list[ndarray(C,R)]` | Input para optimización |
| 2 | Función de evaluación | `(escenarios, estaciones) → float` | Función objetivo |
| 3 | Ruta óptima | `ndarray(N, 2)` | — |

1. **Decisiones de diseño**:
   - Greedy comienza con estación individual óptima (no vacío)
   - Recocido usa movimientos: reemplazo, desplazamiento adyacente
   - Visualizaciones incluyen múltiples escenarios para robustez

2. **Resultados típicos** (R=20, C=20, N=6):
   - Greedy: aprox 3.8 celdas
   - Recocido: aprox 3.7 celdas
   - Mejora vs 1 estación central: aprox 50%

3. **Análisis de sensibilidad**:
   - Más estaciones → menor costo (rendimientos decrecientes)
   - Punto óptimo típico: 6-8 estaciones para cuadrícula 20×20

4. **Limitaciones y mejoras futuras**:
   - No se consideran restricciones de presupuesto
   - Ruta es lista ordenada, no necesariamente ciclo óptimo
   - Posibles extensiones: algoritmos genéticos, búsqueda tabú