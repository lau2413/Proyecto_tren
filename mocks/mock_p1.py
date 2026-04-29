import numpy as np
from config import R, C

def generar_escenarios(t, num_escenarios):
    """Escenarios sintéticos con población uniforme. Solo para desarrollo."""
    return [np.ones((R, C), dtype=int) * 10 for _ in range(num_escenarios)]