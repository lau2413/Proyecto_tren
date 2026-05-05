import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from config import T, NUM_ESCENARIOS, SEMILLA
from src.p1_ciudad import generar_escenarios, pr, _ii, _jj

np.random.seed(SEMILLA)

# 1. verificar que pr suma 1 en algunos nodos
for (i, j) in [(10, 10), (5, 10), (4, 5)]:
    total = pr(i, j, T, np.arange(300)).sum()
    print(f"pr suma en ({i},{j}): {total:.4f}")

# 2. generar escenarios
escenarios = generar_escenarios(T, NUM_ESCENARIOS)
print(f"Escenarios generados: {len(escenarios)}, shape: {escenarios[0].shape}")

# 3. visualizar un escenario
plt.figure(figsize=(7, 6))
plt.contourf(_ii, _jj, escenarios[0], 50)
plt.colorbar()
plt.scatter(_ii, _jj, marker='.', color='r', s=8)
plt.title("Escenario 1 — distribución de población")
plt.xlabel("i")
plt.ylabel("j")
plt.tight_layout()
plt.show()

# 4. visualizar variabilidad entre escenarios
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for k, ax in enumerate(axes):
    cf = ax.contourf(_ii, _jj, escenarios[k], 50)
    plt.colorbar(cf, ax=ax)
    ax.set_title(f"Escenario {k+1}")
plt.tight_layout()
plt.show()