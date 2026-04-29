import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from config import R, C

# meshgrid compartido
_ii, _jj = np.meshgrid(np.arange(R), np.arange(C))

# funciones del profesor

def pr(i, j, t, n):
    rate = np.sin(2 * np.pi * i / 20) * np.sin(2 * np.pi * j / 15) * 2 + 2
    media = rate * t
    return poisson.pmf(n, mu=media)

def gen(i, j, t):
    rate = np.sin(2 * np.pi * i / 20) * np.sin(2 * np.pi * j / 15) * 2 + 4
    media = rate * t
    return np.random.poisson(lam=media)


def generar_escenarios(t, num_escenarios):
    escenarios = []
    for _ in range(num_escenarios):
        pob = gen(_ii, _jj, t).astype(int)
        escenarios.append(pob)
    return escenarios