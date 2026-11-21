"""Mutación puntual con el modelo de sustitución de Kimura de 2 parámetros (K80).

Codificación de bases: A=0, C=1, G=2, T=3.
- Transiciones (purina<->purina, pirimidina<->pirimidina): A<->G, C<->T  ==  b ^ 2
- Transversiones: las otras dos bases                                   ==  b ^ 1, b ^ 3
kappa = tasa de transición / tasa de cada transversión. kappa = 1 es Jukes-Cantor.
En organismos reales kappa suele estar entre 2 y 10.
"""
import numpy as np


def mutate(gametes, mu, rng, kappa=1.0):
    """Muta in-place un arreglo (n, L) uint8. mu = tasa por sitio por generación.
    Devuelve el número de mutaciones introducidas."""
    if mu <= 0:
        return 0
    flat = gametes.reshape(-1)
    k = rng.poisson(mu * flat.size)
    if k == 0:
        return 0
    pos = rng.integers(0, flat.size, size=k)
    old = flat[pos]
    is_transition = rng.random(k) < kappa / (kappa + 2.0)
    tv_mask = np.where(rng.integers(0, 2, size=k).astype(bool), 3, 1).astype(np.uint8)
    flat[pos] = np.where(is_transition, old ^ 2, old ^ tv_mask)
    return k
