"""Ambientes: convierten rasgos (fenotipo) en un componente de fitness.

Por ahora solo selección estabilizadora gaussiana. Más adelante el mundo
ecológico reemplazará esto con supervivencia y reproducción explícitas.
"""
import numpy as np


class StabilizingSelection:
    """w(z) = exp(-sum((z - optimo)^2 / (2 omega^2))). Omega grande = selección débil."""

    def __init__(self, optimum, omega):
        self.optimum = np.asarray(optimum, dtype=float)
        self.omega = np.broadcast_to(np.asarray(omega, dtype=float), self.optimum.shape)

    def log_fitness(self, traits):
        return -(((traits - self.optimum) / self.omega) ** 2).sum(axis=1) / 2.0
