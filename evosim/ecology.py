"""Mundo ecológico: paisaje espacial, regulación por densidad, apareamiento local y dispersión.

A diferencia de Population (Wright-Fisher, N fijo), aquí el tamaño poblacional
es una consecuencia de la ecología: puede crecer, contraerse o extinguirse.

Ciclo de vida por generación (generaciones discretas, organismos anuales):
  1. Fitness individual w = fitness intrínseco (genoma) × ajuste al ambiente local.
  2. Fecundidad esperada (Beverton-Holt por celda):
         E[hijos_i] = λ · w_i / (1 + (λ - 1) · n_celda / K_celda)
     Con w = 1, la celda se estabiliza en n = K.
  3. Número de hijos ~ Poisson (estocasticidad demográfica).
  4. Cada hijo tiene como madre al individuo i y un padre elegido dentro de la
     misma celda con probabilidad ∝ w (selección también por la vía masculina).
  5. Meiosis + mutación.
  6. Dispersión de los hijos.
  7. Los padres mueren (generaciones no solapadas).
"""
from dataclasses import dataclass, field

import numpy as np

from .meiosis import make_gametes
from .mutation import mutate


class Landscape:
    """Cuadrícula de height × width celdas con capas ambientales y capacidad de carga."""

    def __init__(self, width, height, K, layers=None):
        self.width, self.height = int(width), int(height)
        self.n_cells = self.width * self.height
        self.K = np.broadcast_to(np.asarray(K, dtype=float), (self.height, self.width)).astype(float).ravel()
        self.layers = {name: np.broadcast_to(np.asarray(v, float), (self.height, self.width)).astype(float).ravel()
                       for name, v in (layers or {}).items()}
        self.x = np.tile(np.arange(self.width), self.height)
        self.y = np.repeat(np.arange(self.height), self.width)
        self._updaters = []

    def cell(self, x, y):
        return np.asarray(y) * self.width + np.asarray(x)

    def grid(self, values):
        return np.asarray(values).reshape(self.height, self.width)

    def add_updater(self, fn):
        """fn(landscape, generation) se llama al inicio de cada generación (p. ej. cambio climático)."""
        self._updaters.append(fn)

    def update(self, generation):
        for fn in self._updaters:
            fn(self, generation)

    @staticmethod
    def gradient(width, height, low, high, axis="x"):
        """Capa con un gradiente lineal (p. ej. temperatura de sur a norte)."""
        if axis == "x":
            row = np.linspace(low, high, width)
            return np.tile(row, (height, 1))
        col = np.linspace(low, high, height)
        return np.tile(col[:, None], (1, width))


@dataclass
class NicheAxis:
    """Un rasgo debe coincidir con una variable ambiental: w *= exp(-(z - E)^2 / 2ω^2)."""
    trait: int
    layer: str
    omega: float


@dataclass
class EcoParams:
    fecundity: float = 2.0                 # λ: hijos por individuo con w = 1 y sin competencia
    niche: list = field(default_factory=list)   # [NicheAxis, ...]
    dispersal: str = "stepping_stone"      # "stepping_stone" | "island" | "gaussian" | "none"
    migration_rate: float = 0.05           # m: prob. de que un hijo deje su celda (stepping_stone / island)
    dispersal_sigma: float = 1.0           # σ en celdas (gaussian)
    selfing: bool = True                   # False: autofecundación imposible (un individuo solo no se reproduce)


class EcoPopulation:
    def __init__(self, arch, landscape, rng, genomes, cells, params=None):
        self.arch = arch
        self.landscape = landscape
        self.rng = rng
        self.params = params or EcoParams()
        self.genomes = np.asarray(genomes, dtype=np.uint8)
        self.cells = np.asarray(cells, dtype=np.int64)
        if self.genomes.shape[0] != len(self.cells):
            raise ValueError("genomes y cells deben tener el mismo número de individuos")
        self.generation = 0
        self.last_w = None
        self.last_traits = None

    @classmethod
    def founders(cls, arch, landscape, rng, cells, params=None):
        """Población inicial con genomas ancestrales en las celdas indicadas (una entrada por individuo)."""
        cells = np.asarray(cells, dtype=np.int64)
        g = np.broadcast_to(arch.ancestral, (len(cells), 2, arch.layout.L)).copy()
        return cls(arch, landscape, rng, g, cells, params)

    @property
    def N(self):
        return len(self.cells)

    def cell_counts(self):
        return np.bincount(self.cells, minlength=self.landscape.n_cells)

    # ------------------------------------------------------------------
    def fitness(self):
        logw, traits = self.arch.gmap.evaluate(self.genomes, self.rng)
        for ax in self.params.niche:
            env = self.landscape.layers[ax.layer][self.cells]
            logw = logw - (traits[:, ax.trait] - env) ** 2 / (2 * ax.omega ** 2)
        return np.exp(logw), traits

    def _choose_fathers(self, mothers, w):
        """Padre dentro de la celda de la madre, con probabilidad ∝ w."""
        rng = self.rng
        order = np.argsort(self.cells, kind="stable")
        sorted_cells = self.cells[order]
        cw = np.cumsum(w[order])
        c = self.cells[mothers]
        start = np.searchsorted(sorted_cells, c, side="left")
        end = np.searchsorted(sorted_cells, c, side="right")
        before = np.where(start > 0, cw[np.maximum(start - 1, 0)], 0.0)
        total = cw[end - 1] - before

        def draw(k_idx):
            target = before[k_idx] + rng.random(len(k_idx)) * total[k_idx]
            pos = np.searchsorted(cw, target, side="right")
            return order[np.clip(pos, start[k_idx], end[k_idx] - 1)]

        fathers = draw(np.arange(len(mothers)))
        keep = np.ones(len(mothers), bool)
        if not self.params.selfing:
            for _ in range(10):
                bad = np.flatnonzero(fathers == mothers)
                if len(bad) == 0:
                    break
                fathers[bad] = draw(bad)
            keep = fathers != mothers
        return fathers, keep

    def _disperse(self, cells):
        p, L, rng = self.params, self.landscape, self.rng
        n = len(cells)
        if p.dispersal == "none" or n == 0:
            return cells
        x, y = L.x[cells].copy(), L.y[cells].copy()
        if p.dispersal == "gaussian":
            x = np.clip(np.rint(x + rng.normal(0, p.dispersal_sigma, n)), 0, L.width - 1).astype(int)
            y = np.clip(np.rint(y + rng.normal(0, p.dispersal_sigma, n)), 0, L.height - 1).astype(int)
            return L.cell(x, y)
        move = np.flatnonzero(rng.random(n) < p.migration_rate)
        if p.dispersal == "island":
            if L.n_cells < 2:
                return cells
            other = rng.integers(0, L.n_cells - 1, len(move))
            other += other >= cells[move]           # celda distinta a la de origen
            out = cells.copy()
            out[move] = other
            return out
        if p.dispersal == "stepping_stone":
            dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) if L.height > 1 else np.array([[1, 0], [-1, 0]])
            d = dirs[rng.integers(0, len(dirs), len(move))]
            nx, ny = x[move] + d[:, 0], y[move] + d[:, 1]
            inside = (nx >= 0) & (nx < L.width) & (ny >= 0) & (ny < L.height)   # en el borde, se queda
            x[move[inside]], y[move[inside]] = nx[inside], ny[inside]
            return L.cell(x, y)
        raise ValueError(f"dispersión desconocida: {p.dispersal}")

    def step(self):
        self.landscape.update(self.generation)
        if self.N == 0:
            self.generation += 1
            return
        p, rng, L = self.params, self.rng, self.landscape
        w, traits = self.fitness()
        self.last_w, self.last_traits = w, traits

        n_c = self.cell_counts()[self.cells]
        K = L.K[self.cells]
        density = 1.0 + (p.fecundity - 1.0) * n_c / np.maximum(K, 1e-12)
        expected = np.where(K > 0, p.fecundity * w / density, 0.0)
        n_off = rng.poisson(expected)
        mothers = np.repeat(np.arange(self.N), n_off)
        if len(mothers) == 0:
            self.genomes = self.genomes[:0]
            self.cells = self.cells[:0]
            self.generation += 1
            return
        fathers, keep = self._choose_fathers(mothers, w)
        mothers, fathers = mothers[keep], fathers[keep]

        eggs = make_gametes(self.genomes, mothers, self.arch.layout, rng)
        sperm = make_gametes(self.genomes, fathers, self.arch.layout, rng)
        mutate(eggs, self.arch.mutation_rate, rng, self.arch.kappa)
        mutate(sperm, self.arch.mutation_rate, rng, self.arch.kappa)
        self.genomes = np.stack([eggs, sperm], axis=1)
        self.cells = self._disperse(self.cells[mothers])
        self.generation += 1

    def run(self, generations, callback=None, every=1):
        for _ in range(generations):
            self.step()
            if callback is not None and self.generation % every == 0:
                if callback(self) is False:
                    break
            if self.N == 0:
                break
