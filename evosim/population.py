"""Población diploide de tamaño constante con reproducción sexual (Wright-Fisher).

Cada generación:
  1. Se evalúa el fitness de cada individuo (genotipo -> fenotipo -> ambiente).
  2. Cada hijo tiene una madre y un padre elegidos con probabilidad proporcional
     al fitness (selección). El muestreo finito es la deriva genética.
  3. Cada padre aporta un gameto producido por meiosis (recombinación).
  4. Los gametos mutan.
Organismos hermafroditas; la autofecundación ocurre al azar salvo que se desactive.
"""
import numpy as np

from .meiosis import make_gametes
from .mutation import mutate


class Population:
    def __init__(self, arch, N, rng, genomes=None, environment=None, selfing=True):
        self.arch = arch
        self.N = N
        self.rng = rng
        self.environment = environment
        self.selfing = selfing
        if genomes is None:
            genomes = np.broadcast_to(arch.ancestral, (N, 2, arch.layout.L)).copy()
        if genomes.shape != (N, 2, arch.layout.L):
            raise ValueError(f"genomes debe tener forma {(N, 2, arch.layout.L)}")
        self.genomes = genomes.astype(np.uint8, copy=False)
        self.generation = 0
        self.last_log_fitness = None
        self.last_traits = None

    def evaluate(self):
        logw, traits = self.arch.gmap.evaluate(self.genomes, self.rng)
        if self.environment is not None and traits is not None:
            logw = logw + self.environment.log_fitness(traits)
        return logw, traits

    def step(self):
        logw, traits = self.evaluate()
        self.last_log_fitness, self.last_traits = logw, traits
        w = np.exp(logw - logw.max())
        p = w / w.sum()
        N, rng = self.N, self.rng
        mothers = rng.choice(N, N, p=p)
        fathers = rng.choice(N, N, p=p)
        if not self.selfing:
            same = mothers == fathers
            while same.any():
                fathers[same] = rng.choice(N, same.sum(), p=p)
                same = mothers == fathers
        eggs = make_gametes(self.genomes, mothers, self.arch.layout, rng)
        sperm = make_gametes(self.genomes, fathers, self.arch.layout, rng)
        mutate(eggs, self.arch.mutation_rate, rng, self.arch.kappa)
        mutate(sperm, self.arch.mutation_rate, rng, self.arch.kappa)
        self.genomes = np.stack([eggs, sperm], axis=1)
        self.generation += 1

    def run(self, generations, callback=None, every=1):
        for _ in range(generations):
            self.step()
            if callback is not None and self.generation % every == 0:
                if callback(self) is False:
                    break
