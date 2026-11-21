"""Herramientas para estudiar especiación: cruzas híbridas, aislamiento reproductivo,
incompatibilidades de Dobzhansky-Muller y radiaciones de linajes aislados.

Nada de esto altera la dinámica: son "experimentos de laboratorio" sobre genomas que
evolucionaron por separado (cruzas controladas, introgresión de alelos).
"""
from dataclasses import dataclass, field

import numpy as np

from .architecture import INTERGENIC
from .meiosis import make_gametes
from .population import Population
from .stats import consensus


# ----------------------------------------------------------------------------- fitness
def log_fitness(arch, genomes, environment=None):
    """Fitness de genomas arbitrarios (sin ruido ambiental, para comparar clases de cruza)."""
    logw, traits = arch.gmap.evaluate(genomes, None)
    if environment is not None and traits is not None:
        logw = logw + environment.log_fitness(traits)
    return logw


# ----------------------------------------------------------------------------- cruzas
def cross(mothers, fathers, layout, rng, n):
    """n hijos: madre al azar de `mothers`, padre al azar de `fathers` (arrays (N,2,L))."""
    eggs = make_gametes(mothers, rng.integers(0, len(mothers), n), layout, rng)
    sperm = make_gametes(fathers, rng.integers(0, len(fathers), n), layout, rng)
    return np.stack([eggs, sperm], axis=1)


def hybrid_classes(ga, gb, layout, rng, n=400):
    """Parentales, F1, F2 y retrocruzas, cada clase con n individuos."""
    f1 = cross(ga, gb, layout, rng, n)
    return {
        "A": cross(ga, ga, layout, rng, n),
        "B": cross(gb, gb, layout, rng, n),
        "F1": f1,
        "F2": cross(f1, f1, layout, rng, n),
        "BC_A": cross(f1, ga, layout, rng, n),
        "BC_B": cross(f1, gb, layout, rng, n),
    }


def hybrid_fitness(arch, ga, gb, rng, environment=None, n=400):
    """Fitness medio de cada clase de cruza (no logarítmico)."""
    classes = hybrid_classes(ga, gb, arch.layout, rng, n)
    return {k: float(np.exp(log_fitness(arch, g, environment)).mean()) for k, g in classes.items()}


def isolation_index(hf, hybrid="F2"):
    """Aislamiento postcigótico RI = 1 − w_híbrido / w_parental (Coyne & Orr 1989).
    0 = sin aislamiento; 1 = híbridos completamente inviables/estériles."""
    parental = 0.5 * (hf["A"] + hf["B"])
    return 1.0 - hf[hybrid] / parental


# ----------------------------------------------------------------------------- DMIs
def _major_freq(genomes, cons):
    return (genomes == cons).mean(axis=(0, 1))


def fixed_differences(ga, gb, sites=None, min_freq=0.9):
    """Sitios donde las poblaciones tienen alelos mayoritarios distintos, cada uno con
    frecuencia ≥ min_freq en su población (diferencias fijadas o casi fijadas)."""
    ca, cb = consensus(ga), consensus(gb)
    ok = (ca != cb) & (_major_freq(ga, ca) >= min_freq) & (_major_freq(gb, cb) >= min_freq)
    diff = np.flatnonzero(ok)
    if sites is not None:
        diff = np.intersect1d(diff, sites)
    return diff, ca, cb


def count_dmis(arch, ga, gb, environment=None, tol=0.005, functional_only=True, min_freq=0.9):
    """Cuenta incompatibilidades de Dobzhansky-Muller entre dos poblaciones.

    Definición operacional (como en Orr 1995 y estudios de introgresión): sobre el fondo
    homocigoto de la población X, un par de diferencias fijadas (i, j) es una DMI si
    introducir el alelo de Y en j es deletéreo (log w cae más de `tol`) y además
    introducir también el alelo de Y en i lo rescata (epistasis positiva > tol):
    el alelo de Y en j solo es malo junto al alelo de X en i.
    Se cuenta en ambos fondos (X=A, X=B). Devuelve (n_DMIs, K diferencias, detalles).

    Con environment=None se mide solo el fitness intrínseco (viabilidad molecular). Con un
    ambiente de selección estabilizadora, cualquier par de alelos de efectos opuestos sobre el
    rasgo "se rescata" mutuamente (modelo geométrico de Fisher) y el conteo por pares pierde
    sentido; para eso conviene medir el aislamiento con hybrid_fitness.
    """
    sites = np.flatnonzero(arch.site_class != INTERGENIC) if functional_only else None
    diff, ca, cb = fixed_differences(ga, gb, sites, min_freq)
    K = len(diff)
    pairs = []
    if K == 0:
        return 0, 0, pairs
    for X, Y, label in ((ca, cb, "A"), (cb, ca, "B")):
        base = np.broadcast_to(X, (1, 2, len(X)))
        w0 = log_fitness(arch, base.copy(), environment)[0]
        singles = np.broadcast_to(X, (K, 2, len(X))).copy()
        singles[np.arange(K), :, diff] = Y[diff][:, None]
        w1 = log_fitness(arch, singles, environment) - w0                  # efecto de cada introgresión
        bad = np.flatnonzero(w1 < -tol)
        for j in bad:
            g = np.broadcast_to(singles[j], (K, 2, len(X))).copy()
            g[np.arange(K), :, diff] = Y[diff][:, None]
            w2 = log_fitness(arch, g, environment) - w0                    # doble introgresión (i, j)
            eps = w2 - w1 - w1[j]
            for i in np.flatnonzero((eps > tol) & (np.arange(K) != j)):
                pairs.append((label, int(diff[i]), int(diff[j]), float(w1[j]), float(eps[i])))
    # un mismo par puede detectarse en ambos fondos; se cuenta una vez
    unique = {tuple(sorted((p[1], p[2]))) for p in pairs}
    return len(unique), K, pairs


# ----------------------------------------------------------------------------- radiación
@dataclass
class Lineage:
    name: str
    pop: Population
    born: int
    parent: str = None
    children: list = field(default_factory=list)
    died: int = None


class Radiation:
    """Linajes en alopatría: una población ancestral que se divide (vicarianza) en
    momentos dados. Sin flujo génico entre linajes; cada uno evoluciona por su cuenta.
    Se registra el árbol verdadero para compararlo con la filogenia reconstruida.
    """

    def __init__(self, arch, N, rng, environment=None, genomes=None, selfing=False):
        self.arch, self.N, self.rng, self.environment = arch, N, rng, environment
        self.selfing = selfing
        self.generation = 0
        root = Population(arch, N, rng, genomes=genomes, environment=environment, selfing=selfing)
        self.lineages = {"L0": Lineage("L0", root, 0)}
        self._next = 1

    @property
    def tips(self):
        return [l for l in self.lineages.values() if not l.children and l.died is None]

    def split(self, name):
        """Divide el linaje en dos poblaciones hijas, cada una fundada con N individuos
        muestreados de la madre (sin cuello de botella fuerte)."""
        parent = self.lineages[name]
        names = []
        for _ in range(2):
            idx = self.rng.integers(0, self.N, self.N)
            g = parent.pop.genomes[idx].copy()
            child = f"L{self._next}"
            self._next += 1
            pop = Population(self.arch, self.N, self.rng, genomes=g,
                             environment=self.environment, selfing=self.selfing)
            pop.generation = self.generation
            self.lineages[child] = Lineage(child, pop, self.generation, parent=name)
            parent.children.append(child)
            names.append(child)
        return names

    def step(self):
        for l in self.tips:
            l.pop.step()
        self.generation += 1

    def run(self, generations, callback=None, every=1):
        for _ in range(generations):
            self.step()
            if callback is not None and self.generation % every == 0:
                callback(self)

    def true_tree(self):
        """Árbol verdadero en Newick, longitudes de rama en generaciones."""
        def rec(name):
            l = self.lineages[name]
            end = self.generation if not l.children else self.lineages[l.children[0]].born
            length = end - l.born
            if not l.children:
                return f"{name}:{length}"
            return "(" + ",".join(rec(c) for c in l.children) + f"):{length}"
        return rec("L0") + ";"
