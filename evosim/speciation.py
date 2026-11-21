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


# Mecanismos de una DMI (según dónde caen los dos sitios del par)
TF_SITE = "tf_sitio"          # hélice de reconocimiento de un TF × sitio de unión que ese TF regula
REGULATORY = "regulatoria"    # otro par dentro de un módulo TF→blanco (actividad del TF, gen blanco, etc.)
INTRAGENIC = "intragenica"    # ambos sitios en el mismo gen (codificante/promotor): epistasis de flujo o expresión
OTHER = "otra"                # genes distintos sin relación regulatoria (p. ej. vía rasgos)
MECHANISMS = (TF_SITE, REGULATORY, INTRAGENIC, OTHER)


def site_annotation(arch):
    """Para cada sitio del genoma: (gen al que pertenece o −1, tipo).
    Tipos: "codificante", "reconocimiento" (hélice de un TF), "promotor", "sitio_union", "" (intergénico)."""
    gm = arch.gmap
    L = arch.layout.L
    gene = np.full(L, -1)
    kind = np.full(L, "", dtype=object)
    for g, gn in enumerate(getattr(gm, "genes", [])):
        gene[gn.coding_sites] = g
        kind[gn.coding_sites] = "codificante"
        if len(gn.promoter_sites):
            gene[gn.promoter_sites] = g
            kind[gn.promoter_sites] = "promotor"
    for t, rs in getattr(gm, "_recog_sites", {}).items():
        kind[rs.ravel()] = "reconocimiento"
    for r in getattr(gm, "regulations", []):
        gene[r.site] = r.target
        kind[r.site] = "sitio_union"
    return gene, kind


def dmi_mechanism(arch, i, j, annotation=None):
    """Clasifica el par de sitios (i, j) de una DMI por mecanismo molecular."""
    gene, kind = annotation if annotation is not None else site_annotation(arch)
    regs = getattr(arch.gmap, "regulations", [])
    gi, gj = gene[i], gene[j]
    for r in regs:
        for a, b in ((i, j), (j, i)):
            if kind[a] == "reconocimiento" and gene[a] == r.tf and kind[b] == "sitio_union" and gene[b] == r.target:
                return TF_SITE
    for r in regs:
        if {gi, gj} <= {r.tf, r.target} and (gi != gj or "sitio_union" in (kind[i], kind[j])
                                             or "reconocimiento" in (kind[i], kind[j])):
            return REGULATORY
    if gi == gj and gi >= 0:
        return INTRAGENIC
    return OTHER


def dmi_mechanisms(arch, pairs):
    """Cuenta los pares únicos de `count_dmis` por mecanismo: {mecanismo: n}."""
    ann = site_annotation(arch)
    unique = {tuple(sorted((p[1], p[2]))) for p in pairs}
    counts = dict.fromkeys(MECHANISMS, 0)
    for i, j in unique:
        counts[dmi_mechanism(arch, i, j, ann)] += 1
    return counts


def _introgression_units(arch, diff):
    """Agrupa las diferencias fijadas en unidades de introgresión: todas las diferencias de un
    mismo codón van juntas (en un híbrido la recombinación casi nunca parte un codón, así que
    los codones quiméricos no aparecen); cada sitio no codificante es su propia unidad."""
    gm = arch.gmap
    if not hasattr(gm, "codon_sites"):
        return [np.array([d]) for d in diff]
    cod = np.full(arch.layout.L, -1)
    cod[gm.codon_sites.ravel()] = np.repeat(np.arange(len(gm.codon_sites)), 3)
    groups = {}
    for d in diff:
        key = ("c", int(cod[d])) if cod[d] >= 0 else ("s", int(d))
        groups.setdefault(key, []).append(int(d))
    return [np.array(v) for v in groups.values()]


def count_dmis(arch, ga, gb, environment=None, tol=0.005, functional_only=True, min_freq=0.9):
    """Cuenta incompatibilidades de Dobzhansky-Muller entre dos poblaciones.

    Definición operacional (como en Orr 1995 y estudios de introgresión): sobre el fondo
    homocigoto de la población X, un par de diferencias fijadas (i, j) es una DMI si
      1. introducir el alelo de Y en j es deletéreo: Δw(j) < −tol, y
      2. introducir también el alelo de Y en i lo rescata: w(X+i+j) − w(X+j) > tol, y
      3. el rescate es por interacción: epistasis w(X+i+j) − w(X+i) − w(X+j) + w(X) > tol.
    Es decir, el alelo de Y en j solo es malo junto al alelo de X en i. La condición 2 descarta
    la epistasis positiva sin rescate (i menos dañino sobre un fondo ya dañado por j, por la
    concavidad del flujo o porque j anula un gen); la 3 descarta los i beneficiosos por sí
    solos (p. ej. X fijó por deriva un alelo deletéreo en i), que compensan sin interactuar.
    Las diferencias de un mismo codón se introgresan juntas (ver `_introgression_units`), y
    i, j son esas unidades; los detalles reportan el primer sitio de cada una.
    Se cuenta en ambos fondos (X=A, X=B). Devuelve (n_DMIs, K diferencias, detalles).
    Cada detalle es (fondo, i, j, efecto de j solo, rescate); `dmi_mechanisms` los clasifica.

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
    units = _introgression_units(arch, diff)
    U = len(units)
    for X, Y, label in ((ca, cb, "A"), (cb, ca, "B")):
        base = np.broadcast_to(X, (1, 2, len(X)))
        w0 = log_fitness(arch, base.copy(), environment)[0]
        singles = np.broadcast_to(X, (U, 2, len(X))).copy()
        for u, s in enumerate(units):
            singles[u][:, s] = Y[s]
        w1 = log_fitness(arch, singles, environment) - w0                  # efecto de cada introgresión
        for j in np.flatnonzero(w1 < -tol):
            g = np.broadcast_to(singles[j], (U, 2, len(X))).copy()
            for u, s in enumerate(units):
                g[u][:, s] = Y[s]
            w2 = log_fitness(arch, g, environment) - w0                    # doble introgresión (i, j)
            rescue = w2 - w1[j]
            eps = rescue - w1
            for i in np.flatnonzero((rescue > tol) & (eps > tol) & (np.arange(U) != j)):
                pairs.append((label, int(units[i][0]), int(units[j][0]), float(w1[j]), float(rescue[i])))
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
