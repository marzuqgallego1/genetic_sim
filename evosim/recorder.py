"""Graba una corrida de EcoPopulation en un formato JSON compacto para el visor web."""
import json

import numpy as np

from . import stats
from .architecture import INTERGENIC


def _r(x, nd=3):
    """Redondea y convierte NaN en None (JSON null)."""
    a = np.round(np.asarray(x, dtype=float), nd)
    out = a.tolist()
    if a.ndim == 0:
        return None if np.isnan(a) else out
    return [None if (v != v) else v for v in out]


class Recorder:
    NULL_THRESHOLD = 1e-3   # producto génico por debajo de esto = alelo nulo (gen roto)

    def __init__(self, pop, scenario, title, description, trait=0, layer="temp", every=5):
        self.pop, self.every, self.trait, self.layer = pop, every, trait, layer
        land, arch, gm = pop.landscape, pop.arch, pop.arch.gmap
        self.neutral_sites = np.flatnonzero(arch.site_class == INTERGENIC)
        niche = next((a for a in pop.params.niche if a.trait == trait), None)
        self.meta = {
            "scenario": scenario,
            "title": title,
            "description": description,
            "width": land.width,
            "height": land.height,
            "K": _r(land.K, 1),
            "fecundity": pop.params.fecundity,
            "omega": niche.omega if niche else None,
            "migration_rate": pop.params.migration_rate,
            "dispersal": pop.params.dispersal,
            "mutation_rate": arch.mutation_rate,
            "genome_length": int(arch.layout.L),
            "n_chrom": int(arch.layout.n_chrom),
            "n_enzyme_genes": int(len(gm.enzyme_idx)),
            "trait_genes": [
                {"name": gm.genes[i].name, "weight": round(float(gm.pleiotropy[k, trait]), 3)}
                for k, i in enumerate(gm.trait_idx)
            ],
            "every": every,
        }
        self.frames = []

    def capture(self):
        pop = self.pop
        land, gm = pop.landscape, pop.arch.gmap
        f = {"gen": int(pop.generation), "N": int(pop.N), "env": _r(land.layers[self.layer], 3)}
        if pop.N == 0:
            f.update(density=[0] * land.n_cells, trait=[None] * land.n_cells)
            self.frames.append(f)
            return
        counts = pop.cell_counts()
        logw_int, z = gm.evaluate(pop.genomes, None)          # valor genético, sin ruido ambiental
        zt = z[:, self.trait]
        s = np.bincount(pop.cells, weights=zt, minlength=land.n_cells)
        mean_z = np.where(counts > 0, s / np.maximum(counts, 1), np.nan)
        w_total, _ = pop.fitness()

        out = gm.gene_outputs(pop.genomes)[:, :, gm.trait_idx]  # (N, 2, Gt) por haplotipo
        null = out < self.NULL_THRESHOLD
        healthy = (~null).sum(axis=(0, 1))
        level_sum = np.where(null, 0.0, np.log(np.maximum(out, 1e-12))).sum(axis=(0, 1))
        level = np.where(healthy > 0, level_sum / np.maximum(healthy, 1), np.nan)   # sin copias sanas -> null
        cols = land.x[pop.cells]
        has_cols = len(np.unique(cols)) > 1
        f.update(
            density=counts.tolist(),
            trait=_r(mean_z, 3),
            mean_w=_r(w_total.mean(), 4),
            load=_r(1 - np.exp(logw_int).mean(), 4),
            vg=_r(zt.var(), 4),
            pi=_r(stats.nucleotide_diversity(pop.genomes[:, :, self.neutral_sites]), 5),
            fst=_r(stats.fst(pop.genomes, cols, pop.arch.ancestral, self.neutral_sites), 4) if has_cols else None,
            gene_null=_r(null.mean(axis=(0, 1)), 4),
            gene_level=_r(level, 3),
        )
        self.frames.append(f)

    def run(self, generations, progress=None):
        self.capture()
        for _ in range(generations):
            self.pop.step()
            if self.pop.generation % self.every == 0 or self.pop.N == 0:
                self.capture()
                if progress:
                    progress(self.pop)
            if self.pop.N == 0:
                break

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"meta": self.meta, "frames": self.frames}, fh, separators=(",", ":"), allow_nan=False)
