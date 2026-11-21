"""Meiosis: produce gametos haploides a partir de padres diploides.

Modelo:
- Segregación independiente: en cada cromosoma el gameto empieza copiando una
  de las dos cromátidas del padre al azar (1ª ley de Mendel / 2ª ley entre cromosomas).
- Crossing-over: el número de entrecruzamientos por cromosoma es Poisson con
  media igual a su longitud en Morgans, en posiciones uniformes (sin
  interferencia). Esto equivale a la función de mapeo de Haldane.
"""
import numpy as np


def make_gametes(genomes, parents, layout, rng):
    """genomes: (N, 2, L) uint8; parents: (n,) índices -> gametos (n, L) uint8."""
    parents = np.asarray(parents)
    n, L = len(parents), layout.L

    # toggles[g, i] = 1 si en el sitio i el gameto g cambia de cromátida
    toggles = np.zeros((n, L), dtype=np.uint8)

    # Cromátida inicial aleatoria en cada cromosoma (segregación independiente)
    toggles[:, layout.chrom_start] = rng.integers(0, 2, size=(n, layout.n_chrom), dtype=np.uint8)

    # Crossing-overs
    counts = rng.poisson(layout.morgans_per_chrom, size=(n, layout.n_chrom))
    total = int(counts.sum())
    if total:
        gam = np.repeat(np.repeat(np.arange(n), layout.n_chrom), counts.ravel())
        chrom = np.repeat(np.tile(np.arange(layout.n_chrom), n), counts.ravel())
        pos = layout.morgan_offset[chrom] + rng.random(total) * layout.morgans_per_chrom[chrom]
        idx = np.searchsorted(layout.site_morgan, pos, side="left")
        ok = (idx > layout.chrom_start[chrom]) & (idx < layout.chrom_end[chrom])
        # (un crossover antes del primer sitio solo re-aleatoriza la cromátida inicial,
        #  que ya es aleatoria; uno después del último sitio no tiene efecto)
        np.bitwise_xor.at(toggles, (gam[ok], idx[ok]), 1)

    strand = np.bitwise_xor.accumulate(toggles, axis=1)
    return np.where(strand.astype(bool), genomes[parents, 1], genomes[parents, 0])
