"""Estadísticos de genética de poblaciones y evolución molecular."""
import itertools

import numpy as np

from .genetic_code import CODON_TO_AA, STOP


def haplotypes(genomes):
    return genomes.reshape(-1, genomes.shape[-1])          # (2N, L)


def base_counts(genomes):
    h = haplotypes(genomes)
    return np.stack([(h == b).sum(axis=0) for b in range(4)], axis=1)   # (L, 4)


def derived_freq(genomes, ancestral):
    return (genomes != ancestral).mean(axis=(0, 1))


def expected_heterozygosity(genomes):
    """H = 1 - sum p_i^2 por sitio (diversidad génica de Nei, sin corrección muestral)."""
    p = base_counts(genomes) / (2 * genomes.shape[0])
    return 1.0 - (p ** 2).sum(axis=1)


def nucleotide_diversity(genomes):
    """pi: diferencias por sitio promedio entre dos haplotipos tomados sin reemplazo."""
    n = 2 * genomes.shape[0]
    return expected_heterozygosity(genomes).mean() * n / (n - 1)


def observed_heterozygosity(genomes):
    return (genomes[:, 0] != genomes[:, 1]).mean(axis=0)


def genotype_counts(genomes, site, ancestral_base):
    d = (genomes[:, :, site] != ancestral_base).sum(axis=1)
    return np.bincount(d, minlength=3)   # [AA, Aa, aa] respecto al ancestral


def linkage_disequilibrium(genomes, ancestral, i, j):
    h = haplotypes(genomes)
    x, y = h[:, i] != ancestral[i], h[:, j] != ancestral[j]
    return (x & y).mean() - x.mean() * y.mean()


def consensus(genomes):
    return base_counts(genomes).argmax(axis=1).astype(np.uint8)


def fixed_derived(genomes, ancestral):
    """Sitios donde toda la población comparte una base distinta a la ancestral (sustituciones)."""
    h = haplotypes(genomes)
    return (h == h[0]).all(axis=0) & (h[0] != ancestral)


# ---------------------------------------------------------------- estructura poblacional
def _group_freqs(genomes, groups, sites, ancestral):
    """Frecuencia del alelo derivado y n° de haplotipos por grupo: (G, S), (G,)."""
    labels, inv = np.unique(groups, return_inverse=True)
    d = (genomes[:, :, sites] != ancestral[sites]).sum(axis=1)     # (N, S) copias derivadas
    G = len(labels)
    counts = np.zeros((G, len(sites)))
    np.add.at(counts, inv, d)
    n = 2 * np.bincount(inv, minlength=G).astype(float)
    return counts / n[:, None], n, labels


def fst(genomes, groups, ancestral, sites=None, min_size=2):
    """F_ST = (H_T - H_S) / H_T (Nei), promediado como cociente de promedios sobre sitios.
    H_S corrige el sesgo de muestreo con n/(n-1). Grupos con < min_size individuos se ignoran."""
    if sites is None:
        sites = np.arange(genomes.shape[-1])
    p, n, _ = _group_freqs(genomes, groups, sites, ancestral)
    ok = n >= 2 * min_size
    p, n = p[ok], n[ok]
    hs = (2 * p * (1 - p) * (n / (n - 1))[:, None]).mean(axis=0)
    pbar = p.mean(axis=0)
    ht = 2 * pbar * (1 - pbar)
    return float((ht - hs).sum() / ht.sum()) if ht.sum() > 0 else np.nan


def pairwise_fst(genomes, groups, ancestral, sites=None):
    """F_ST de Hudson (1992) entre cada par de grupos: (H_entre − H_dentro) / H_entre.

    Es el F_ST de la teoría de identidad por descendencia (Q_dentro − Q_entre)/(1 − Q_entre)
    usado por Rousset (1997). Ojo: el G_ST de Nei con dos poblaciones da ~la mitad,
    porque su H_T mezcla pares de dentro y de entre. Devuelve (matriz, etiquetas)."""
    if sites is None:
        sites = np.arange(genomes.shape[-1])
    p, n, labels = _group_freqs(genomes, groups, sites, ancestral)
    G = len(labels)
    hw_g = 2 * p * (1 - p) * (n / (n - 1))[:, None]
    M = np.zeros((G, G))
    for i in range(G):
        for j in range(i + 1, G):
            hb = (p[i] * (1 - p[j]) + p[j] * (1 - p[i])).sum()
            hw = ((hw_g[i] + hw_g[j]) / 2).sum()
            M[i, j] = M[j, i] = (hb - hw) / hb if hb > 0 else np.nan
    return M, labels


# ---------------------------------------------------------------- evolución molecular
def _aa(codon):
    return CODON_TO_AA[16 * codon[0] + 4 * codon[1] + codon[2]]


def synonymous_sites(codons):
    """Sitios sinónimos y no sinónimos (Nei & Gojobori 1986). codons: (C, 3)."""
    S = 0.0
    for cod in codons:
        aa0 = _aa(cod)
        for pos in range(3):
            for b in range(4):
                if b == cod[pos]:
                    continue
                mut = cod.copy()
                mut[pos] = b
                if _aa(mut) == aa0:
                    S += 1 / 3
    return S, 3 * len(codons) - S


def count_differences(codons_a, codons_b):
    """Diferencias sinónimas (Sd) y no sinónimas (Nd) promediando rutas evolutivas."""
    Sd = Nd = 0.0
    for ca, cb in zip(codons_a, codons_b):
        diff = np.flatnonzero(ca != cb)
        if len(diff) == 0:
            continue
        s_tot = n_tot = 0.0
        n_paths = 0
        for order in itertools.permutations(diff):
            cur = ca.copy()
            s = n = 0
            valid = True
            for pos in order:
                prev_aa = _aa(cur)
                cur[pos] = cb[pos]
                new_aa = _aa(cur)
                if new_aa == STOP and pos != order[-1]:
                    valid = False
                    break
                if new_aa == prev_aa:
                    s += 1
                else:
                    n += 1
            if valid:
                s_tot += s
                n_tot += n
                n_paths += 1
        if n_paths:
            Sd += s_tot / n_paths
            Nd += n_tot / n_paths
    return Sd, Nd


def jukes_cantor(p):
    p = np.asarray(p, dtype=float)
    return np.where(p < 0.75, -0.75 * np.log(np.maximum(1 - 4 * p / 3, 1e-12)), np.inf)


def dn_ds(seq_a, seq_b, coding_codon_sites):
    """dN, dS (corregidos por Jukes-Cantor) entre dos secuencias en los codones dados."""
    ca, cb = seq_a[coding_codon_sites], seq_b[coding_codon_sites]
    S, N = synonymous_sites(ca)
    Sd, Nd = count_differences(ca, cb)
    dS, dN = float(jukes_cantor(Sd / S)), float(jukes_cantor(Nd / N))
    return {"dN": dN, "dS": dS, "omega": dN / dS if dS > 0 else np.nan,
            "Sd": Sd, "Nd": Nd, "S_sites": S, "N_sites": N}
