"""Organización física y genética del genoma: cromosomas, sitios y mapa genético.

Cada sitio es un nucleótido. Los cromosomas se colocan uno tras otro en un eje
global de posiciones en Morgans (unidad de distancia genética: 1 Morgan = en
promedio 1 crossing-over por meiosis). Los sitios de un cromosoma se reparten
uniformemente a lo largo de su longitud genética.
"""
import numpy as np


class GenomeLayout:
    def __init__(self, sites_per_chrom, morgans_per_chrom):
        s = np.atleast_1d(np.asarray(sites_per_chrom, dtype=np.int64))
        m = np.broadcast_to(np.asarray(morgans_per_chrom, dtype=float), s.shape).copy()
        if (s <= 0).any():
            raise ValueError("cada cromosoma necesita al menos un sitio")
        if (m < 0).any():
            raise ValueError("la longitud genética no puede ser negativa")

        self.sites_per_chrom = s
        self.morgans_per_chrom = m
        self.n_chrom = len(s)
        self.L = int(s.sum())
        self.chrom_start = np.concatenate([[0], np.cumsum(s)[:-1]])
        self.chrom_end = self.chrom_start + s
        self.chrom_of_site = np.repeat(np.arange(self.n_chrom), s)
        self.morgan_offset = np.concatenate([[0.0], np.cumsum(m)[:-1]])

        local = np.arange(self.L) - self.chrom_start[self.chrom_of_site]
        c = self.chrom_of_site
        self.site_morgan = self.morgan_offset[c] + (local + 0.5) / s[c] * m[c]

    def map_distance(self, i, j):
        """Distancia genética (Morgans) entre dos sitios; inf si están en cromosomas distintos."""
        if self.chrom_of_site[i] != self.chrom_of_site[j]:
            return np.inf
        return abs(self.site_morgan[i] - self.site_morgan[j])

    def recombination_fraction(self, i, j):
        """Fracción de recombinación r según la función de mapeo de Haldane."""
        d = self.map_distance(i, j)
        if np.isinf(d):
            return 0.5
        return 0.5 * (1.0 - np.exp(-2.0 * d))
