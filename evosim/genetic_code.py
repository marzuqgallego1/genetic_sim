"""Código genético estándar y propiedades fisicoquímicas de los aminoácidos."""
import itertools

import numpy as np

BASES = "ACGT"  # codificación interna A=0 C=1 G=2 T=3
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY*"
STOP = AMINO_ACIDS.index("*")

# Tabla estándar en orden T, C, A, G (codones TTT, TTC, TTA, ...)
_TCAG_TABLE = (
    "FFLLSSSSYY**CC*W"
    "LLLLPPPPHHQQRRRR"
    "IIIMTTTTNNKKSSRR"
    "VVVVAAAADDEEGGGG"
)
_TO_TCAG = {0: 2, 1: 1, 2: 3, 3: 0}  # A C G T -> posición en el orden TCAG

CODON_TO_AA = np.empty(64, dtype=np.int8)
for b1, b2, b3 in itertools.product(range(4), repeat=3):
    t = 16 * _TO_TCAG[b1] + 4 * _TO_TCAG[b2] + _TO_TCAG[b3]
    CODON_TO_AA[16 * b1 + 4 * b2 + b3] = AMINO_ACIDS.index(_TCAG_TABLE[t])

SENSE_CODONS = np.flatnonzero(CODON_TO_AA != STOP)

# Clases fisicoquímicas: sustituir dentro de la misma clase es "conservativo"
_CLASSES = {
    "AVLIM": "alifático hidrofóbico",
    "FWY": "aromático",
    "STNQC": "polar sin carga",
    "KRH": "carga positiva",
    "DE": "carga negativa",
    "GP": "estructural",
}
_aa_class = np.full(21, -1)
for ci, members in enumerate(_CLASSES):
    for aa in members:
        _aa_class[AMINO_ACIDS.index(aa)] = ci

CONSERVATIVE_COST = 0.3
RADICAL_COST = 1.0

# AA_DISTANCE[a, b]: severidad relativa de cambiar el aminoácido a por b
AA_DISTANCE = np.where(_aa_class[:, None] == _aa_class[None, :], CONSERVATIVE_COST, RADICAL_COST)
np.fill_diagonal(AA_DISTANCE, 0.0)
AA_DISTANCE[:, STOP] = np.inf
AA_DISTANCE[STOP, :] = np.inf
AA_DISTANCE[STOP, STOP] = 0.0


def codon_index(b1, b2, b3):
    return 16 * b1.astype(np.int16) + 4 * b2 + b3


def translate(seq):
    """Traduce una secuencia (array de bases) a una cadena de aminoácidos."""
    seq = np.asarray(seq)
    c = seq[: len(seq) // 3 * 3].reshape(-1, 3)
    return "".join(AMINO_ACIDS[a] for a in CODON_TO_AA[codon_index(c[:, 0], c[:, 1], c[:, 2])])
