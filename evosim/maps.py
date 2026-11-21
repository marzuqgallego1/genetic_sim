"""Mapas genotipo -> (fitness intrínseco, fenotipo).

Todos los mapas implementan:
    evaluate(genomes, rng) -> (log_fitness (N,), rasgos (N, T) o None)

- NeutralMap: todas las mutaciones son neutras (para validar deriva/mutación/recombinación).
- SiteEffectsMap: modelo clásico de genética de poblaciones; cada sitio tiene un
  coeficiente de selección s y una dominancia h asignados a mano. Sirve para
  comparar con fórmulas teóricas exactas (Kimura, recursiones deterministas).
- CodonMap: modelo molecular. Los genes se traducen con el código genético; la
  actividad de cada proteína depende de qué tan radicales son sus cambios de
  aminoácido y de si hay codones de paro prematuros. La dominancia NO se asigna:
  emerge de la cinética enzimática (Kacser & Burns 1981) y de la dosis génica.
  Opcionalmente incluye regulación en trans: factores de transcripción (TF) cuyos
  aminoácidos de reconocimiento se unen a sitios de ADN en los promotores de sus
  genes blanco (modelo termodinámico; Tulchinsky et al. 2014, Khatri & Goldstein 2015).
  La coevolución TF–sitio en linajes aislados produce incompatibilidades de
  Dobzhansky-Muller que NO se programan: emergen al juntar en un híbrido un TF de
  un linaje con sitios de unión del otro.
"""
from dataclasses import dataclass, field

import numpy as np

from .genetic_code import AA_DISTANCE, CODON_TO_AA, STOP, codon_index


class NeutralMap:
    def evaluate(self, genomes, rng=None):
        return np.zeros(genomes.shape[0]), None


class SiteEffectsMap:
    """Fitness multiplicativo entre sitios.
    Genotipos en un sitio: ancestral/ancestral = 1, het = 1 + h*s, derivado/derivado = 1 + s.
    """

    def __init__(self, sites, sel, h):
        self.sites = np.asarray(sites, dtype=np.int64)
        self.sel = np.asarray(sel, dtype=float).reshape(len(self.sites), 4)  # s de cada base
        self.h = np.asarray(h, dtype=float).reshape(len(self.sites))

    @classmethod
    def single(cls, site, ancestral_base, s, h):
        sel = np.full((1, 4), float(s))
        sel[0, ancestral_base] = 0.0
        return cls([site], sel, [h])

    def evaluate(self, genomes, rng=None):
        N = genomes.shape[0]
        if len(self.sites) == 0:
            return np.zeros(N), None
        b = genomes[:, :, self.sites]                         # (N, 2, k)
        a = self.sel[np.arange(len(self.sites))[None, None, :], b]
        a1, a2 = a[:, 0], a[:, 1]
        c = np.where(a1 == a2, a1, self.h * (a1 + a2))
        return np.log(np.maximum(1.0 + c, 1e-12)).sum(axis=1), None


@dataclass
class Gene:
    """Un gen codificante, opcionalmente con región promotora (cis-regulatoria).

    role = "enzyme": su producto contribuye al fitness a través del flujo metabólico.
    role = "trait":  su nivel de producto contribuye a rasgos cuantitativos (pleiotropía).
    """
    name: str
    role: str
    coding_sites: np.ndarray          # índices de sitios, longitud múltiplo de 3
    constraint: np.ndarray            # (n_codones,) sensibilidad de cada posición a cambios
    promoter_sites: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    promoter_effects: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # efecto en log-expresión
    K: float = 0.1                    # constante de saturación del flujo (enzimas)
    importance: float = 0.0           # pérdida de fitness del homocigoto nulo (enzimas)
    pleiotropy: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (T,) efecto en cada rasgo
    recognition_codons: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # TF: hélice de reconocimiento


@dataclass
class Regulation:
    """Un TF (gen `tf`) se une al sitio `site` (bases del genoma) del promotor del gen `target`.

    La posición j del sitio es contactada por el codón de reconocimiento j del TF.
    """
    tf: int
    target: int
    site: np.ndarray


@dataclass
class RecognitionCode:
    """Física de la unión proteína-ADN, común a toda la especie (y sus descendientes).

    energy[aa, base]: energía (en kT) de que el aminoácido aa contacte la base; 0 = contacto
    óptimo. Ocupación de un sitio con energía E por un TF a concentración x·e^mu:
        θ = Z / (1 + Z),   Z = Σ_alelos x_k e^(mu − E_k)        (Boltzmann, unión competitiva)
    """
    energy: np.ndarray          # (21, 4)
    mu: float = 4.0             # potencial químico (log-concentración) del TF silvestre

    @property
    def preferred_base(self):
        return self.energy.argmin(axis=1)


class CodonMap:
    def __init__(self, genes, ancestral, n_traits=0, env_noise_sd=0.0, min_output=1e-3,
                 regulations=(), recognition=None):
        self.genes = list(genes)
        self.regulations = list(regulations)
        self.recognition = recognition
        self.n_traits = n_traits
        self.env_noise_sd = env_noise_sd
        self.min_output = min_output
        G = len(self.genes)

        codon_sites, codon_gene, codon_c = [], [], []
        prom_sites, prom_gene, prom_eff = [], [], []
        for g, gene in enumerate(self.genes):
            cs = np.asarray(gene.coding_sites).reshape(-1, 3)
            codon_sites.append(cs)
            codon_gene.append(np.full(len(cs), g))
            codon_c.append(np.asarray(gene.constraint, dtype=float))
            if len(gene.promoter_sites):
                prom_sites.append(np.asarray(gene.promoter_sites))
                prom_gene.append(np.full(len(gene.promoter_sites), g))
                prom_eff.append(np.asarray(gene.promoter_effects, dtype=float))

        self.codon_sites = np.concatenate(codon_sites)
        self.codon_c = np.concatenate(codon_c)
        cg = np.concatenate(codon_gene)
        self.gene_codon_start = np.flatnonzero(np.r_[True, cg[1:] != cg[:-1]])

        anc_codons = ancestral[self.codon_sites]
        self.anc_aa = CODON_TO_AA[codon_index(anc_codons[:, 0], anc_codons[:, 1], anc_codons[:, 2])]
        if (self.anc_aa == STOP).any():
            raise ValueError("la secuencia ancestral tiene codones de paro dentro de un gen")

        # Tabla precalculada: log-actividad de cada posición de codón para cada uno de los 64 codones
        d = AA_DISTANCE[self.anc_aa[:, None], CODON_TO_AA[None, :]]          # (C, 64)
        with np.errstate(invalid="ignore"):                                   # 0 × inf en columnas de paro
            loss = np.minimum(1.0, self.codon_c[:, None] * d)
        loss[:, CODON_TO_AA == STOP] = 1.0                                    # paro prematuro = alelo nulo
        self.codon_log_activity = np.log(np.maximum(1.0 - loss, 1e-9))
        self._codon_flat_offset = (np.arange(len(cg)) * 64).astype(np.int64)

        if prom_sites:
            self.promoter_sites = np.concatenate(prom_sites)
            self.promoter_eff = np.concatenate(prom_eff)
            pg = np.concatenate(prom_gene)
            self.promoter_gene_ind = np.zeros((len(pg), G))
            self.promoter_gene_ind[np.arange(len(pg)), pg] = 1.0
        else:
            self.promoter_sites = np.zeros(0, dtype=np.int64)

        self.enzyme_idx = np.array([i for i, g in enumerate(self.genes) if g.role == "enzyme"], dtype=int)
        self.trait_idx = np.array([i for i, g in enumerate(self.genes) if g.role == "trait"], dtype=int)
        self.K = np.array([self.genes[i].K for i in self.enzyme_idx])
        self.importance = np.array([self.genes[i].importance for i in self.enzyme_idx])
        if len(self.trait_idx):
            self.pleiotropy = np.stack([self.genes[i].pleiotropy for i in self.trait_idx])  # (Gt, T)

        # Regulación en trans: sitios de los codones de reconocimiento de cada TF
        self.tf_idx = np.array([i for i, g in enumerate(self.genes) if g.role == "tf"], dtype=int)
        self._recog_sites = {}
        for i in self.tf_idx:
            cs = np.asarray(self.genes[i].coding_sites).reshape(-1, 3)
            self._recog_sites[i] = cs[np.asarray(self.genes[i].recognition_codons)]   # (Lr, 3)
        targets = [r.target for r in self.regulations]
        if len(set(targets)) != len(targets):
            raise ValueError("cada gen puede tener a lo sumo un sitio de unión")
        if any(self.genes[r.target].role == "tf" for r in self.regulations):
            raise ValueError("los TF no pueden ser blancos de regulación (sin cascadas por ahora)")
        if self.regulations:
            if recognition is None:
                raise ValueError("la regulación requiere un RecognitionCode")
            anc = ancestral[None, None, :]
            self._theta0 = {}
            for r in self.regulations:
                act = np.ones((1, 2))
                self._theta0[r.target] = self._occupancy(anc, r, act)[0, 0]

    # ------------------------------------------------------------------
    def binding_energy(self, genomes, reg):
        """E[n, k, h]: energía entre el TF del haplotipo k y el sitio del haplotipo h (kT)."""
        rs = self._recog_sites[reg.tf]
        b = genomes[:, :, rs]                                                 # (N,2,Lr,3)
        aa = CODON_TO_AA[(b[..., 0] << 4) | (b[..., 1] << 2) | b[..., 2]]     # (N,2,Lr)
        site = genomes[:, :, reg.site]                                        # (N,2,Lr)
        e = self.recognition.energy[aa[:, :, None, :], site[:, None, :, :]]  # (N,k,h,Lr)
        return e.sum(axis=3)

    def _occupancy(self, genomes, reg, tf_activity):
        """θ[n, h]: ocupación del sitio del haplotipo h; ambos alelos del TF compiten por él."""
        E = self.binding_energy(genomes, reg)                                 # (N,2,2)
        x = 0.5 * tf_activity[:, :, None]                                     # dosis de cada alelo
        Z = (x * np.exp(self.recognition.mu - E)).sum(axis=1)                 # (N,2)
        return Z / (1.0 + Z)

    def occupancy(self, genomes, reg):
        """Ocupación relativa a la ancestral, por haplotipo del sitio: (N, 2)."""
        act = self._raw_outputs(genomes)[:, :, reg.tf]
        return self._occupancy(genomes, reg, act) / self._theta0[reg.target]

    # ------------------------------------------------------------------
    def gene_outputs(self, genomes):
        """Producto génico de cada haplotipo: (N, 2, G). Tipo silvestre = 1.

        Incluye actividad de la proteína (codones), expresión cis (promotor) y, para genes
        regulados, la ocupación de su sitio de unión por el TF (en trans: depende de ambos alelos
        del TF del individuo).
        """
        out = self._raw_outputs(genomes)
        if self.regulations:
            act = {i: out[:, :, i].copy() for i in self.tf_idx}
            for r in self.regulations:
                theta = self._occupancy(genomes, r, act[r.tf])
                out[:, :, r.target] *= theta / self._theta0[r.target]
        return out

    def _raw_outputs(self, genomes):
        b = genomes[:, :, self.codon_sites]                                   # (N,2,C,3)
        codon = (b[..., 0] << 4) | (b[..., 1] << 2) | b[..., 2]               # 0..63, (N,2,C)
        # Optimización: las posiciones idénticas en toda la población aportan lo mismo a todos
        first = codon[0, 0]
        poly = np.flatnonzero((codon != first).any(axis=(0, 1)))
        table = self.codon_log_activity
        mono_val = table[np.arange(len(first)), first]
        mono_val[poly] = 0.0
        log_out = np.broadcast_to(np.add.reduceat(mono_val, self.gene_codon_start),
                                  codon.shape[:2] + (len(self.genes),)).copy()
        if len(poly):
            vals = table[poly[None, None, :], codon[:, :, poly]]              # (N,2,P)
            gene_of = np.searchsorted(self.gene_codon_start, poly, side="right") - 1
            for g in np.unique(gene_of):
                log_out[:, :, g] += vals[:, :, gene_of == g].sum(axis=2)   # actividad de la proteína
        if len(self.promoter_sites):
            pb = genomes[:, :, self.promoter_sites]
            e = self.promoter_eff[np.arange(len(self.promoter_sites))[None, None, :], pb]
            log_out = log_out + e @ self.promoter_gene_ind                    # nivel de expresión
        return np.exp(log_out)

    def evaluate(self, genomes, rng=None):
        N = genomes.shape[0]
        out = self.gene_outputs(genomes).mean(axis=1)    # dosis diploide (N, G), silvestre = 1
        logw = np.zeros(N)
        if len(self.enzyme_idx):
            E = out[:, self.enzyme_idx]
            flux = E * (1.0 + self.K) / (E + self.K)      # Michaelis-Menten normalizado: flujo(1) = 1
            logw += np.log(np.maximum(1.0 - self.importance * (1.0 - flux), 1e-12)).sum(axis=1)
        traits = None
        if len(self.trait_idx):
            level = np.log(np.maximum(out[:, self.trait_idx], self.min_output))
            traits = level @ self.pleiotropy
            if self.env_noise_sd > 0 and rng is not None:
                traits = traits + rng.normal(0.0, self.env_noise_sd, traits.shape)
        return logw, traits
