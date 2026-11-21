"""Arquitectura genética de una especie: layout + secuencia ancestral + mapa G->P + mutación."""
from dataclasses import dataclass, field

import numpy as np

from .genetic_code import CODON_TO_AA, SENSE_CODONS, codon_index
from .layout import GenomeLayout
from .maps import CodonMap, Gene, NeutralMap, RecognitionCode, Regulation

INTERGENIC, PROMOTER, CODING = 0, 1, 2


@dataclass
class GenomeArchitecture:
    layout: GenomeLayout
    ancestral: np.ndarray        # (L,) uint8
    gmap: object                 # NeutralMap | SiteEffectsMap | CodonMap
    mutation_rate: float = 0.0   # por sitio por generación
    kappa: float = 1.0           # razón transición/transversión
    site_class: np.ndarray = field(default=None)  # INTERGENIC / PROMOTER / CODING


def neutral_architecture(rng, n_chrom, sites_per_chrom, morgans_per_chrom, mutation_rate=0.0, kappa=1.0):
    layout = GenomeLayout([sites_per_chrom] * n_chrom, morgans_per_chrom)
    ancestral = rng.integers(0, 4, layout.L, dtype=np.uint8)
    return GenomeArchitecture(layout, ancestral, NeutralMap(), mutation_rate, kappa,
                              np.full(layout.L, INTERGENIC))


def codon_architecture(
    rng,
    n_chrom=4,
    morgans_per_chrom=0.5,
    n_enzyme_genes=12,
    n_trait_genes=6,
    n_traits=2,
    codons_per_gene=100,
    promoter_len=30,
    intergenic_len=150,
    mutation_rate=1e-5,
    kappa=2.0,
    constraint_shape=0.5,       # distribución gamma de la restricción por posición:
    constraint_mean=0.15,       # pocas posiciones críticas, muchas tolerantes
    K_range=(0.05, 0.5),
    importance_mean=0.3,
    promoter_effect_sd=0.05,
    pleiotropy_prob=0.5,
    env_noise_sd=0.1,
    n_tf=0,                     # factores de transcripción (regulación en trans)
    tf_targets=2,               # genes regulados por cada TF
    recognition_len=8,          # codones de la hélice de reconocimiento = largo del sitio de unión
    tf_mu=4.0,                  # log-concentración del TF: θ ancestral = e^mu/(1+e^mu)
    contact_energy=(1.0, 3.0),  # rango de energías (kT) de un contacto aminoácido-base no óptimo
    tf_target_roles=("enzyme", "trait"),  # qué tipo de genes pueden ser regulados
    expression_omega=None,      # ancho de la selección estabilizadora sobre la expresión regulada (None = apagada)
):
    """Construye un genoma con genes reales (codones) repartidos en cromosomas.

    Estructura de cada gen: [intergénico][sitio de unión del TF, si es blanco][promotor (genes de rasgo)][codificante]
    """
    genes_spec = ["enzyme"] * n_enzyme_genes + ["trait"] * n_trait_genes + ["tf"] * n_tf
    order = rng.permutation(len(genes_spec))
    target_of = {}      # gen blanco -> TF que lo regula
    if n_tf:
        candidates = [i for i, r in enumerate(genes_spec) if r in tf_target_roles]
        if n_tf * tf_targets > len(candidates):
            raise ValueError("no hay suficientes genes para tantos blancos de TF")
        chosen = rng.permutation(candidates)[: n_tf * tf_targets]
        tf_ids = [i for i, r in enumerate(genes_spec) if r == "tf"]
        for k, gi in enumerate(chosen):
            target_of[int(gi)] = tf_ids[k // tf_targets]
    per_chrom = [[] for _ in range(n_chrom)]
    for k, gi in enumerate(order):
        per_chrom[k % n_chrom].append(gi)

    seq_parts, class_parts, sites_per_chrom = [], [], []
    gene_sites = {}
    binding_sites = {}
    pos = 0

    def add(part, cls):
        nonlocal pos
        seq_parts.append(part)
        class_parts.append(np.full(len(part), cls))
        idx = np.arange(pos, pos + len(part))
        pos += len(part)
        return idx

    for c in range(n_chrom):
        start = pos
        for gi in per_chrom[c]:
            add(rng.integers(0, 4, intergenic_len, dtype=np.uint8), INTERGENIC)
            if gi in target_of:
                binding_sites[gi] = add(np.zeros(recognition_len, dtype=np.uint8), PROMOTER)   # se fija abajo
            prom = None
            if genes_spec[gi] == "trait":
                prom = add(rng.integers(0, 4, promoter_len, dtype=np.uint8), PROMOTER)
            codons = rng.choice(SENSE_CODONS, codons_per_gene)
            bases = np.stack([codons // 16, (codons // 4) % 4, codons % 4], axis=1).ravel().astype(np.uint8)
            coding = add(bases, CODING)
            gene_sites[gi] = (prom, coding)
        add(rng.integers(0, 4, intergenic_len, dtype=np.uint8), INTERGENIC)
        sites_per_chrom.append(pos - start)

    ancestral = np.concatenate(seq_parts)
    site_class = np.concatenate(class_parts)
    layout = GenomeLayout(sites_per_chrom, morgans_per_chrom)

    genes = []
    for gi, role in enumerate(genes_spec):
        prom, coding = gene_sites[gi]
        constraint = rng.gamma(constraint_shape, constraint_mean / constraint_shape, codons_per_gene)
        if role == "enzyme":
            genes.append(Gene(
                name=f"enz{gi}", role=role, coding_sites=coding, constraint=constraint,
                K=float(np.exp(rng.uniform(np.log(K_range[0]), np.log(K_range[1])))),
                importance=float(min(1.0, rng.exponential(importance_mean))),
            ))
        elif role == "tf":
            # Hélice de reconocimiento: sus aminoácidos determinan qué bases prefiere el TF.
            # No afectan la estabilidad de la proteína (restricción 0), solo la especificidad.
            start = codons_per_gene // 4
            recog = np.arange(start, start + recognition_len)
            constraint[recog] = 0.0
            genes.append(Gene(name=f"tf{gi}", role=role, coding_sites=coding, constraint=constraint,
                              recognition_codons=recog))
        else:
            eff = rng.normal(0.0, promoter_effect_sd, (promoter_len, 4))
            eff[np.arange(promoter_len), ancestral[prom]] = 0.0
            mask = rng.random(n_traits) < pleiotropy_prob
            mask[rng.integers(n_traits)] = True
            genes.append(Gene(
                name=f"trait{gi}", role=role, coding_sites=coding, constraint=constraint,
                promoter_sites=prom, promoter_effects=eff,
                pleiotropy=rng.normal(0.0, 1.0, n_traits) * mask,
            ))

    regulations, recognition = [], None
    if n_tf:
        # Código de reconocimiento: cada aminoácido tiene una base preferida (energía 0)
        # y energías de contacto positivas para las demás. Es física, igual en todos los linajes.
        energy = rng.uniform(*contact_energy, (21, 4))
        energy[np.arange(21), rng.integers(0, 4, 21)] = 0.0
        recognition = RecognitionCode(energy, tf_mu)
        for gi, tf in target_of.items():
            # Estado ancestral evolucionado: el sitio coincide con lo que reconoce el TF ancestral
            rc = genes[tf].coding_sites.reshape(-1, 3)[genes[tf].recognition_codons]
            aa = CODON_TO_AA[codon_index(ancestral[rc[:, 0]], ancestral[rc[:, 1]], ancestral[rc[:, 2]])]
            ancestral[binding_sites[gi]] = recognition.preferred_base[aa]
            regulations.append(Regulation(tf=tf, target=gi, site=binding_sites[gi]))

    gmap = CodonMap(genes, ancestral, n_traits=n_traits, env_noise_sd=env_noise_sd,
                    regulations=regulations, recognition=recognition,
                    expression_omega=expression_omega)
    return GenomeArchitecture(layout, ancestral, gmap, mutation_rate, kappa, site_class)
