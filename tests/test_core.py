import numpy as np

from evosim import GenomeLayout, Population, SiteEffectsMap, codon_architecture, neutral_architecture
from evosim.genetic_code import STOP, CODON_TO_AA, translate
from evosim.meiosis import make_gametes
from evosim.mutation import mutate
from evosim import stats


def test_genetic_code():
    assert (CODON_TO_AA == STOP).sum() == 3
    # ATG GCT TGG TAA  ->  M A W *
    seq = np.array([0, 3, 2, 2, 1, 3, 3, 2, 2, 3, 0, 0], dtype=np.uint8)
    assert translate(seq) == "MAW*"


def test_gametes_come_from_parent():
    rng = np.random.default_rng(1)
    layout = GenomeLayout([500, 500], 2.0)
    g = np.zeros((1, 2, layout.L), np.uint8)
    g[0, 1] = 1
    gam = make_gametes(g, np.zeros(2000, int), layout, rng)
    assert set(np.unique(gam)) <= {0, 1}
    # cada sitio viene 50/50 de cada cromátida
    assert abs(gam.mean() - 0.5) < 0.02


def test_recombination_fraction_matches_haldane():
    rng = np.random.default_rng(2)
    layout = GenomeLayout([101], 1.0)
    g = np.zeros((1, 2, layout.L), np.uint8)
    g[0, 1] = 1
    gam = make_gametes(g, np.zeros(40000, int), layout, rng)
    for j in [5, 20, 50, 100]:
        r_obs = (gam[:, 0] != gam[:, j]).mean()
        assert abs(r_obs - layout.recombination_fraction(0, j)) < 0.01


def test_unlinked_chromosomes_assort_independently():
    rng = np.random.default_rng(3)
    layout = GenomeLayout([10, 10], 0.0)  # sin crossovers
    g = np.zeros((1, 2, layout.L), np.uint8)
    g[0, 1] = 1
    gam = make_gametes(g, np.zeros(20000, int), layout, rng)
    assert (gam[:, :10] == gam[:, [0]]).all()          # ligamiento completo dentro del cromosoma
    assert abs((gam[:, 0] != gam[:, 10]).mean() - 0.5) < 0.015


def test_mutation_rate_and_kappa():
    rng = np.random.default_rng(4)
    x = np.zeros((1000, 1000), np.uint8)
    k = mutate(x, 1e-2, rng, kappa=4.0)
    assert abs(k / x.size - 1e-2) < 1e-3
    changed = x[x != 0]
    assert abs((changed == 2).mean() - 4 / 6) < 0.03   # transiciones A->G


def test_site_effects_genotype_fitness():
    m = SiteEffectsMap.single(0, 0, s=0.2, h=0.25)
    g = np.array([[[0], [0]], [[0], [1]], [[2], [2]]], np.uint8)
    w = np.exp(m.evaluate(g)[0])
    assert np.allclose(w, [1.0, 1.05, 1.2])


def test_population_runs_and_mutates():
    rng = np.random.default_rng(5)
    arch = neutral_architecture(rng, 2, 100, 0.5, mutation_rate=1e-3)
    pop = Population(arch, 50, rng)
    pop.run(20)
    assert stats.nucleotide_diversity(pop.genomes) > 0


def test_codon_architecture_wildtype_fitness_is_one():
    rng = np.random.default_rng(6)
    arch = codon_architecture(rng)
    g = np.broadcast_to(arch.ancestral, (3, 2, arch.layout.L)).copy()
    logw, traits = arch.gmap.evaluate(g)
    assert np.allclose(logw, 0.0)
    assert np.allclose(traits, 0.0)


def test_codon_null_allele_is_recessive():
    """Kacser-Burns: un alelo nulo en heterocigosis casi no reduce el flujo."""
    rng = np.random.default_rng(7)
    arch = codon_architecture(rng, n_trait_genes=0, importance_mean=10.0)  # importancia ~1
    gene = arch.gmap.genes[0]
    g = np.broadcast_to(arch.ancestral, (3, 2, arch.layout.L)).copy()
    stop = np.array([3, 0, 0], np.uint8)   # TAA
    first = gene.coding_sites[:3]
    g[1, 0, first] = stop
    g[2, 0, first] = stop
    g[2, 1, first] = stop
    w = np.exp(arch.gmap.evaluate(g)[0])
    s = 1 - w[2]
    h = (1 - w[1]) / s
    assert s > 0.5 and h < 0.3
