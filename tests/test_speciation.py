import numpy as np

from evosim import codon_architecture, neutral_architecture
from evosim.genetic_code import CODON_TO_AA, SENSE_CODONS
from evosim.phylo import (jc_distance_matrix, neighbor_joining, parse_newick, robinson_foulds)
from evosim.speciation import Radiation, count_dmis, cross, hybrid_fitness, isolation_index


def _tf_arch(seed=4):
    return codon_architecture(np.random.default_rng(seed), n_enzyme_genes=6, n_trait_genes=2, n_traits=1,
                              n_tf=2, codons_per_gene=40, promoter_len=10, intergenic_len=20)


def _compensatory_pair(arch, reg, j):
    """Construye un cambio del TF en la posición j de reconocimiento que prefiere otra base,
    y el cambio del sitio de unión que lo compensa. Devuelve (sitios del codón, codón nuevo, sitio, base)."""
    gm = arch.gmap
    code = gm.recognition
    rc = gm._recog_sites[reg.tf][j]
    anc_base = arch.ancestral[reg.site[j]]
    for c in SENSE_CODONS:
        aa = CODON_TO_AA[c]
        if code.preferred_base[aa] != anc_base:
            return rc, np.array([c // 16, (c // 4) % 4, c % 4], np.uint8), reg.site[j], code.preferred_base[aa]
    raise AssertionError


def test_tf_ancestral_is_wild_type():
    arch = _tf_arch()
    g = np.broadcast_to(arch.ancestral, (3, 2, arch.layout.L)).copy()
    assert np.allclose(arch.gmap.gene_outputs(g), 1.0)
    assert np.allclose(arch.gmap.evaluate(g)[0], 0.0)


def test_binding_site_mismatch_and_compensation():
    arch = _tf_arch()
    gm = arch.gmap
    reg = gm.regulations[0]
    rc, codon, site, base = _compensatory_pair(arch, reg, 0)
    g = np.broadcast_to(arch.ancestral, (1, 2, arch.layout.L)).copy()
    g[0, :, site] = base                                    # solo el sitio cambia: desajuste
    assert gm.gene_outputs(g)[0, 0, reg.target] < 0.95
    g[0][:, rc] = codon                                       # el TF coevoluciona: se rescata
    assert gm.gene_outputs(g)[0, 0, reg.target] > 0.99
    # el otro blanco del mismo TF ahora queda desajustado (pleiotropía del TF)
    other = [r for r in gm.regulations if r.tf == reg.tf and r.target != reg.target][0]
    assert gm.gene_outputs(g)[0, 0, other.target] < 0.95


def test_tf_acts_in_trans():
    """Un heterocigoto para el TF expresa ambos sitios: el alelo que coincide los ocupa."""
    arch = _tf_arch()
    gm = arch.gmap
    reg = gm.regulations[0]
    rc, codon, site, base = _compensatory_pair(arch, reg, 0)
    g = np.broadcast_to(arch.ancestral, (1, 2, arch.layout.L)).copy()
    g[0, :, site] = base
    g[0, 1, rc] = codon                                     # TF nuevo solo en el haplotipo 1
    out = gm.gene_outputs(g)[0, :, reg.target]
    assert np.isclose(out[0], out[1])                       # mismo sitio, misma ocupación en trans
    homo_bad = g.copy(); homo_bad[0, 1, rc] = arch.ancestral[rc]
    assert out[0] > gm.gene_outputs(homo_bad)[0, 0, reg.target]


def test_count_dmis_finds_constructed_incompatibility():
    arch = _tf_arch()
    reg = arch.gmap.regulations[0]
    rc, codon, site, base = _compensatory_pair(arch, reg, 0)
    other = [r for r in arch.gmap.regulations if r.tf == reg.tf and r.target != reg.target][0]
    ga = np.broadcast_to(arch.ancestral, (20, 2, arch.layout.L)).copy()
    gb = ga.copy()
    gb[:, :, rc] = codon
    gb[:, :, site] = base
    gb[:, :, other.site[0]] = base                         # linaje B: TF y ambos sitios coevolucionados
    n, K, pairs = count_dmis(arch, ga, gb)
    assert K == int((arch.ancestral[rc] != codon).sum()) + 2
    assert n >= 2
    found = {tuple(sorted((p[1], p[2]))) for p in pairs}
    assert any(site in f for f in found)
    rng = np.random.default_rng(0)
    hf = hybrid_fitness(arch, ga, gb, rng, n=300)
    assert hf["F1"] > hf["F2"]                              # ruptura en F2: DMIs recesivas
    assert isolation_index(hf, "F2") > 0


def test_cross_mixes_parents():
    arch = neutral_architecture(np.random.default_rng(0), 2, 50, 1.0)
    a = np.zeros((10, 2, arch.layout.L), np.uint8)
    b = np.ones((10, 2, arch.layout.L), np.uint8)
    kids = cross(a, b, arch.layout, np.random.default_rng(1), 50)
    assert (kids[:, 0] == 0).all() and (kids[:, 1] == 1).all()


def test_neighbor_joining_recovers_additive_tree():
    true = parse_newick("((a:1,b:2):3,(c:1,(d:2,e:1):1):2);")
    leaves = ["a", "b", "c", "d", "e"]
    # distancias aditivas del árbol
    paths = {"a": [("a", 1), ("ab", 3)], "b": [("b", 2), ("ab", 3)], "c": [("c", 1), ("cde", 2)],
             "d": [("d", 2), ("de", 1), ("cde", 2)], "e": [("e", 1), ("de", 1), ("cde", 2)]}
    D = np.zeros((5, 5))
    for i, x in enumerate(leaves):
        for j, y in enumerate(leaves):
            if i != j:
                px, py = dict(paths[x]), dict(paths[y])
                D[i, j] = sum(v for k, v in px.items() if k not in py) + sum(v for k, v in py.items() if k not in px)
    tree = neighbor_joining(D, leaves)
    assert robinson_foulds(tree, true) == 0
    assert robinson_foulds(parse_newick("((a,c),(b,(d,e)));"), true) == 2


def test_radiation_tree_and_divergence():
    rng = np.random.default_rng(2)
    arch = neutral_architecture(rng, 2, 500, 1.0, mutation_rate=2e-3)
    rad = Radiation(arch, 30, rng)
    a, b = rad.split("L0")
    rad.run(100)
    c, d = rad.split(a)
    rad.run(150)
    true = parse_newick(rad.true_tree())
    assert sorted(true.leaves()) == sorted([b, c, d])
    from evosim.stats import consensus
    seqs = [consensus(rad.lineages[n].pop.genomes) for n in (b, c, d)]
    D = jc_distance_matrix(seqs)
    assert D[1, 2] < D[0, 1] and D[1, 2] < D[0, 2]        # c y d son hermanos
