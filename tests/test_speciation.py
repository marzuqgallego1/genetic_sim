import numpy as np

from evosim import codon_architecture, neutral_architecture
from evosim.genetic_code import CODON_TO_AA, SENSE_CODONS
from evosim.phylo import (jc_distance_matrix, neighbor_joining, parse_newick, robinson_foulds)
from evosim.speciation import (TF_SITE, INTRAGENIC, REGULATORY, Radiation, count_dmis, cross, dmi_mechanism,
                               dmi_mechanisms, hybrid_fitness, isolation_index, site_annotation)


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
    # Solo el blanco enzimático cuenta para el fitness intrínseco; el otro blanco es un gen de
    # rasgo y su desajuste no tiene costo sin ambiente: no hay rescate y no es una DMI.
    assert n == 1
    found = {tuple(sorted((p[1], p[2]))) for p in pairs}
    assert any(site in f for f in found)
    mech = dmi_mechanisms(arch, pairs)
    assert sum(mech.values()) == n
    assert mech[TF_SITE] == 1                               # hélice del TF × sitio del blanco enzimático
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


def test_expression_selection_off_by_default_and_penalizes_misregulation():
    """Selección estabilizadora sobre la expresión: apagada por defecto, 0 en el silvestre,
    y castiga la mala regulación más que el flujo metabólico solo."""
    plain = _tf_arch()
    strict = codon_architecture(np.random.default_rng(4), n_enzyme_genes=6, n_trait_genes=2, n_traits=1,
                                n_tf=2, codons_per_gene=40, promoter_len=10, intergenic_len=20,
                                expression_omega=0.2)
    assert plain.gmap.expression_omega is None
    assert np.array_equal(plain.ancestral, strict.ancestral)
    wt = np.broadcast_to(strict.ancestral, (2, 2, strict.ancestral.size)).copy()
    assert np.allclose(strict.gmap.evaluate(wt)[0], 0.0)
    assert np.allclose(strict.gmap.expression_levels(wt), 1.0)

    reg = strict.gmap.regulations[0]
    bad = wt[:1].copy()
    j = reg.site[:3]                                               # 3 de 8 posiciones desajustadas
    bad[0][:, j] = (strict.ancestral[j] + 1) % 4
    k = strict.gmap.regulations.index(reg)
    expr = strict.gmap.expression_levels(bad)[0, k]
    assert 1e-3 < expr < 0.99
    w_plain = plain.gmap.evaluate(bad)[0][0]
    w_strict = strict.gmap.evaluate(bad)[0][0]
    assert np.isclose(w_strict, w_plain - np.log(expr) ** 2 / (2 * 0.2 ** 2))
    assert w_strict < w_plain


def test_dmi_mechanism_labels():
    arch = _tf_arch()
    gm = arch.gmap
    ann = site_annotation(arch)
    reg = gm.regulations[0]
    rc = gm._recog_sites[reg.tf].ravel()
    tf_coding = np.setdiff1d(gm.genes[reg.tf].coding_sites, rc)
    tgt = gm.genes[reg.target].coding_sites
    enz = [i for i, g in enumerate(gm.genes) if g.role == "enzyme" and i not in {r.target for r in gm.regulations}][0]
    ec = gm.genes[enz].coding_sites
    assert dmi_mechanism(arch, rc[0], reg.site[2], ann) == TF_SITE
    assert dmi_mechanism(arch, reg.site[2], rc[0], ann) == TF_SITE
    assert dmi_mechanism(arch, tf_coding[0], tgt[0], ann) == REGULATORY
    assert dmi_mechanism(arch, reg.site[0], tgt[5], ann) == REGULATORY
    assert dmi_mechanism(arch, ec[0], ec[9], ann) == INTRAGENIC
    assert dmi_mechanism(arch, tf_coding[0], tf_coding[4], ann) == INTRAGENIC


def test_count_dmis_requires_real_rescue():
    """Dos codones de paro en la misma enzima tienen epistasis positiva (el segundo no daña más
    a un gen ya nulo), pero ninguno rescata al otro: no es una DMI."""
    arch = _tf_arch()
    gm = arch.gmap
    g = [i for i, gn in enumerate(gm.genes) if gn.role == "enzyme" and gn.importance > 0.05][0]
    ga = np.broadcast_to(arch.ancestral, (10, 2, arch.layout.L)).copy()
    gb = ga.copy()
    cs = np.asarray(gm.genes[g].coding_sites).reshape(-1, 3)
    for k in (5, 20):
        gb[:, :, cs[k]] = [3, 0, 0]                         # TAA
    n, K, pairs = count_dmis(arch, ga, gb)
    assert K >= 2 and n == 0
