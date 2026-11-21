import numpy as np

from evosim import EcoParams, EcoPopulation, Landscape, NicheAxis, codon_architecture, neutral_architecture, stats


def _neutral(rng, L=20):
    return neutral_architecture(rng, 2, L // 2, 0.5)


def test_population_regulates_to_K():
    rng = np.random.default_rng(1)
    land = Landscape(3, 2, K=100)
    pop = EcoPopulation.founders(_neutral(rng), land, rng, np.repeat(np.arange(6), 10),
                                 EcoParams(fecundity=2.0, dispersal="none"))
    sizes = []
    pop.run(80, callback=lambda p: sizes.append(p.N))
    assert abs(np.mean(sizes[40:]) - 600) < 30


def test_zero_K_cell_produces_no_offspring():
    rng = np.random.default_rng(2)
    land = Landscape(2, 1, K=[[50, 0]])
    pop = EcoPopulation.founders(_neutral(rng), land, rng, [0] * 20 + [1] * 20, EcoParams(dispersal="none"))
    pop.step()
    assert (pop.cells == 0).all()


def test_no_selfing_means_lone_individual_is_sterile():
    rng = np.random.default_rng(3)
    land = Landscape(1, 1, K=100)
    pop = EcoPopulation.founders(_neutral(rng), land, rng, [0], EcoParams(selfing=False, fecundity=5.0))
    pop.step()
    assert pop.N == 0


def test_fathers_come_from_same_cell():
    rng = np.random.default_rng(4)
    arch = _neutral(rng)
    land = Landscape(5, 1, K=50)
    cells = np.repeat(np.arange(5), 20)
    pop = EcoPopulation.founders(arch, land, rng, cells, EcoParams(dispersal="none"))
    mothers = rng.integers(0, 100, 5000)
    fathers, keep = pop._choose_fathers(mothers, np.ones(100))
    assert (cells[fathers] == cells[mothers]).all() and keep.all()
    # proporcional a w: en la celda 0 solo el individuo 0 tiene fitness
    w = np.ones(100)
    w[1:20] = 1e-9
    f, _ = pop._choose_fathers(np.zeros(1000, int), w)
    assert (f == 0).mean() > 0.99


def test_stepping_stone_moves_one_step_at_most():
    rng = np.random.default_rng(5)
    land = Landscape(10, 10, K=10)
    pop = EcoPopulation.founders(_neutral(rng), land, rng, [55], EcoParams(migration_rate=0.5))
    start = np.full(20000, 55)
    new = pop._disperse(start)
    dist = np.abs(land.x[new] - 5) + np.abs(land.y[new] - 5)
    assert dist.max() == 1 and abs((dist == 1).mean() - 0.5) < 0.02


def test_island_migrants_leave_their_cell():
    rng = np.random.default_rng(6)
    land = Landscape(5, 1, K=10)
    pop = EcoPopulation.founders(_neutral(rng), land, rng, [0], EcoParams(dispersal="island", migration_rate=1.0))
    new = pop._disperse(np.zeros(5000, int))
    assert (new != 0).all()
    assert np.allclose(np.bincount(new, minlength=5)[1:] / 5000, 0.25, atol=0.03)


def test_niche_mismatch_reduces_fitness():
    rng = np.random.default_rng(7)
    arch = codon_architecture(rng, n_traits=1, env_noise_sd=0.0)
    land = Landscape(2, 1, K=50, layers={"temp": [[0.0, 2.0]]})
    pop = EcoPopulation.founders(arch, land, rng, [0] * 5 + [1] * 5,
                                 EcoParams(niche=[NicheAxis(0, "temp", 1.0)]))
    w, _ = pop.fitness()
    assert np.allclose(w[:5], 1.0) and np.allclose(w[5:], np.exp(-2.0))


def test_fst_extremes():
    anc = np.zeros(4, np.uint8)
    g = np.zeros((20, 2, 4), np.uint8)
    g[10:] = 2                                   # dos grupos fijados para alelos distintos
    groups = np.repeat([0, 1], 10)
    assert abs(stats.fst(g, groups, anc) - 1.0) < 1e-9
    rng = np.random.default_rng(8)
    g = (rng.random((2000, 2, 50)) < 0.5).astype(np.uint8) * 2
    assert abs(stats.fst(g, rng.integers(0, 4, 2000), np.zeros(50, np.uint8))) < 0.01
