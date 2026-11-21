"""Graba escenarios para el visor web (viewer/).

Uso:
    python3 examples/record_run.py gradient     # expansión de rango sobre un gradiente térmico
    python3 examples/record_run.py climate      # calentamiento rápido hasta la extinción
    python3 examples/record_run.py gradient --generations 300   # corrida corta de prueba

Escribe viewer/data/<escenario>.json
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evosim import EcoParams, EcoPopulation, Landscape, NicheAxis, codon_architecture  # noqa: E402
from evosim.recorder import Recorder  # noqa: E402

DATA = ROOT / "viewer" / "data"
DATA.mkdir(parents=True, exist_ok=True)


def _arch(rng):
    return codon_architecture(
        rng, n_chrom=4, n_enzyme_genes=6, n_trait_genes=4, n_traits=1, codons_per_gene=60,
        promoter_len=20, intergenic_len=40, mutation_rate=1e-4, promoter_effect_sd=0.1, env_noise_sd=0.1,
    )


def gradient(generations, seed):
    rng = np.random.default_rng(seed)
    W, H, K = 30, 3, 30
    arch = _arch(rng)
    land = Landscape(W, H, K=K, layers={"temp": Landscape.gradient(W, H, -2.0, 2.0)})
    params = EcoParams(fecundity=2.5, niche=[NicheAxis(0, "temp", omega=0.6)],
                       dispersal="stepping_stone", migration_rate=0.1, selfing=False)
    center = land.cell(np.array([W // 2 - 1, W // 2] * H), np.repeat(np.arange(H), 2))
    pop = EcoPopulation.founders(arch, land, rng, np.repeat(center, K), params)
    rec = Recorder(
        pop, "gradient", "Expansión sobre un gradiente térmico",
        "Una especie adaptada a temperatura 0 coloniza un paisaje de −2 a +2. "
        "Solo puede ocupar lugares nuevos si su tolerancia térmica evoluciona.",
    )
    return rec, generations or 1500


def climate(generations, seed, rate=0.02, burn=400):
    rng = np.random.default_rng(seed)
    W, H, K = 10, 2, 50
    arch = _arch(rng)
    land = Landscape(W, H, K=K, layers={"temp": np.zeros((H, W))})
    params = EcoParams(fecundity=2.5, niche=[NicheAxis(0, "temp", omega=0.6)],
                       dispersal="stepping_stone", migration_rate=0.1, selfing=False)
    pop = EcoPopulation.founders(arch, land, rng, np.repeat(np.arange(land.n_cells), K), params)

    def warming(l, gen):
        if gen >= burn:
            l.layers["temp"][:] = rate * (gen - burn + 1)
    land.add_updater(warming)
    rec = Recorder(
        pop, "climate", "Calentamiento rápido",
        f"La población vive {burn} generaciones a temperatura estable y luego el clima "
        f"se calienta {rate} unidades por generación.",
    )
    rec.meta["warming_start"] = burn
    rec.meta["warming_rate"] = rate
    return rec, generations or (burn + 600)


SCENARIOS = {"gradient": gradient, "climate": climate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=list(SCENARIOS))
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else {"gradient": 7, "climate": 11}[args.scenario]
    rec, T = SCENARIOS[args.scenario](args.generations, seed)
    t0 = time.time()

    def progress(pop):
        if pop.generation % 100 == 0:
            print(f"gen {pop.generation}/{T}: N = {pop.N}  ({time.time() - t0:.0f}s)", flush=True)

    rec.run(T, progress)
    out = DATA / f"{args.scenario}.json"
    rec.save(out)
    print(f"guardado {out} ({out.stat().st_size / 1024:.0f} KB, {len(rec.frames)} cuadros)")


if __name__ == "__main__":
    main()
