"""Cambio climático: ¿la especie se adapta a tiempo o se extingue?

Una población adaptada a temperatura 0 acumula variación genética durante un
periodo estable. Luego la temperatura sube a una velocidad constante. Se comparan
varias velocidades de calentamiento con la misma población inicial.

Teoría (Lynch & Lande 1993; Bürger & Lynch 1995): existe una tasa crítica de
cambio ambiental. Por debajo, la población sigue al óptimo con un retraso
constante; por encima, el retraso crece, la fecundidad cae bajo el reemplazo
y la población se extingue. La tasa crítica crece con la varianza genética
aditiva y con la tasa reproductiva.

Uso:  python3 examples/climate_change.py
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evosim import EcoParams, EcoPopulation, Landscape, NicheAxis, codon_architecture  # noqa: E402

OUT = ROOT / "examples" / "output"
OUT.mkdir(parents=True, exist_ok=True)


def warming(rate, start):
    def update(land, gen):
        if gen >= start:
            land.layers["temp"][:] = rate * (gen - start + 1)
    return update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burn", type=int, default=400)
    ap.add_argument("--generations", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    W, H, K = 10, 2, 50
    arch = codon_architecture(
        rng, n_chrom=4, n_enzyme_genes=6, n_trait_genes=4, n_traits=1, codons_per_gene=60,
        promoter_len=20, intergenic_len=40, mutation_rate=1e-4, promoter_effect_sd=0.1, env_noise_sd=0.1,
    )
    land = Landscape(W, H, K=K, layers={"temp": np.zeros((H, W))})
    params = EcoParams(fecundity=2.5, niche=[NicheAxis(0, "temp", omega=0.6)],
                       dispersal="stepping_stone", migration_rate=0.1, selfing=False)
    base = EcoPopulation.founders(arch, land, rng, np.repeat(np.arange(land.n_cells), K), params)
    t0 = time.time()
    base.run(args.burn)
    _, z = arch.gmap.evaluate(base.genomes, None)
    print(f"tras {args.burn} generaciones estables: N = {base.N}, varianza genética del rasgo = {z[:, 0].var():.4f}"
          f"  ({time.time() - t0:.0f}s)", flush=True)

    rates = [0.0, 0.002, 0.005, 0.01, 0.02]
    results = {}
    for rate in rates:
        pop = copy.deepcopy(base)
        pop.rng = np.random.default_rng(args.seed + int(rate * 1e4))
        pop.landscape.add_updater(warming(rate, pop.generation))
        N, zm, opt, vg = [], [], [], []
        for _ in range(args.generations):
            pop.step()
            N.append(pop.N)
            opt.append(pop.landscape.layers["temp"][0])
            if pop.N:
                _, z = arch.gmap.evaluate(pop.genomes, None)
                zm.append(z[:, 0].mean())
                vg.append(z[:, 0].var())
            else:
                zm.append(np.nan)
                vg.append(np.nan)
        results[rate] = (np.array(N), np.array(zm), np.array(opt), np.array(vg))
        fate = "EXTINTA en gen %d" % (np.argmax(np.array(N) == 0) + 1) if N[-1] == 0 else f"N final = {N[-1]}"
        print(f"calentamiento {rate}/gen: {fate}  ({time.time() - t0:.0f}s)", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, rate in enumerate(rates):
        N, zm, opt, vg = results[rate]
        c = plt.cm.coolwarm(k / (len(rates) - 1))
        axes[0].plot(N, color=c, label=f"{rate}/gen")
        axes[1].plot(opt - zm, color=c)
        axes[2].plot(vg, color=c)
    axes[0].set_title("Tamaño poblacional")
    axes[0].set_ylabel("N")
    axes[0].legend(title="calentamiento", fontsize=8)
    axes[1].set_title("Retraso adaptativo (óptimo − rasgo medio)")
    axes[1].set_ylabel("desajuste")
    axes[1].axhline(0.6 * np.sqrt(2 * np.log(params.fecundity)), color="gray", ls=":", lw=1)
    axes[1].text(5, 0.6 * np.sqrt(2 * np.log(params.fecundity)) + 0.03, "desajuste con λw = 1 (reemplazo)",
                 fontsize=8, color="gray")
    axes[2].set_title("Varianza genética del rasgo")
    axes[2].set_ylabel("V_G")
    for ax in axes:
        ax.set_xlabel("generaciones desde el inicio del calentamiento")
    fig.tight_layout()
    fig.savefig(OUT / "climate_change.png", dpi=120)
    print("figura:", OUT / "climate_change.png")


if __name__ == "__main__":
    main()
