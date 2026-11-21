"""Expansión de rango sobre un gradiente térmico (adaptación local).

Una población fundadora adaptada a temperatura 0 se establece en el centro de
un paisaje cuya temperatura va de frío (−2) a caliente (+2). Fuera de su
óptimo los organismos casi no se reproducen, así que la especie solo puede
expandirse si EVOLUCIONA: mutaciones en promotores y genes de rasgo cambian su
tolerancia térmica, y la selección local las favorece en cada borde.

Fenómenos esperados (no programados):
  - Clina adaptativa: el rasgo medio sigue al gradiente de temperatura.
  - Expansión lenta y escalonada, limitada por la tasa de adaptación.
  - Desajuste en los bordes por flujo génico desde el centro (Kirkpatrick & Barton 1997).

Uso:  python3 examples/range_expansion.py [--generations 1500] [--gif]
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gif", action="store_true", help="también genera una animación")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    W, H, K = 30, 3, 30
    arch = codon_architecture(
        rng, n_chrom=4, n_enzyme_genes=6, n_trait_genes=4, n_traits=1, codons_per_gene=60,
        promoter_len=20, intergenic_len=40, mutation_rate=1e-4, promoter_effect_sd=0.1, env_noise_sd=0.1,
    )
    temp = Landscape.gradient(W, H, -2.0, 2.0)
    land = Landscape(W, H, K=K, layers={"temp": temp})
    params = EcoParams(fecundity=2.5, niche=[NicheAxis(0, "temp", omega=0.6)],
                       dispersal="stepping_stone", migration_rate=0.1, selfing=False)
    center = land.cell(np.array([W // 2 - 1, W // 2] * H), np.repeat(np.arange(H), 2))
    pop = EcoPopulation.founders(arch, land, rng, np.repeat(center, K), params)

    T = args.generations
    density = np.zeros((T, W))
    trait_by_x = np.full((T, W), np.nan)
    sizes = []
    t0 = time.time()
    for t in range(T):
        pop.step()
        sizes.append(pop.N)
        if pop.N == 0:
            print("extinción en la generación", t)
            break
        x = land.x[pop.cells]
        density[t] = np.bincount(x, minlength=W) / (H * K)
        _, traits = arch.gmap.evaluate(pop.genomes, None)   # valor genético (sin ruido ambiental)
        s = np.bincount(x, weights=traits[:, 0], minlength=W)
        c = np.bincount(x, minlength=W)
        trait_by_x[t] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        if (t + 1) % 250 == 0:
            occ = (density[t] > 0.1).sum()
            print(f"gen {t + 1}: N = {pop.N}, columnas ocupadas = {occ}/{W}  ({time.time() - t0:.0f}s)", flush=True)

    temp_x = temp[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    im = ax.imshow(density.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, T, temp_x[0], temp_x[-1]])
    ax.set_xlabel("generación")
    ax.set_ylabel("temperatura del lugar")
    ax.set_title("Densidad a lo largo del gradiente")
    fig.colorbar(im, ax=ax, label="densidad / K")

    ax = axes[1]
    snaps = [t for t in (50, T // 4, T // 2, T - 1) if t < T]
    for k, t in enumerate(snaps):
        ax.plot(temp_x, trait_by_x[t], "o-", ms=3, color=plt.cm.plasma(k / len(snaps)), label=f"gen {t + 1}")
    ax.plot(temp_x, temp_x, "k--", lw=1, label="óptimo local")
    ax.set_xlabel("temperatura del lugar")
    ax.set_ylabel("rasgo genético medio (tolerancia)")
    ax.set_title("Formación de una clina adaptativa")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(sizes)
    ax.axhline(W * H * K, color="gray", lw=0.8, label="capacidad total")
    ax.set_xlabel("generación")
    ax.set_ylabel("tamaño poblacional")
    ax.set_title("Crecimiento de la especie")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "range_expansion.png", dpi=120)
    print("figura:", OUT / "range_expansion.png")

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
        frames = list(range(0, T, max(T // 120, 1)))

        def draw(t):
            a1.clear()
            a2.clear()
            a1.bar(temp_x, density[t], width=temp_x[1] - temp_x[0], color="C2")
            a1.set_ylim(0, 1.2)
            a1.set_ylabel("densidad / K")
            a1.set_title(f"Generación {t + 1}   N = {sizes[t]}")
            a2.plot(temp_x, temp_x, "k--", lw=1, label="óptimo local")
            a2.plot(temp_x, trait_by_x[t], "o-", ms=3, color="C3", label="rasgo medio")
            a2.set_ylim(-2.3, 2.3)
            a2.set_xlabel("temperatura del lugar")
            a2.set_ylabel("tolerancia térmica")
            a2.legend(loc="upper left", fontsize=8)

        FuncAnimation(fig, draw, frames=frames).save(OUT / "range_expansion.gif", writer=PillowWriter(fps=12))
        print("animación:", OUT / "range_expansion.gif")


if __name__ == "__main__":
    main()
