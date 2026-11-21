"""Demo fase 3: especiación alopátrica por coevolución de factores de transcripción y sus
sitios de unión.

Una especie ancestral se divide varias veces (vicarianza). Cada linaje evoluciona aislado,
bajo la misma selección. No se programa ninguna incompatibilidad: cada linaje deriva por su
propia "red neutral" de combinaciones TF–sitio que funcionan, y los híbridos combinan un TF
de un linaje con sitios del otro.

Al final:
  - árbol verdadero vs. Neighbor-Joining desde los genomas;
  - aislamiento postcigótico (F1, F2) entre cada par vs. tiempo de divergencia (Coyne & Orr 1989);
  - número de DMIs vs. diferencias fijadas (la "bola de nieve" de Orr 1995);
  - fitness de cada clase de cruza para el par más divergente.

Antes de la primera vicarianza la especie ancestral corre hasta el equilibrio
mutación-selección-deriva (fitness estable); si no, la carga genética que sigue
acumulándose confundiría la comparación híbridos vs. parentales. Ese calentamiento
(~20 000 generaciones) se guarda en examples/output/ y se reutiliza.

Uso:  python3 examples/speciation.py [--generations 10000] [--seed 5]
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evosim import Population, StabilizingSelection, codon_architecture, stats  # noqa: E402
from evosim import phylo  # noqa: E402
from evosim.architecture import INTERGENIC  # noqa: E402
from evosim.speciation import (Radiation, count_dmis, hybrid_fitness,  # noqa: E402
                               isolation_index)

OUT = ROOT / "examples" / "output"


def build(seed, mu):
    rng = np.random.default_rng(seed)
    arch = codon_architecture(rng, n_enzyme_genes=10, n_trait_genes=4, n_traits=1, n_tf=5, tf_targets=2, tf_mu=6.0, tf_target_roles=("enzyme",),
                              codons_per_gene=40, promoter_len=20, intergenic_len=40,
                              mutation_rate=mu, promoter_effect_sd=0.1)
    return rng, arch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=10000)
    ap.add_argument("--burn", type=int, default=20000)
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--mu", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T = args.generations
    rng, arch = build(args.seed, args.mu)
    env = StabilizingSelection([0.0], 1.0)
    cache = OUT / f"speciation_burnin_seed{args.seed}.npy"
    if cache.exists():
        genomes = np.load(cache)
        print(f"calentamiento leído de {cache}")
    else:
        anc = Population(arch, args.N, rng, environment=env, selfing=False)
        for k in range(args.burn // 1000):
            anc.run(1000)
            print(f"calentamiento {anc.generation}: fitness medio "
                  f"{np.exp(anc.evaluate()[0]).mean():.3f}", flush=True)
        genomes = anc.genomes
        np.save(cache, genomes)
    rad = Radiation(arch, args.N, rng, env, genomes=genomes)

    # Vicarianzas: la primera al inicio, las demás repartidas (árbol asimétrico de 5 especies)
    schedule = {0: "L0", int(T * 0.25): "L1", int(T * 0.5): "L2", int(T * 0.75): "L3"}
    t0 = time.time()
    for t in range(T):
        if t in schedule:
            kids = rad.split(schedule[t])
            print(f"gen {t}: {schedule[t]} -> {kids}", flush=True)
        rad.step()
        if (t + 1) % 500 == 0:
            print(f"gen {t + 1}  linajes {len(rad.tips)}  {time.time() - t0:.0f}s", flush=True)

    tips = sorted((l.name for l in rad.tips), key=lambda n: int(n[1:]))
    G = {n: rad.lineages[n].pop.genomes for n in tips}
    neutral = np.flatnonzero(arch.site_class == INTERGENIC)

    def split_time(x, y):
        def path(n):
            out = []
            while n is not None:
                out.append(n)
                n = rad.lineages[n].parent
            return out
        common = next(n for n in path(x) if n in path(y))
        return T - rad.lineages[rad.lineages[common].children[0]].born

    jc = lambda p: float(stats.jukes_cantor(np.asarray(p)))
    pis = {n: jc(stats.nucleotide_diversity(G[n][:, :, neutral])) for n in tips}
    rows = []
    n = len(tips)
    DA = np.zeros((n, n))
    for (i, x), (j, y) in itertools.combinations(enumerate(tips), 2):
        dxy = jc(phylo.mean_pairwise_distance(G[x], G[y], sites=neutral, rng=rng, max_pairs=4000))
        DA[i, j] = DA[j, i] = dxy - 0.5 * (pis[x] + pis[y])
        hf = hybrid_fitness(arch, G[x], G[y], rng, env, n=1000)
        n_dmi, K, _ = count_dmis(arch, G[x], G[y])
        rows.append({"a": x, "b": y, "t": split_time(x, y), "dA": DA[i, j], "K": K, "dmi": n_dmi,
                     "RI_F1": isolation_index(hf, "F1"), "RI_F2": isolation_index(hf, "F2"), **hf})
        r = rows[-1]
        print(f"{x}×{y}: t={r['t']}  d_A={r['dA']:.4f}  K={K}  DMIs={n_dmi}  "
              f"RI_F1={r['RI_F1']:.3f}  RI_F2={r['RI_F2']:.3f}", flush=True)

    tree = phylo.neighbor_joining(DA, tips)
    true = phylo.collapse_unary(phylo.parse_newick(rad.true_tree()))
    rf = phylo.robinson_foulds(tree, true)
    print("Árbol verdadero: ", true.newick(4))
    print("Neighbor-Joining:", tree.newick(4))
    print("Robinson-Foulds: ", rf)

    # ------------------------------------------------------------------ figura
    t = np.array([r["t"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    ax.plot(t, [r["RI_F2"] for r in rows], "o", label="F2")
    ax.plot(t, [r["RI_F1"] for r in rows], "s", mfc="none", label="F1")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("generaciones desde el ancestro común")
    ax.set_ylabel("aislamiento postcigótico RI = 1 − w_híbrido / w_parental")
    ax.set_title("Aislamiento vs. tiempo (Coyne & Orr)")
    ax.legend()

    ax = axes[0, 1]
    K = np.array([r["K"] for r in rows])
    D = np.array([r["dmi"] for r in rows])
    ax.plot(K, D, "o")
    if (D > 0).sum() >= 3:
        m = D > 0
        a, b = np.polyfit(np.log(K[m]), np.log(D[m]), 1)
        kk = np.linspace(K[m].min(), K.max(), 50)
        ax.plot(kk, np.exp(b) * kk ** a, "-", label=f"ajuste: DMIs ∝ K^{a:.2f}")
        ax.legend()
    ax.set_xlabel("diferencias fijadas en sitios funcionales (K)")
    ax.set_ylabel("DMIs detectadas")
    ax.set_title("Bola de nieve de incompatibilidades (Orr)")

    ax = axes[1, 0]
    far = max(rows, key=lambda r: r["t"])
    cls = ["A", "B", "F1", "F2", "BC_A", "BC_B"]
    ax.bar(cls, [far[c] for c in cls], color=["C0", "C0", "C2", "C3", "C1", "C1"])
    ax.set_ylabel("fitness medio")
    ax.set_title(f"Cruzas {far['a']} × {far['b']} ({far['t']} generaciones de divergencia)")
    ax.set_ylim(0, max(far[c] for c in cls) * 1.15)

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0, 0.95, "Árbol verdadero (generaciones):\n" + true.newick(4), va="top", family="monospace", fontsize=8, wrap=True)
    ax.text(0, 0.55, "Neighbor-Joining (d_A neutral):\n" + tree.newick(3), va="top", family="monospace", fontsize=8, wrap=True)
    ax.text(0, 0.15, f"Robinson-Foulds = {rf}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "speciation.png", dpi=120)
    (OUT / "speciation.json").write_text(json.dumps(
        {"true_tree": true.newick(6), "nj_tree": tree.newick(6), "rf": rf, "pairs": rows}, indent=1))
    print(f"-> {OUT / 'speciation.png'}")


if __name__ == "__main__":
    main()
