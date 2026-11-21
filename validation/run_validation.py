"""Banco de validación: compara el simulador con resultados teóricos conocidos.

Uso:
    python3 validation/run_validation.py            # completo (~10 min)
    python3 validation/run_validation.py --quick    # rápido, menos réplicas
    python3 validation/run_validation.py --only kimura drift

Cada experimento produce una figura en validation/figures/ y una fila en
validation/RESULTS.md con el valor teórico, el observado y si pasa.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evosim import (  # noqa: E402
    EcoParams, EcoPopulation, GenomeLayout, Landscape, Population, SiteEffectsMap,
    codon_architecture, neutral_architecture, stats,
)
from evosim.architecture import CODING, GenomeArchitecture  # noqa: E402
from evosim.maps import NeutralMap  # noqa: E402

FIG = ROOT / "validation" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SCALE = 1.0


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- 1
def hardy_weinberg(rng):
    """Una población de puros homocigotos alcanza H-W en una sola generación de apareamiento aleatorio."""
    N, n_sites = 5000, 200
    arch = neutral_architecture(rng, 20, n_sites // 20, 1.0)
    p = rng.uniform(0.1, 0.9, n_sites)
    allele = (rng.random((N, n_sites)) < p).astype(np.uint8)
    derived = (arch.ancestral + 1) % 4
    hap = np.where(allele, derived, arch.ancestral).astype(np.uint8)
    genomes = np.stack([hap, hap], axis=1)                    # todos homocigotos
    pop = Population(arch, N, rng, genomes=genomes)
    het0 = stats.observed_heterozygosity(pop.genomes).mean()
    p_par = stats.derived_freq(pop.genomes, arch.ancestral)
    pop.step()
    chi2 = 0.0
    obs_het, exp_het = [], []
    for i in range(n_sites):
        cnt = stats.genotype_counts(pop.genomes, i, arch.ancestral[i])
        q = cnt @ [0, 1, 2] / (2 * N)
        exp = N * np.array([(1 - q) ** 2, 2 * q * (1 - q), q ** 2])
        chi2 += ((cnt - exp) ** 2 / exp).sum()
        obs_het.append(cnt[1] / N)
        exp_het.append(2 * p_par[i] * (1 - p_par[i]))
    pval = sps.chi2.sf(chi2, df=n_sites)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(exp_het, obs_het, s=10, alpha=0.7)
    ax.plot([0, 0.5], [0, 0.5], "k--", lw=1)
    ax.set_xlabel("2pq esperado (frecuencias de los padres)")
    ax.set_ylabel("Heterocigotos observados en los hijos")
    ax.set_title(f"Hardy-Weinberg tras 1 generación\n(heterocigosidad inicial = {het0:.0%})")
    _save(fig, "01_hardy_weinberg")
    return {
        "nombre": "Equilibrio de Hardy-Weinberg",
        "teoria": "Genotipos p², 2pq, q² tras 1 generación",
        "observado": f"χ² = {chi2:.0f} con {n_sites} g.l. (p = {pval:.2f})",
        "pasa": pval > 0.001,
    }


# ----------------------------------------------------------------------------- 2
def drift(rng):
    """La heterocigosidad decae como H_t = H_0 (1 - 1/2N)^t."""
    N, T = 50, 200
    arch = neutral_architecture(rng, 40, 50, 1.0)
    allele = rng.random((N, 2, arch.layout.L)) < 0.5
    genomes = np.where(allele, (arch.ancestral + 1) % 4, arch.ancestral).astype(np.uint8)
    pop = Population(arch, N, rng, genomes=genomes)
    H = [stats.expected_heterozygosity(pop.genomes).mean()]
    for _ in range(T):
        pop.step()
        H.append(stats.expected_heterozygosity(pop.genomes).mean())
    H = np.array(H)
    t = np.arange(T + 1)
    theory = H[0] * (1 - 1 / (2 * N)) ** t
    ok = H > 0.02
    slope = np.polyfit(t[ok], np.log(H[ok]), 1)[0]
    Ne_est = -1 / (2 * slope)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, H, label="simulación")
    ax.plot(t, theory, "k--", label="teoría  H₀(1 − 1/2N)ᵗ")
    ax.set_xlabel("generación")
    ax.set_ylabel("heterocigosidad esperada H")
    ax.set_title(f"Deriva genética (N = {N})")
    ax.legend()
    _save(fig, "02_deriva")
    return {
        "nombre": "Pérdida de variación por deriva",
        "teoria": f"Ne = N = {N}",
        "observado": f"Ne estimado = {Ne_est:.1f}",
        "pasa": abs(Ne_est - N) / N < 0.15,
    }


# ----------------------------------------------------------------------------- 3
def neutral_fixation(rng):
    """Una mutación neutra nueva se fija con probabilidad 1/2N, en un tiempo medio ≈ 4N."""
    N = 25
    arch = neutral_architecture(rng, 40, 100, 1.0)
    L = arch.layout.L
    genomes = np.broadcast_to(arch.ancestral, (N, 2, L)).copy()
    who = rng.integers(0, N, L)
    strand = rng.integers(0, 2, L)
    genomes[who, strand, np.arange(L)] = (arch.ancestral + 1) % 4
    pop = Population(arch, N, rng, genomes=genomes)
    fix_time = np.full(L, -1)
    lost = np.zeros(L, bool)
    while pop.generation < 60 * N:
        pop.step()
        f = stats.derived_freq(pop.genomes, arch.ancestral)
        newly = (f == 1) & (fix_time < 0)
        fix_time[newly] = pop.generation
        lost |= f == 0
        if ((fix_time >= 0) | lost).all():
            break
    n_fix = int((fix_time >= 0).sum())
    p_exp = 1 / (2 * N)
    p_test = sps.binomtest(n_fix, L, p_exp).pvalue
    times = fix_time[fix_time >= 0]
    t_exp = 4 * N * (1 - p_exp) * -np.log(1 - p_exp) / p_exp  # Kimura & Ohta 1969
    t_z = (times.mean() - t_exp) / (times.std(ddof=1) / np.sqrt(len(times)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(times, bins=20, alpha=0.8)
    ax.axvline(t_exp, color="k", ls="--", label=f"teoría: {t_exp:.0f}")
    ax.axvline(times.mean(), color="C3", label=f"simulación: {times.mean():.0f}")
    ax.set_xlabel("generaciones hasta la fijación")
    ax.set_ylabel("mutaciones")
    ax.set_title(f"Fijación de mutaciones neutras (N = {N})\n{n_fix}/{L} fijadas; esperado {L * p_exp:.0f}")
    ax.legend()
    _save(fig, "03_fijacion_neutra")
    return {
        "nombre": "Fijación de mutaciones neutras",
        "teoria": f"P = 1/2N = {p_exp:.3f};  t̄ = {t_exp:.0f} gen",
        "observado": f"P = {n_fix / L:.3f} (p = {p_test:.2f});  t̄ = {times.mean():.0f} gen (z = {t_z:.1f})",
        "pasa": p_test > 0.001 and abs(t_z) < 3.5,
    }


# ----------------------------------------------------------------------------- 4
def kimura_u(N, s, h, p):
    """Probabilidad de fijación (aproximación de difusión de Kimura 1962) con dominancia."""
    if s == 0:
        return p
    G = lambda x: np.exp(-4 * N * s * (h * x + (1 - 2 * h) * x ** 2 / 2))
    return integrate.quad(G, 0, p)[0] / integrate.quad(G, 0, 1)[0]


def _single_site(N, s, h):
    layout = GenomeLayout([1], 0.0)
    anc = np.zeros(1, np.uint8)
    return GenomeArchitecture(layout, anc, SiteEffectsMap.single(0, 0, s, h))


def kimura(rng):
    """Probabilidad de fijación de una mutación nueva según su efecto (s) y dominancia (h)."""
    N = 100
    R = int(2000 * SCALE)
    conditions = [(s, 0.5) for s in (-0.01, 0.0, 0.01, 0.02, 0.05, 0.1)] + [(0.05, 0.1), (0.05, 0.9)]
    rows = []
    for s, h in conditions:
        arch = _single_site(N, s, h)
        n_fix = 0
        for _ in range(R):
            g = np.zeros((N, 2, 1), np.uint8)
            g[0, 0, 0] = 2
            pop = Population(arch, N, rng, genomes=g)
            while True:
                pop.step()
                k = int((pop.genomes != 0).sum())
                if k == 0:
                    break
                if k == 2 * N:
                    n_fix += 1
                    break
        u = kimura_u(N, s, h, 1 / (2 * N))
        pv = sps.binomtest(n_fix, R, u).pvalue
        rows.append((s, h, n_fix / R, u, pv, n_fix))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    add = [r for r in rows if r[1] == 0.5]
    ss = np.linspace(-0.012, 0.105, 200)
    ax.plot(ss, [kimura_u(N, x, 0.5, 1 / (2 * N)) for x in ss], "k--", label="Kimura (h = 0.5)")
    ax.errorbar([r[0] for r in add], [r[2] for r in add],
                yerr=[1.96 * np.sqrt(r[3] * (1 - r[3]) / R) for r in add], fmt="o", label="simulación aditiva")
    for r, mk, lab in [(rows[-2], "v", "recesiva h = 0.1"), (rows[-1], "^", "dominante h = 0.9")]:
        ax.plot(r[0], r[2], mk, ms=9, label=f"sim. {lab}")
        ax.plot(r[0], r[3], mk, ms=9, mfc="none", mec="k")
    ax.axhline(1 / (2 * N), color="gray", lw=0.8)
    ax.set_xlabel("coeficiente de selección s (homocigoto)")
    ax.set_ylabel("probabilidad de fijación")
    ax.set_title(f"Fijación de mutaciones nuevas (N = {N}, {R} réplicas)\n"
                 "marcadores huecos = teoría con dominancia (tamiz de Haldane)")
    ax.legend(fontsize=8)
    _save(fig, "04_kimura")

    worst = min(r[4] for r in rows)
    detail = "; ".join(f"s={r[0]:+.2f},h={r[1]}: {r[2]:.3f} vs {r[3]:.3f}" for r in rows)
    return {
        "nombre": "Probabilidad de fijación (Kimura) y tamiz de Haldane",
        "teoria": "Difusión de Kimura con dominancia",
        "observado": detail + f"  (p mínimo = {worst:.3f})",
        "pasa": worst > 0.001 / len(rows),
    }


# ----------------------------------------------------------------------------- 5
def mutation_drift(rng):
    """Diversidad en equilibrio mutación-deriva: H = 3θ/(3 + 4θ), θ = 4Nμ (modelo de 4 alelos)."""
    N = 100
    burn, meas = 1500, int(2000 * SCALE)
    rows = []
    for mu in (5e-5, 2.5e-4, 1e-3):
        arch = neutral_architecture(rng, 20, 100, 1.0, mutation_rate=mu, kappa=1.0)
        pop = Population(arch, N, rng)
        pop.run(burn)
        vals = []
        pop.run(meas, callback=lambda p: vals.append(stats.expected_heterozygosity(p.genomes).mean()), every=10)
        theta = 4 * N * mu
        rows.append((theta, np.mean(vals), 3 * theta / (3 + 4 * theta)))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    th = np.linspace(0, 0.45, 100)
    ax.plot(th, 3 * th / (3 + 4 * th), "k--", label="teoría")
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o", ms=8, label="simulación")
    ax.set_xlabel("θ = 4Nμ")
    ax.set_ylabel("heterocigosidad en equilibrio")
    ax.set_title("Equilibrio mutación-deriva")
    ax.legend()
    _save(fig, "05_mutacion_deriva")
    rel = max(abs(r[1] - r[2]) / r[2] for r in rows)
    return {
        "nombre": "Equilibrio mutación-deriva",
        "teoria": "H = 3θ/(3+4θ)",
        "observado": "; ".join(f"θ={r[0]:.2f}: {r[1]:.4f} vs {r[2]:.4f}" for r in rows),
        "pasa": rel < 0.15,
    }


# ----------------------------------------------------------------------------- 6
def linkage(rng):
    """El desequilibrio de ligamiento decae como D_t = D_0 (1 - r)^t."""
    N, T = 5000, 40
    layout = GenomeLayout([21, 1], [0.42, 0.0])   # sitios separados 0.02 M + uno en otro cromosoma
    anc = np.zeros(layout.L, np.uint8)
    arch = GenomeArchitecture(layout, anc, NeutralMap())
    hap = np.zeros((2 * N, layout.L), np.uint8)
    hap[: N] = 2                                    # mitad todo-derivado, mitad todo-ancestral: D = 0.25
    hap = hap[rng.permutation(2 * N)]
    pop = Population(arch, N, rng, genomes=hap.reshape(N, 2, layout.L))
    pairs = [1, 5, 10, 20, 21]
    D = {j: [stats.linkage_disequilibrium(pop.genomes, anc, 0, j)] for j in pairs}
    for _ in range(T):
        pop.step()
        for j in pairs:
            D[j].append(stats.linkage_disequilibrium(pop.genomes, anc, 0, j))
    t = np.arange(T + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    errs = []
    for k, j in enumerate(pairs):
        r = layout.recombination_fraction(0, j)
        d = np.array(D[j])
        theo = d[0] * (1 - r) ** t
        ax.plot(t, d, color=f"C{k}", label=f"r = {r:.3f}")
        ax.plot(t, theo, "--", color=f"C{k}", lw=1)
        # ajustar solo la ventana inicial donde D >> ruido de muestreo (~0.0025)
        end = int(np.argmax(np.r_[d, 0.0] < 0.03))
        r_est = 1 - np.exp(np.polyfit(t[:end], np.log(d[:end]), 1)[0]) - 1 / (2 * N)
        errs.append((r, r_est))
    ax.set_xlabel("generación")
    ax.set_ylabel("desequilibrio de ligamiento D")
    ax.set_title("Recombinación rompe asociaciones entre loci\n(líneas punteadas = teoría)")
    ax.legend()
    _save(fig, "06_ligamiento")
    return {
        "nombre": "Decaimiento del desequilibrio de ligamiento",
        "teoria": "D_t = D₀(1 − r)ᵗ, r de Haldane",
        "observado": "; ".join(f"r={a:.3f}→{b:.3f}" for a, b in errs),
        "pasa": all(abs(a - b) < 0.1 * a + 0.005 for a, b in errs),
    }


# ----------------------------------------------------------------------------- 7
def _recursion(p, s, h):
    q = 1 - p
    wbar = p * p * (1 + s) + 2 * p * q * (1 + h * s) + q * q
    return p * (p * (1 + s) + q * (1 + h * s)) / wbar


def deterministic_selection(rng):
    """Cada generación, p' debe seguir la recursión p' = p(p w11 + q w12)/w̄ más ruido de deriva.

    Se prueba paso a paso: z = (p_obs - p_esperado) / sqrt(p(1-p)/2N) debe ser ~ N(0, 1).
    (Comparar trayectorias completas no sirve: al inicio de un barrido recesivo la deriva
    domina y desplaza en el tiempo toda la curva, aunque cada paso sea correcto.)"""
    N, T, s, p0 = 10000, 300, 0.1, 0.05
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    zs = []
    for k, (h, lab) in enumerate([(0.0, "recesiva"), (0.5, "aditiva"), (1.0, "dominante")]):
        arch = _single_site(N, s, h)
        for rep in range(2):
            g = (rng.random((N, 2, 1)) < p0).astype(np.uint8) * 2
            pop = Population(arch, N, rng, genomes=g)
            sim = [stats.derived_freq(pop.genomes, arch.ancestral)[0]]
            for _ in range(T):
                pop.step()
                sim.append(stats.derived_freq(pop.genomes, arch.ancestral)[0])
            sim = np.array(sim)
            f = _recursion(sim[:-1], s, h)
            ok = (f > 0) & (f < 1)
            zs.append((sim[1:][ok] - f[ok]) / np.sqrt(f[ok] * (1 - f[ok]) / (2 * N)))
            traj = [sim[0]]
            for _ in range(T):
                traj.append(_recursion(traj[-1], s, h))
            ax.plot(sim, color=f"C{k}", alpha=0.8, label=f"{lab} (h = {h})" if rep == 0 else None)
            ax.plot(traj, "--", color=f"C{k}", lw=1)
    z = np.concatenate(zs)
    tstat = z.mean() * np.sqrt(len(z))
    ax.set_xlabel("generación")
    ax.set_ylabel("frecuencia del alelo beneficioso")
    ax.set_title(f"Selección con distinta dominancia (s = {s}, N = {N})\n"
                 f"punteado = recursión determinista; residuos z: media {z.mean():+.2f}, d.e. {z.std():.2f}")
    ax.legend()
    _save(fig, "07_seleccion_dominancia")
    return {
        "nombre": "Respuesta a la selección generación a generación",
        "teoria": "p' = p(p·w₁₁ + q·w₁₂)/w̄;  residuos z ~ N(0,1)",
        "observado": f"z: media = {z.mean():+.3f}, d.e. = {z.std():.3f} ({len(z)} pasos)",
        "pasa": abs(tstat) < 3.5 and 0.85 < z.std() < 1.15,
    }


# ----------------------------------------------------------------------------- 8
def emergent_dominance(rng):
    """Modelo molecular: la dominancia emerge de la cinética enzimática (Kacser & Burns 1981).
    Patrón empírico esperado: mutaciones de gran efecto son más recesivas (Agrawal & Whitlock 2011)."""
    arch = codon_architecture(rng, n_trait_genes=0, n_enzyme_genes=30)
    gm = arch.gmap
    base = np.broadcast_to(arch.ancestral, (3, 2, arch.layout.L)).copy()
    coding = np.flatnonzero(arch.site_class == CODING)
    S, H = [], []
    for site in rng.choice(coding, 3000, replace=False):
        g = base.copy()
        b = (arch.ancestral[site] + rng.integers(1, 4)) % 4
        g[1, 0, site] = b
        g[2, :, site] = b
        w = np.exp(gm.evaluate(g)[0])
        s = 1 - w[2]
        if s > 1e-6:
            S.append(s)
            H.append((1 - w[1]) / s)
    S, H = np.array(S), np.array(H)
    rho = sps.spearmanr(S, H)[0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(S, H, s=8, alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("efecto del homocigoto  s")
    ax.set_ylabel("dominancia emergente  h")
    ax.set_title(f"Dominancia emergente en el modelo molecular\nSpearman ρ = {rho:.2f}  (h promedio = {H.mean():.2f})")
    _save(fig, "08_dominancia_emergente")
    neutral_frac = 1 - len(S) / 3000
    return {
        "nombre": "Dominancia emergente (Kacser-Burns)",
        "teoria": "Mutaciones deletéreas mayormente recesivas; h decrece con s",
        "observado": f"h medio = {H.mean():.2f}; ρ(s,h) = {rho:.2f}; {neutral_frac:.0%} de mutaciones codificantes neutras (sinónimas)",
        "pasa": H.mean() < 0.5 and rho < -0.3,
    }


# ----------------------------------------------------------------------------- 9
def dnds(rng):
    """Firma de selección purificadora: dN/dS < 1; en control sin restricción, dN/dS ≈ 1.
    Reloj molecular: las sustituciones sinónimas (neutras) se acumulan a tasa ≈ μ."""
    N, T, mu = 100, int(3000 * SCALE), 2e-4
    out = {}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for k, (label, cmean) in enumerate([("con restricción", 0.15), ("control neutro", 0.0)]):
        arch = codon_architecture(np.random.default_rng(42), n_trait_genes=0, n_enzyme_genes=24,
                                  # kappa = 1: Nei-Gojobori asume mutaciones equiprobables; con kappa > 1
                                  # subestima los sitios sinónimos (sesgo conocido también con datos reales)
                                  mutation_rate=mu, kappa=1.0,
                                  constraint_mean=max(cmean, 1e-12), importance_mean=0.5,
                                  codons_per_gene=120)
        if cmean == 0.0:
            arch.gmap.codon_log_activity[:] = 0.0   # control: toda mutación codificante es neutra
        codon_sites = np.concatenate([g.coding_sites.reshape(-1, 3) for g in arch.gmap.genes])
        pop = Population(arch, N, rng)
        tt, dn, ds = [], [], []

        def rec(p):
            r = stats.dn_ds(arch.ancestral, stats.consensus(p.genomes), codon_sites)
            tt.append(p.generation)
            dn.append(r["dN"])
            ds.append(r["dS"])

        pop.run(T, callback=rec, every=max(T // 15, 1))
        out[label] = (tt, dn, ds)
        ax.plot(tt, ds, "-", color=f"C{k}", label=f"dS {label}")
        ax.plot(tt, dn, "--", color=f"C{k}", label=f"dN {label}")
    t = np.array(out["con restricción"][0])
    ax.plot(t, mu * t, "k:", lw=1, label="μ·t (reloj molecular neutro)")
    ax.set_xlabel("generación")
    ax.set_ylabel("divergencia respecto al ancestro")
    ax.set_title(f"Evolución molecular (N = {N}, μ = {mu})")
    ax.legend(fontsize=8)
    _save(fig, "09_dnds")
    c = out["con restricción"]
    n = out["control neutro"]
    w_sel = c[1][-1] / c[2][-1]
    w_neu = n[1][-1] / n[2][-1]
    return {
        "nombre": "dN/dS y reloj molecular",
        "teoria": "ω < 1 con selección purificadora; ω ≈ 1 sin restricción",
        "observado": f"ω = {w_sel:.2f} con restricción; ω = {w_neu:.2f} en control; dS final = {c[2][-1]:.3f} (μ·t = {mu * T:.3f})",
        # La teoría predice ω < 1 bajo selección purificadora (su valor exacto depende de la
        # distribución de efectos elegida), ω = 1 sin restricción, y dS ≈ μ·t (reloj neutro).
        "pasa": w_sel < 0.8 and 0.85 < w_neu < 1.15 and abs(n[2][-1] / (mu * T) - 1) < 0.15,
    }


# ============================================================================ ecología
def _uniform_founders(arch, land, rng, per_cell, p0=0.5):
    """Fundadores con variación neutra preexistente (frecuencia p0 en cada sitio)."""
    cells = np.repeat(np.arange(land.n_cells), per_cell)
    allele = rng.random((len(cells), 2, arch.layout.L)) < p0
    g = np.where(allele, (arch.ancestral + 1) % 4, arch.ancestral).astype(np.uint8)
    return g, cells


# ----------------------------------------------------------------------------- 10
def population_growth(rng):
    """Crecimiento con regulación por densidad: N_{t+1} = λN / (1 + (λ-1)N/K)  (Beverton-Holt)."""
    K, lam, N0, T, R = 500, 2.0, 10, 40, int(40 * SCALE) + 5
    arch = neutral_architecture(rng, 1, 10, 0.5)
    traj = np.zeros((R, T + 1))
    for r in range(R):
        land = Landscape(1, 1, K=K)
        pop = EcoPopulation.founders(arch, land, rng, np.zeros(N0, int), EcoParams(fecundity=lam, dispersal="none"))
        traj[r, 0] = pop.N
        for t in range(T):
            pop.step()
            traj[r, t + 1] = pop.N
    det = [N0]
    for _ in range(T):
        det.append(lam * det[-1] / (1 + (lam - 1) * det[-1] / K))
    det = np.array(det)
    mean = traj.mean(axis=0)
    rel = np.max(np.abs(mean[1:] - det[1:]) / det[1:])
    eq = traj[:, 25:].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in range(min(R, 15)):
        ax.plot(traj[r], color="C0", alpha=0.25, lw=0.8)
    ax.plot(mean, color="C0", lw=2, label="promedio simulación")
    ax.plot(det, "k--", label="Beverton-Holt determinista")
    ax.axhline(K, color="gray", lw=0.8)
    ax.set_xlabel("generación")
    ax.set_ylabel("tamaño poblacional N")
    ax.set_title(f"Crecimiento regulado por densidad (λ = {lam}, K = {K})")
    ax.legend()
    _save(fig, "10_crecimiento")
    return {
        "nombre": "Crecimiento poblacional con regulación por densidad",
        "teoria": f"Beverton-Holt; equilibrio N* = K = {K}",
        "observado": f"desviación máx. del promedio = {rel:.1%}; N en equilibrio = {eq:.0f}",
        "pasa": rel < 0.1 and abs(eq - K) / K < 0.03,
    }


# ----------------------------------------------------------------------------- 11
def island_model(rng):
    """Modelo de islas de Wright: F_ST ≈ 1 / (1 + 4Nm·(d/(d-1))²)."""
    d, K = 20, 50
    burn, meas = 300, int(500 * SCALE) + 100
    arch = neutral_architecture(rng, 20, 20, 1.0)
    rows = []
    for m in (0.005, 0.02, 0.08):
        land = Landscape(d, 1, K=K)
        g, cells = _uniform_founders(arch, land, rng, K)
        pop = EcoPopulation(arch, land, rng, g, cells, EcoParams(dispersal="island", migration_rate=m))
        pop.run(burn)
        vals = []
        pop.run(meas, callback=lambda p: vals.append(stats.fst(p.genomes, p.cells, arch.ancestral)), every=10)
        rows.append((m, np.mean(vals), 1 / (1 + 4 * K * m * (d / (d - 1)) ** 2)))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    mm = np.logspace(-3, -0.9, 100)
    ax.plot(4 * K * mm, 1 / (1 + 4 * K * mm * (d / (d - 1)) ** 2), "k--", label="teoría")
    ax.plot([4 * K * r[0] for r in rows], [r[1] for r in rows], "o", ms=8, label="simulación")
    ax.set_xscale("log")
    ax.set_xlabel("4Nm (migrantes efectivos por generación × 4)")
    ax.set_ylabel("F_ST")
    ax.set_title(f"Modelo de islas de Wright ({d} demes, N = {K})")
    ax.legend()
    _save(fig, "11_islas_fst")
    rel = max(abs(r[1] - r[2]) / r[2] for r in rows)
    return {
        "nombre": "Diferenciación entre poblaciones (islas de Wright)",
        "teoria": "F_ST = 1/(1 + 4Nm(d/(d−1))²)",
        "observado": "; ".join(f"m={r[0]}: {r[1]:.3f} vs {r[2]:.3f}" for r in rows),
        "pasa": rel < 0.2,
    }


# ----------------------------------------------------------------------------- 12
def isolation_by_distance(rng):
    """Aislamiento por distancia en 1D (Rousset 1997): F_ST/(1−F_ST) ≈ a + r / (4Dσ²)."""
    n_demes, K, m = 40, 50, 0.1
    T, meas = int(2500 * SCALE) + 1000, 1000
    arch = neutral_architecture(rng, 20, 20, 1.0)
    land = Landscape(n_demes, 1, K=K)
    g, cells = _uniform_founders(arch, land, rng, K)
    pop = EcoPopulation(arch, land, rng, g, cells, EcoParams(dispersal="stepping_stone", migration_rate=m))
    pop.run(T - meas)
    dists = np.arange(1, 11)
    acc = []

    def rec(p):
        M, labels = stats.pairwise_fst(p.genomes, p.cells, arch.ancestral)
        pos = land.x[labels]
        central = (pos >= 10) & (pos < n_demes - 10)
        out = []
        for r in dists:
            vals = [M[i, j] for i in np.flatnonzero(central) for j in range(len(labels)) if pos[j] - pos[i] == r]
            v = np.mean(vals)
            out.append(v / (1 - v))
        acc.append(out)

    pop.run(meas, callback=rec, every=50)
    y = np.mean(acc, axis=0)
    slope, icpt = np.polyfit(dists, y, 1)
    sigma2 = m          # un paso ±1 con probabilidad m  =>  varianza de dispersión = m
    theory = 1 / (4 * K * sigma2)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(dists, y, "o", label="simulación")
    ax.plot(dists, icpt + slope * dists, "-", color="C0", lw=1, label=f"ajuste: pendiente {slope:.4f}")
    ax.plot(dists, icpt + theory * dists, "k--", label=f"Rousset: pendiente {theory:.4f}")
    ax.set_xlabel("distancia entre demes (celdas)")
    ax.set_ylabel("F_ST / (1 − F_ST)")
    ax.set_title("Aislamiento por distancia (stepping-stone 1D)")
    ax.legend()
    _save(fig, "12_aislamiento_distancia")
    return {
        "nombre": "Aislamiento por distancia (Rousset)",
        "teoria": f"pendiente = 1/(4Dσ²) = {theory:.4f}",
        "observado": f"pendiente = {slope:.4f}",
        "pasa": abs(slope - theory) / theory < 0.3,
    }


EXPERIMENTS = {
    "hw": hardy_weinberg, "drift": drift, "fixation": neutral_fixation, "kimura": kimura,
    "mutation": mutation_drift, "linkage": linkage, "selection": deterministic_selection,
    "dominance": emergent_dominance, "dnds": dnds,
    "growth": population_growth, "islands": island_model, "ibd": isolation_by_distance,
}


def main():
    global SCALE
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", choices=list(EXPERIMENTS))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if args.quick:
        SCALE = 0.3
    rng = np.random.default_rng(args.seed)
    names = args.only or list(EXPERIMENTS)
    results = []
    for name in names:
        t0 = time.time()
        r = EXPERIMENTS[name](rng)
        r["tiempo"] = time.time() - t0
        results.append(r)
        print(f"[{'PASA' if r['pasa'] else 'FALLA'}] {r['nombre']}  ({r['tiempo']:.0f}s)\n"
              f"        teoría:    {r['teoria']}\n        observado: {r['observado']}", flush=True)

    if not args.only:
        lines = ["# Resultados de validación", "",
                 f"Semilla {args.seed}{' (modo rápido)' if args.quick else ''}.", "",
                 "| Experimento | Teoría | Observado | Resultado |", "|---|---|---|---|"]
        for r in results:
            lines.append(f"| {r['nombre']} | {r['teoria']} | {r['observado']} | {'✅' if r['pasa'] else '❌'} |")
        (ROOT / "validation" / "RESULTS.md").write_text("\n".join(lines) + "\n")
    sys.exit(0 if all(r["pasa"] for r in results) else 1)


if __name__ == "__main__":
    main()
