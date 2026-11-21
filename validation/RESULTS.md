# Resultados de validación

Semilla 2026.

| Experimento | Teoría | Observado | Resultado |
|---|---|---|---|
| Equilibrio de Hardy-Weinberg | Genotipos p², 2pq, q² tras 1 generación | χ² = 196 con 200 g.l. (p = 0.57) | ✅ |
| Pérdida de variación por deriva | Ne = N = 50 | Ne estimado = 50.4 | ✅ |
| Fijación de mutaciones neutras | P = 1/2N = 0.020;  t̄ = 99 gen | P = 0.017 (p = 0.19);  t̄ = 84 gen (z = -2.5) | ✅ |
| Probabilidad de fijación (Kimura) y tamiz de Haldane | Difusión de Kimura con dominancia | s=-0.01,h=0.5: 0.002 vs 0.002; s=+0.00,h=0.5: 0.005 vs 0.005; s=+0.01,h=0.5: 0.011 vs 0.012; s=+0.02,h=0.5: 0.018 vs 0.020; s=+0.05,h=0.5: 0.054 vs 0.049; s=+0.10,h=0.5: 0.088 vs 0.095; s=+0.05,h=0.1: 0.021 vs 0.023; s=+0.05,h=0.9: 0.085 vs 0.081  (p mínimo = 0.232) | ✅ |
| Equilibrio mutación-deriva | H = 3θ/(3+4θ) | θ=0.02: 0.0188 vs 0.0195; θ=0.10: 0.0877 vs 0.0882; θ=0.40: 0.2584 vs 0.2609 | ✅ |
| Decaimiento del desequilibrio de ligamiento | D_t = D₀(1 − r)ᵗ, r de Haldane | r=0.020→0.016; r=0.091→0.086; r=0.165→0.153; r=0.275→0.280; r=0.500→0.496 | ✅ |
| Respuesta a la selección generación a generación | p' = p(p·w₁₁ + q·w₁₂)/w̄;  residuos z ~ N(0,1) | z: media = -0.033, d.e. = 0.990 (1623 pasos) | ✅ |
| Dominancia emergente (Kacser-Burns) | Mutaciones deletéreas mayormente recesivas; h decrece con s | h medio = 0.45; ρ(s,h) = -0.82; 26% de mutaciones codificantes neutras (sinónimas) | ✅ |
| dN/dS y reloj molecular | ω < 1 con selección purificadora; ω ≈ 1 sin restricción | ω = 0.62 con restricción; ω = 1.01 en control; dS final = 0.588 (μ·t = 0.600) | ✅ |
