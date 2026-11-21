# live_sim — simulador de evolución biológica

Motor de evolución basado en **genética de poblaciones**: la evolución no se programa,
emerge de la herencia mendeliana, la mutación, la recombinación, la deriva y la selección.
El motor se **valida contra resultados teóricos conocidos** antes de construir encima.

## Qué modela (fase 1: motor genético)

| Concepto | Implementación |
|---|---|
| ADN | Secuencias A/C/G/T por sitio, en varios cromosomas |
| Diploidía | Cada individuo tiene dos copias de cada cromosoma |
| Meiosis | Segregación independiente + crossing-over (Poisson, mapa de Haldane) |
| Mutación | Sustituciones puntuales, modelo de Kimura 2P (transiciones/transversiones, `kappa`) |
| Reproducción | Sexual, Wright-Fisher, padres elegidos ∝ fitness |
| Deriva genética | Surge del muestreo finito de padres |
| Genes | Codones traducidos con el código genético estándar |
| Mutaciones sinónimas / no sinónimas / sin sentido | Emergen de la traducción; un codón de paro prematuro produce un alelo nulo |
| Severidad de un cambio de aminoácido | Clases fisicoquímicas (conservativo vs. radical) × restricción de cada posición |
| Dominancia | **Emerge** de la cinética enzimática (Kacser & Burns 1981), no se asigna |
| Regulación cis | Promotores cuyas mutaciones cambian el nivel de expresión |
| Pleiotropía | Un gen de rasgo afecta varios rasgos cuantitativos |
| Plasticidad | Ruido ambiental en el fenotipo (heredabilidad < 1) |
| Selección sobre rasgos | Selección estabilizadora gaussiana (provisional, la reemplazará la ecología) |

## Estructura

```
evosim/
  layout.py        cromosomas, sitios y mapa genético
  meiosis.py       formación de gametos con recombinación
  mutation.py      mutación puntual K80
  genetic_code.py  código genético y propiedades de aminoácidos
  maps.py          genotipo -> fitness/fenotipo (Neutral, SiteEffects clásico, Codon molecular)
  architecture.py  construye genomas (neutros o con genes reales)
  environment.py   selección sobre rasgos
  population.py    ciclo de vida Wright-Fisher
  stats.py         estadísticos: H, π, LD, H-W, dN/dS (Nei-Gojobori + Jukes-Cantor)
tests/             pruebas unitarias (pytest)
validation/        experimentos que comparan la simulación con la teoría
```

## Uso

```bash
python3 -m pytest -q tests                          # pruebas unitarias
python3 validation/run_validation.py --quick        # validación rápida
python3 validation/run_validation.py                # validación completa -> validation/RESULTS.md
```

Ejemplo mínimo:

```python
import numpy as np
from evosim import Population, StabilizingSelection, codon_architecture, stats

rng = np.random.default_rng(1)
arch = codon_architecture(rng, mutation_rate=1e-4)
pop = Population(arch, N=200, rng=rng, environment=StabilizingSelection([0.5, -0.3], omega=1.0))
pop.run(500)
print(stats.nucleotide_diversity(pop.genomes), pop.last_traits.mean(axis=0))
```

## Validación

| # | Experimento | Teoría |
|---|---|---|
| 1 | Hardy-Weinberg | p², 2pq, q² tras una generación |
| 2 | Deriva | H_t = H₀(1 − 1/2N)ᵗ |
| 3 | Fijación neutra | P = 1/2N, t̄ ≈ 4N (Kimura & Ohta) |
| 4 | Fijación con selección | Difusión de Kimura con dominancia; tamiz de Haldane |
| 5 | Mutación-deriva | H = 3θ/(3+4θ), θ = 4Nμ |
| 6 | Ligamiento | D_t = D₀(1 − r)ᵗ |
| 7 | Respuesta a la selección | recursión determinista, residuos ~ N(0,1) |
| 8 | Dominancia emergente | mutaciones de gran efecto más recesivas |
| 9 | Evolución molecular | dN/dS < 1 con selección purificadora; reloj molecular neutro |

Resultados en `validation/RESULTS.md` y figuras en `validation/figures/`.

## Limitaciones actuales (honestas)

- Tamaño poblacional constante y generaciones discretas (sin edad ni solapamiento).
- Hermafroditas; aún no hay sexos separados ni elección de pareja.
- Solo sustituciones: faltan inserciones/deleciones y duplicaciones de genes.
- Sin redes de regulación génica en trans; la epistasis surge solo de los mapas no lineales
  (flujo enzimático, selección gaussiana).
- Sin espacio ni ecología: el fitness por rasgos es una función fija.

## Próximas fases

2. Mundo ecológico: espacio, recursos, densidad, tamaño poblacional variable.
3. Subdivisión y migración (F_ST), especiación alopátrica, incompatibilidades Dobzhansky-Muller.
4. Sexos separados y selección sexual.
5. Indels y duplicación génica; filogenias reconstruidas desde los genomas.
