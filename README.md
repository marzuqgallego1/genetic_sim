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
| Selección sobre rasgos | Selección estabilizadora gaussiana (Wright-Fisher) o ajuste al ambiente local (ecología) |

## Qué modela (fase 2: mundo ecológico)

| Concepto | Implementación |
|---|---|
| Espacio | Cuadrícula de celdas con capas ambientales (p. ej. temperatura) y capacidad de carga K |
| Nicho | Cada rasgo debe coincidir con una variable ambiental: w ∝ exp(−(z − E)²/2ω²) |
| Regulación por densidad | Beverton-Holt por celda; el tamaño poblacional emerge y puede llegar a 0 (extinción) |
| Estocasticidad demográfica | Número de hijos ~ Poisson |
| Apareamiento local | Padre elegido dentro de la celda ∝ fitness; autofecundación opcional (sin ella hay efecto Allee) |
| Dispersión | Stepping-stone, islas o kernel gaussiano |
| Cambio ambiental | Funciones que modifican el paisaje cada generación (p. ej. calentamiento) |

## Qué modela (fase 3: especiación)

| Concepto | Implementación |
|---|---|
| Regulación en trans | Factores de transcripción (TF) con una hélice de reconocimiento: sus aminoácidos determinan qué base prefiere cada posición del sitio de unión |
| Unión TF–ADN | Termodinámica: energía de contacto aminoácido–base, ocupación θ = Z/(1+Z), Z = Σ x·e^(μ−E); ambos alelos del TF compiten por cada sitio (Tulchinsky et al. 2014; Khatri & Goldstein 2015) |
| Incompatibilidades Dobzhansky-Muller | **Emergen**: cada linaje aislado deriva por su red neutral de pares TF–sitio que funcionan; el híbrido junta un TF de un linaje con sitios del otro |
| Alopatría | `Radiation`: vicarianzas sucesivas; cada linaje evoluciona sin flujo génico y se guarda el árbol verdadero |
| Cruzas de laboratorio | F1, F2 y retrocruzas con meiosis real; aislamiento postcigótico RI = 1 − w_híbrido/w_parental |
| Conteo de DMIs | Introgresión de pares de diferencias fijadas: un alelo foráneo deletéreo que se rescata al introducir también su pareja |
| Filogenia | Distancias Jukes-Cantor, divergencia neta de Nei-Li, Neighbor-Joining, distancia de Robinson-Foulds contra el árbol verdadero |

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
  ecology.py       paisaje, regulación por densidad, apareamiento local, dispersión
  stats.py         estadísticos: H, π, LD, H-W, F_ST (Nei y Hudson), dN/dS
  speciation.py    cruzas híbridas, aislamiento reproductivo, conteo de DMIs, radiaciones alopátricas
  phylo.py         distancias, Neighbor-Joining, Newick, Robinson-Foulds
  recorder.py      graba corridas ecológicas en JSON para el visor
viewer/            visor web (index.html), grabaciones (data/) y empaquetador (build.py)
tests/             pruebas unitarias (pytest)
validation/        experimentos que comparan la simulación con la teoría
examples/          demos de fenómenos emergentes (salida en examples/output/)
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
| 10 | Crecimiento poblacional | Beverton-Holt, equilibrio en K |
| 11 | Modelo de islas | F_ST = 1/(1 + 4Nm(d/(d−1))²) (Wright) |
| 12 | Aislamiento por distancia | F_ST/(1−F_ST) crece con pendiente 1/(4Dσ²) (Rousset 1997) |
| 13 | Divergencia y filogenia | d_A = 2μt entre linajes (Nei & Li 1979); Neighbor-Joining recupera el árbol verdadero |

Resultados en `validation/RESULTS.md` y figuras en `validation/figures/`.

## Demos

```bash
python3 examples/range_expansion.py --gif     # expansión de rango sobre un gradiente térmico
python3 examples/climate_change.py            # calentamiento a distintas velocidades
python3 examples/speciation.py                # radiación alopátrica: aislamiento, DMIs y filogenia
```

## Visor interactivo

Reproduce una corrida grabada como un video: mapa del paisaje (densidad, desajuste,
temperatura), clina, genes que se rompen y series de tiempo sincronizadas.

```bash
python3 examples/record_run.py gradient    # graba viewer/data/gradient.json (~6 min)
python3 examples/record_run.py climate     # graba viewer/data/climate.json (~3 min)
python3 viewer/build.py                    # empaqueta todo en viewer/dist/visor.html
```

`viewer/dist/visor.html` se abre directamente en el navegador, sin servidor.

Fenómenos que emergen sin estar programados:

- **Clina adaptativa** y **límite de rango temporal**: la especie coloniza el frío en ~150
  generaciones, pero hacia el calor se detiene ~400 generaciones en tolerancia ≈ 1.2 y solo
  completa el gradiente hacia la generación ~750. Causa: subir la tolerancia solo era fácil por
  pérdida de función de un gen (techo ≈ +1); más allá se requieren mutaciones raras de ganancia
  en promotores, que se acumulan lentamente.
- **Retraso adaptativo** proporcional a la velocidad del cambio, cercano a la predicción de
  Lynch & Lande: k·(ω² + V_P)/V_G.
- **Límite a la adaptación**: con calentamiento rápido la población se adapta rompiendo genes
  hasta agotar ese camino; la varianza genética colapsa y la especie se extingue.

## Limitaciones actuales (honestas)

- Generaciones discretas, sin edad ni solapamiento.
- Hermafroditas; aún no hay sexos separados ni elección de pareja.
- Solo sustituciones: faltan inserciones/deleciones y duplicaciones de genes.
- Sin redes de regulación génica en trans; la epistasis surge solo de los mapas no lineales
  (flujo enzimático, selección sobre rasgos).
- Los recursos no se modelan explícitamente (la competencia es la capacidad K por celda);
  no hay interacciones entre especies.
- Las tasas de mutación de las demos están escaladas hacia arriba (práctica habitual para que
  la evolución ocurra en cientos y no millones de generaciones).

## Próximas fases

3. ~~Especiación alopátrica~~ (hecho). Pendiente: especiación con flujo génico (zonas híbridas
   en el mundo ecológico, refuerzo).
4. Sexos separados y selección sexual.
5. Indels y duplicación génica; filogenias reconstruidas desde los genomas.
