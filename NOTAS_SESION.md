# Notas de sesión — para continuar en otro chat

Fecha: 2026-11-21. Repo: `git@github.com:marzuqgallego1/genetic_sim.git` (rama `main`).
Idioma de trabajo con el usuario: **español**. El usuario quiere evolución basada en
**conceptos reales** (genética de poblaciones), validada contra teoría, y resultados
reportados con honestidad (qué funciona y qué no).

## Estado general

| Fase | Estado | Commit |
|---|---|---|
| 1. Motor genético (ADN, meiosis, mutación K80, codones, dominancia emergente) | ✅ hecho y subido | `0f44b2c` |
| 2. Mundo ecológico (paisaje, Beverton-Holt, dispersión, nicho, cambio climático) | ✅ hecho y subido | `b93f20d` |
| Visor interactivo (`viewer/`, artifact privado en claude.ai) | ✅ hecho y subido | `532f84d` |
| 3. Especiación | 🟡 **en curso, sin commit** | — |

Validaciones 1–12 pasan; 17 pruebas de fases 1–2 + 7 nuevas de fase 3 = **24 pruebas pasan**
(`python3 -m pytest -q tests`).

## Fase 3: lo que ya está implementado (sin commit)

Archivos nuevos/modificados (ver `git status`):

- `evosim/maps.py`
  - `Gene.recognition_codons`: codones de la hélice de reconocimiento de un TF.
  - `Regulation(tf, target, site)`: un TF se une al sitio de unión (bases del genoma) en el promotor del gen blanco.
  - `RecognitionCode(energy[21,4], mu)`: energía de contacto aminoácido–base (kT); `preferred_base`.
  - `CodonMap(..., regulations, recognition)`:
    - `binding_energy(genomes, reg)` → `E[n, alelo_TF, haplotipo_sitio]`;
    - `_occupancy` = Z/(1+Z), Z = Σ_k 0.5·actividad_k·e^(μ−E_k) (ambos alelos del TF compiten, efecto en trans);
    - `gene_outputs` multiplica la expresión del blanco por θ/θ_ancestral.
    - Modelo: Tulchinsky et al. 2014; Khatri & Goldstein 2015.
  - Los TF no pueden ser blancos (sin cascadas) y cada gen tiene a lo sumo un sitio.
- `evosim/architecture.py` → `codon_architecture(..., n_tf=0, tf_targets=2, recognition_len=8, tf_mu=4.0, contact_energy=(1,3), tf_target_roles=("enzyme","trait"))`.
  - Con `n_tf=0` reproduce **exactamente** los genomas anteriores (verificado), así que las validaciones viejas no cambian.
  - El sitio ancestral = bases preferidas por el TF ancestral (E=0).
  - Los codones de reconocimiento tienen restricción 0: solo afectan especificidad, no estabilidad.
- `evosim/speciation.py`:
  - `log_fitness`, `cross`, `hybrid_classes` (A, B, F1, F2, BC_A, BC_B), `hybrid_fitness`.
  - `isolation_index` (RI = 1 − w_híbrido / w_parental).
  - `fixed_differences(min_freq=0.9)`.
  - `count_dmis(arch, ga, gb, environment=None, tol=0.005)`: definición operacional por introgresión homocigota. j foráneo deletéreo + rescate al introducir también i (epistasis > tol). Se cuenta en ambos fondos.
  - `Radiation`: linajes alopátricos, `split(name)`, `step/run`, `tips`, `true_tree()` en Newick.
- `evosim/phylo.py`:
  - distancias: `p_distance_matrix`, `jc_distance_matrix`, `mean_pairwise_distance` (d_xy);
  - árboles: `Node`, `neighbor_joining`, `parse_newick`, `splits`, `robinson_foulds`, `collapse_unary`.
- `evosim/__init__.py`: exporta `RecognitionCode`, `Regulation`, `phylo`, `speciation`.
- `tests/test_speciation.py`: 7 pruebas (TF silvestre = 1, desajuste y compensación, efecto en trans, `count_dmis` encuentra una DMI construida, F1 > F2, cruza, NJ en árbol aditivo, radiación).
- `validation/run_validation.py`: **experimento 13 `phylo`** (Nei-Li d_A = 2μt + NJ). **PASA**: pendiente 9.59e-5 vs 1.00e-4 (4 %), Robinson-Foulds = 0.
  - Ojo: `--only` no reescribe `validation/RESULTS.md`; hay que correr la validación completa (~12 min) para que la fila 13 aparezca ahí.
- `examples/speciation.py`: demo de radiación de 5 especies.
  - Calentamiento de 20 000 generaciones hasta el equilibrio de carga, cacheado en `examples/output/speciation_burnin_seed5.npy` (ignorado por git).
  - Vicarianzas en 0, T/4, T/2 y 3T/4; análisis de los 10 pares.
  - Produce `examples/output/speciation.png/json`.
- `README.md`: ya documenta la fase 3, la validación 13 y la demo.
  - **Pendiente**: añadir los resultados finales y las limitaciones honestas cuando se decida el régimen.
- `.gitignore`: añadido `examples/output/*.npy`.

## Hallazgos (importantes para decidir cómo seguir)

1. **Escalado y carga genética.** Con μ = 1e-4 y N = 100, la tasa genómica deletérea U ≈ 0.4 por gameto produce **fusión mutacional**: el fitness cae linealmente sin estabilizarse (0.85 → 0.44 en 5 000 gens).
   - Con N = 200, μ = 5e-5 y `codons_per_gene=40`, el fitness se estabiliza en ≈ 0.70 hacia la generación 20 000. Por eso la demo usa calentamiento.
   - Hay que mencionarlo como limitación: las tasas escaladas conservan Nμ, pero no Ns.
2. **Demo (N = 200, μ = 5e-5, 10 000 gens; 5 TFs × 2 blancos enzimáticos, tf_mu = 6):**
   - NJ recupera el árbol verdadero (RF = 0) y d_A ≈ 1.0 ≈ 2μt.
   - **Bola de nieve**: DMIs ∝ K^2.37 (de 10 DMIs con K = 271 a ~190 con K ≈ 940). Orr predice ~K².
   - **Ruptura en F2** en los 10 pares (RI_F2 > RI_F1). RI_F1 ≈ 0 o negativo (heterosis por enmascarar recesivas fijadas).
   - **Pero el aislamiento es débil**: RI_F2 de 0.4 % a 7 %, sin relación clara con el tiempo de divergencia.
3. **Diagnóstico de por qué es débil** (energías TF_A sobre sitio_B):
   - Casi todos los cambios del TF son "sinónimos de reconocimiento": aminoácidos distintos que prefieren la misma base.
   - Con 2 blancos por TF, cambiar la preferencia desajusta ambos sitios (pleiotropía), así que TF y sitio casi no coevolucionan. Las energías híbridas quedan dentro del rango parental.
   - La mayoría de las DMIs contadas son **epistasis débil entre mutaciones ligeramente deletéreas** (concavidad del flujo de Kacser-Burns dentro de un mismo gen), no incompatibilidades TF–sitio.
   - Esto es coherente con Khatri & Goldstein: la pleiotropía y la selección eficiente frenan las DMIs regulatorias.
4. **En curso al cerrar esta sesión**: prueba con `n_tf=8, tf_targets=1` (un blanco por TF, como en el modelo de Khatri & Goldstein) para ver si la coevolución TF–sitio se libera y el RI crece.
   - Calentamiento de 20 000 gens + 10 000 de divergencia, imprimiendo DMIs fuertes (`tol=0.02`), RI_F1 y RI_F2 cada 2 000 gens.
   - Script copiado en `examples/diag_tf_divergence.py`. Guarda el calentamiento en `burn_tf1.npy`, en el directorio desde donde se corre.
   - Uso: `python3 examples/diag_tf_divergence.py > diag.log`.
   - Imprime, por cada TF→blanco, las energías de unión en A, en B y cruzadas (TF_A/sitio_B, TF_B/sitio_A), junto con las diferencias en la hélice y en el sitio.

## Próximos pasos sugeridos

1. Terminar la prueba de un blanco por TF y elegir el régimen de la demo:
   - si el RI crece claramente, usar `tf_targets=1`;
   - si no, mantener el actual y reportar el resultado honesto (aislamiento lento; la bola de nieve existe pero dominada por epistasis débil).
2. Opcional: en `count_dmis`, clasificar las DMIs por mecanismo (TF–sitio vs. intragénica) y graficarlo. Considerar `tol=0.02` para "DMIs fuertes".
3. Opcional: validación 14 (bola de nieve: exponente > 1), solo si resulta robusta entre réplicas.
4. Correr la validación completa (`python3 validation/run_validation.py`) para regenerar `RESULTS.md` con la fila 13.
5. Actualizar el README con resultados y limitaciones; commit "Fase 3: especiación alopátrica" y push. **El usuario aprueba los push.**
6. Ideas para después: especiación con flujo génico (zonas híbridas en el mundo ecológico, refuerzo); visor para la fase 3; fase 4 (sexos separados y selección sexual).

## Comandos útiles

```bash
python3 -m pytest -q tests
python3 validation/run_validation.py --only phylo
python3 examples/speciation.py               # ~15 min con el calentamiento en caché
python3 examples/record_run.py gradient && python3 viewer/build.py   # visor
```

Máquina: 2 núcleos. Una `Population` con la arquitectura de la demo tarda ~20–30 ms por generación con N = 200.

## Resultado de la prueba de un blanco por TF (`n_tf=8, tf_targets=1`; terminó después del commit 52b4b3b)

| gen | DMIs fuertes (tol 0.02) | K | RI_F1 | RI_F2 |
|---|---|---|---|---|
| 2000 | 0 | 303 | −0.012 | 0.008 |
| 4000 | 7 | 585 | 0.003 | 0.036 |
| 6000 | 25 | 789 | 0.006 | 0.049 |
| 8000 | 97 | 963 | 0.014 | 0.046 |
| 10000 | 75 | 1051 | 0.010 | 0.053 |

- **Sin pleiotropía, TF y sitio sí coevolucionan.** Las energías híbridas superan a las parentales.
  - Ejemplo: tf20→g0 tiene E ≈ 2 en ambos padres, pero TF_B sobre el sitio de A da 5.3–5.9.
  - Ejemplo: tf21→g6 tiene E ≤ 1.5 en los padres y 5.3 en el híbrido.
  - Con 2 blancos por TF esto casi no pasaba.
- Las DMIs fuertes crecen más rápido que K, y RI_F2 crece de forma casi monótona (0.008 → 0.053). Aun así, el aislamiento total sigue siendo modesto (~5 %) a 10 000 generaciones.
- Algunos sitios de genes poco importantes (g2, g4: importancia ≈ 0.01–0.03) pierden la unión por completo (E > 9): se vuelven efectivamente no regulados y derivan.
- **Recomendación**: usar `tf_targets=1` en la demo.
  - Para un aislamiento más fuerte: blancos con más importancia o más TFs.
  - Más realista aún: hacer que la mala expresión importe más allá del amortiguamiento de Kacser-Burns, por ejemplo con selección estabilizadora sobre el nivel de expresión.
