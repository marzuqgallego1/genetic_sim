# Notas de sesión — para continuar en otro chat

Última actualización: 2026-11-21 (segunda sesión del día).
Repo: `git@github.com:marzuqgallego1/genetic_sim.git` (rama `main`).

Idioma de trabajo con el usuario: **español**. El usuario quiere:
- evolución basada en **conceptos reales** de genética de poblaciones;
- todo validado contra la teoría;
- resultados reportados con honestidad: qué funciona y qué no.

**El usuario aprueba los push.**

## Estado general

| Fase | Estado | Commit |
|---|---|---|
| 1. Motor genético (ADN, meiosis, mutación K80, codones, dominancia emergente) | ✅ hecho y subido | `0f44b2c` |
| 2. Mundo ecológico (paisaje, Beverton-Holt, dispersión, nicho, cambio climático) | ✅ hecho y subido | `b93f20d` |
| Visor interactivo (`viewer/`, artifact privado en claude.ai) | ✅ hecho y subido | `532f84d` |
| 3. Especiación alopátrica | 🟡 **en curso**: maquinaria lista, falta un régimen con aislamiento intrínseco claro | `52b4b3b` y el commit de esta sesión |
| 4. Sexos separados y selección sexual | ⏳ pendiente | — |
| 5. Indels y duplicación génica | ⏳ pendiente | — |

Pruebas: **27 pasan** (`python3 -m pytest -q tests`); son 17 de las fases 1–2 y 10 de la fase 3.
Validaciones: pasan la 1–13 (la 13, `phylo`, se volvió a correr en esta sesión).

## Fase 3: lo que está implementado

### Regulación en trans (`evosim/maps.py`)

- `Gene.recognition_codons`: codones de la hélice de reconocimiento de un TF.
- `Regulation(tf, target, site)`: el TF se une a un sitio (bases del genoma) en el promotor del gen blanco.
- `RecognitionCode(energy[21,4], mu)`: energía de contacto aminoácido–base, en kT.
- `CodonMap`:
  - `binding_energy(genomes, reg)` → `E[n, alelo_TF, haplotipo_sitio]`;
  - ocupación θ = Z/(1+Z), con Z = Σ_k 0.5·actividad_k·e^(μ−E_k); ambos alelos del TF compiten, efecto en trans;
  - `gene_outputs` multiplica la expresión del blanco por θ/θ_ancestral.
- Modelo: Tulchinsky et al. 2014; Khatri & Goldstein 2015.
- Restricciones: los TF no pueden ser blancos (sin cascadas) y cada gen tiene a lo sumo un sitio.
- **Nuevo en esta sesión: `expression_omega`** (por defecto `None`, apagado). Es selección estabilizadora sobre el nivel de transcripción de los genes regulados:
  - log w −= Σ (log expr)² / (2ω²), con expr = ocupación diploide / ocupación ancestral;
  - castiga solo la mala regulación, no los cambios en los codones;
  - `CodonMap.expression_levels(genomes)` devuelve los niveles, en el orden de `regulations`;
  - con `tf_mu = 6` el costo funciona como un umbral:

| E (kT) | expresión | log w, ω = 0.5 |
|---|---|---|
| 3 | 0.95 | −0.004 |
| 4 | 0.88 | −0.03 |
| 5 | 0.73 | −0.19 |
| 6 | 0.50 | −0.95 |

  - Como el sitio ancestral está casi saturado (θ ≈ 0.998), la sobreexpresión casi no puede ocurrir: en la práctica la selección es contra la subexpresión.

### Arquitectura (`evosim/architecture.py`)

- Firma: `codon_architecture(..., n_tf=0, tf_targets=2, recognition_len=8, tf_mu=4.0, contact_energy=(1,3), tf_target_roles=("enzyme","trait"), expression_omega=None)`.
- Con `n_tf=0` reproduce exactamente los genomas anteriores, así que las validaciones viejas no cambian.
- El sitio ancestral está formado por las bases preferidas del TF ancestral (E = 0).
- Los codones de reconocimiento tienen restricción 0: solo afectan la especificidad, no la estabilidad.

### Especiación (`evosim/speciation.py`)

- `log_fitness`, `cross`, `hybrid_classes` (A, B, F1, F2, BC_A, BC_B), `hybrid_fitness`.
- `isolation_index`: RI = 1 − w_híbrido / w_parental.
- `fixed_differences(min_freq=0.9)`.
- **`count_dmis`, corregido en esta sesión.** Sobre el fondo homocigoto X, el par (i, j) es una DMI si se cumplen las tres condiciones:
  1. Δw(j) < −tol;
  2. hay rescate real: w(X+i+j) − w(X+j) > tol;
  3. hay epistasis: w(X+i+j) − w(X+i) − w(X+j) + w(X) > tol.
- Las diferencias de un mismo codón se introgresan juntas.
- Qué contaba mal la versión vieja:
  - codones quiméricos (a veces de paro) que en un híbrido no aparecen;
  - epistasis positiva sin rescate (concavidad del flujo, o `j` que anula un gen);
  - alelos `i` beneficiosos por sí solos.
- **Nuevo: clasificación por mecanismo.**
  - Funciones: `site_annotation(arch)`, `dmi_mechanism(arch, i, j)`, `dmi_mechanisms(arch, pairs)`.
  - Categorías: `tf_sitio` (hélice × sitio que regula), `regulatoria` (otro par del módulo TF→blanco), `intragenica`, `otra`.
- `Radiation`: linajes alopátricos, `split(name)`, `step/run`, `tips`, `true_tree()` en Newick.

### Filogenia (`evosim/phylo.py`)

- Distancias: `p_distance_matrix`, `jc_distance_matrix`, `mean_pairwise_distance` (d_xy).
- Árboles: `neighbor_joining`, `parse_newick`, `splits`, `robinson_foulds`, `collapse_unary`.

### Validación, demos y pruebas

- **Validación 13 `phylo`** (Nei-Li d_A = 2μt, más NJ): **PASA**, con pendiente 9.59e-5 frente a 1.00e-4 (4 % de error) y RF = 0.
  - `--only` no reescribe `validation/RESULTS.md`; para regenerarlo hay que correr la validación completa (~12 min).
- **`examples/speciation.py`**: demo de radiación de 5 especies.
  - Calentamiento de 20 000 generaciones, cacheado en `examples/output/speciation_burnin_seed5.npy`.
  - Vicarianzas en 0, T/4, T/2 y 3T/4.
  - ⚠ Todavía usa `tf_targets=2` y la arquitectura vieja. Hay que rehacerla.
- **`examples/diag_tf_divergence.py [omega]`**: diagnóstico de dos linajes con `n_tf=8, tf_targets=1`.
  - Hace 20 000 generaciones de calentamiento y 10 000 de divergencia.
  - Cada 2 000 generaciones imprime las DMIs fuertes por mecanismo, RI_F1, RI_F2, la expresión y las energías TF/sitio cruzadas.
  - Guarda `burn_tf1_omega{ω}.npy` y `final_tf1_omega{ω}.npy` en el directorio desde donde se corre.
  - Tarda ~25 min con calentamiento y ~12 min sin él.
- **`tests/test_speciation.py`**, 10 pruebas:
  - TF silvestre;
  - desajuste y compensación;
  - efecto en trans;
  - DMI construida, clasificada como `tf_sitio`;
  - el rescate tiene que ser real;
  - etiquetas de mecanismo;
  - selección sobre la expresión;
  - cruzas;
  - NJ;
  - radiación.

## Hallazgos de la fase 3 (lo que sabemos hasta ahora)

1. **Escalado y carga genética.**
   - Con μ = 1e-4 y N = 100 hay fusión mutacional.
   - Con N = 200, μ = 5e-5 y `codons_per_gene=40`, el fitness se estabiliza en ≈ 0.70–0.73 hacia la generación 20 000. Por eso hace falta el calentamiento.
   - Limitación para el README: las tasas escaladas conservan Nμ, pero no Ns.
2. **La filogenia funciona.** NJ recupera el árbol verdadero (RF = 0) y d_A ≈ 2μt.
3. **⚠ La "bola de nieve ∝ K^2.37" de la primera sesión era casi todo artefacto** de la definición vieja de `count_dmis`. No hay que reportarla; el README no la menciona.
4. **Pleiotropía.** Con 2 blancos por TF, el TF y sus sitios casi no coevolucionan. Con `tf_targets=1` sí lo hacen a veces: en la réplica sin ω, las energías híbridas llegaron a 7–12 en genes poco importantes.
5. **Recuento con la definición corregida** a 10 000 generaciones (`tf_targets=1`, `n_tf=8`, `tf_mu=6`, réplica 2):

| | DMIs (tol 0.005 / 0.02) | RI_F2 intrínseco | RI_F2 con ambiente |
|---|---|---|---|
| sin ω | 0 / 0 | 0.000 | 0.048 |
| ω = 0.5 | 16 / 8 (todas en el módulo tf18→enz5) | 0.013 | 0.027 |

   - **Sin ω, todo el RI_F2 es extrínseco.** Viene de la selección estabilizadora sobre el rasgo y de la varianza de segregación en F2 (modelo geométrico de Fisher), no de DMIs. En los genes poco importantes los sitios se degradan hasta perder la unión (expr ≈ 0), sin costo y sin crear incompatibilidad.
   - **Con ω hay aislamiento intrínseco real, pero pequeño.** Solo 1 de 8 pares TF–sitio coevolucionó. Los demás sitios se quedan en E = 0 y los TF solo acumulan cambios sinónimos de reconocimiento (aminoácidos que prefieren la misma base).
   - **Hay mucha variación entre réplicas**: la réplica 1 con ω = 0.5 dio RI_F2 = 0.107 con la definición vieja.
   - RI_F1 ≈ 0 o negativo en todos los casos: es heterosis, porque el F1 enmascara las recesivas fijadas en cada linaje.
6. **El dilema de fondo** (coherente con Khatri & Goldstein):
   - con selección débil, el TF y el sitio coevolucionan, pero el híbrido casi no paga;
   - con selección fuerte, el híbrido pagaría, pero el sitio no se mueve.
   - Para que haya ruptura híbrida hace falta un **margen casi neutral** donde el sitio pueda derivar, seguido de un **umbral** de costo.

## Qué falta para terminar la fase 3

Objetivo: una demo donde el aislamiento postcigótico **intrínseco** crezca con claridad con el tiempo de divergencia, y reportar con honestidad lo que salga.

1. **Encontrar un régimen con coevolución TF–sitio** (paso principal).
   - Primero probar `tf_mu = 8` con ω = 0.5: el margen casi neutral llegaría hasta E ≈ 4, y una energía híbrida de 6–8 seguiría siendo dañina.
   - Si no basta, probar más divergencia (20 000 generaciones o más), sitios más largos o energías de contacto más bajas.
   - Medir siempre el RI **intrínseco** por separado: `hybrid_fitness(..., environment=None)`.
   - Hacer al menos 3 réplicas por régimen, porque la variación es grande.
2. **Rehacer `examples/speciation.py`** con `tf_targets=1`, `expression_omega` y el régimen elegido.
   - Poner los parámetros en el nombre del archivo de caché del calentamiento.
   - Graficar el RI intrínseco y el extrínseco por separado, y las DMIs por mecanismo contra K.
   - Comprobar el patrón de Coyne & Orr: el RI crece con el tiempo de divergencia.
3. **Bola de nieve (opcional, validación 14).**
   - Rehacer la medición con la definición corregida.
   - Agregarla como validación solo si el exponente es mayor que 1 de forma robusta (≥ 3 semillas). Si no, reportarlo como resultado negativo.
4. **Cerrar la fase.**
   - Correr la validación completa (`python3 validation/run_validation.py`, ~12 min) para regenerar `RESULTS.md` con la fila 13 (y la 14 si existe).
   - En el README, agregar resultados y limitaciones honestas:
     - escalado Nμ sin Ns;
     - el calentamiento necesario;
     - el aislamiento intrínseco lento;
     - sin ω el RI es extrínseco;
     - la sobreexpresión casi no ocurre con un sitio saturado;
     - la corrección de `count_dmis`.
   - Commit "Fase 3: especiación alopátrica" y push.
5. **Visor de especiación (opcional, menos de 1 sesión).** Árbol de linajes en el tiempo, RI y DMIs por par, fitness por clase de cruza.

## Fases siguientes, hasta terminar el proyecto

El plan acordado se considera **completo con las fases 3, 4 y 5**. Con ellas el simulador cubre los grandes temas de la biología evolutiva:

- herencia, mutación, deriva y selección (fase 1);
- ecología y adaptación (fase 2);
- origen de especies (fase 3);
- selección sexual (fase 4);
- origen de genes nuevos (fase 5).

| Fase | Qué aporta | Estado | Esfuerzo estimado |
|---|---|---|---|
| 3. Especiación alopátrica | DMIs, aislamiento postcigótico, filogenias | Maquinaria lista; falta el régimen, la demo y el cierre | 1 sesión |
| 4. Sexos separados y selección sexual | Cromosomas sexuales, elección de pareja, regla de Haldane | Pendiente | 1–2 sesiones largas |
| 5. Indels y duplicación génica | Genes nuevos, sub/neofuncionalización, árboles de genes | Pendiente | 1–2 sesiones largas |
| *3b. Especiación con flujo génico* (opcional) | Zonas híbridas y refuerzo en el mundo ecológico | Pendiente | 1 sesión |
| *Visor de especiación* (opcional) | Árbol de linajes y aislamiento en el visor | Pendiente | < 1 sesión |

### Fase 4: sexos separados y selección sexual

- Cromosomas sexuales (XY y ZW), proporción de sexos y compensación de dosis.
- Preferencias de pareja y caracteres sexuales: modelo de Lande-Kirkpatrick y desbocamiento de Fisher.
- **Regla de Haldane**: el sexo heterogamético sufre más en los híbridos. También el efecto del X grande. Usa la maquinaria de DMIs de la fase 3.
- Validaciones posibles:
  - proporción de sexos 1:1 según Fisher;
  - Ne con sexos desiguales, Ne = 4NmNf/(Nm+Nf);
  - diversidad del X igual a 3/4 de la autosómica.

### Fase 5: indels y duplicación génica

- Inserciones y deleciones (desplazamiento del marco de lectura) y duplicaciones de genes completos.
- Destinos de una copia duplicada: pseudogenización, subfuncionalización (modelo DDC de Force et al. 1999) o neofuncionalización.
- Filogenias de genes frente a filogenias de especies: parálogos y ortólogos.
- Validación posible: la tasa de pérdida de duplicados frente a la teoría de Lynch & Conery.

### 3b. Especiación con flujo génico (opcional)

- Dos linajes divergidos en alopatría vuelven a encontrarse en el paisaje de la fase 2 (contacto secundario).
- ¿Se mantiene la zona híbrida como una clina de tensión (ancho ∝ σ/√s, según Barton & Hewitt) o se fusionan las especies?
- **Refuerzo**: con una preferencia de apareamiento heredable, ¿la selección contra los híbridos aumenta el aislamiento precigótico?
- Requisito: que la fase 3 produzca aislamiento **intrínseco** apreciable; si no, no hay selección contra los híbridos que estudiar.

Después de la fase 5 vendrían extras sin fin definido: varias especies que interactúan (depredador-presa, competencia) o la evolución del desarrollo.

## Comandos útiles

```bash
python3 -m pytest -q tests
python3 validation/run_validation.py --only phylo
python3 examples/diag_tf_divergence.py 0.5 > diag.log   # ~25 min con calentamiento
python3 examples/speciation.py                         # ~15 min con el calentamiento en caché
python3 examples/record_run.py gradient && python3 viewer/build.py   # visor
```

Máquina: 2 núcleos. Una `Population` con la arquitectura de la demo tarda ~20–30 ms por generación con N = 200.
Se pueden correr dos diagnósticos en paralelo.
