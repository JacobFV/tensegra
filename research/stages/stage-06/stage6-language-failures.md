# Stage 6 TCN: reproduced failure examples

Six final-step errors below reproduce the exact public text and shuffled answer options from the frozen `4c36557` run. Predictions are copied from archived failure buffers; no model inference or training was performed. Semantic, source, compiler, adapter and surface hashes were checked during regeneration.

These examples span both graph-supervised arms, all three lessons, English, Spanish, symbols and controlled renamed entities. Spanish was exposed during multisurface training; symbols remained withheld. The renamed labels are a bounded pool reused across construction scopes.

The archived buffers retain at most the first five errors per cell. Repeated choices such as `erin`, `dave`, `o3` and `novelentity3` across different prompts are consistent with a weak, potentially input-insensitive answer preference. They do not establish a frequency shortcut: correct predictions and later errors are absent, surfaces reuse constructions, and training-label frequencies were not compared. Full logged-buffer counts, explicitly not full prediction histograms, are in the accompanying JSON.

## 1. semantic_supervision · variable_binding · english

Model seed 7, update 256, generator seed 610240; evaluation index 0, option shuffle seed 100007.

```text
The substitution is: D bind alice; A bind carol; B bind erin; E bind bob.
What is the value of D?
```

Visible options in actor order: `erin`, `bob`, `dave`, `frank`, `carol`, `alice`.

Recorded prediction: **erin**. Gold answer: **alice**.

Semantic digest: `5117ede130ed8dae3e9510fc8a4555a4ba2dc583b1d8dc885d94fc967e6788a1`.

## 2. semantic_supervision · unification · spanish

Model seed 7, update 256, generator seed 610241; evaluation index 1, option shuffle seed 100008.

```text
Se conocen estos hechos: es padre de: carol, dave, erin y erin; es padre de: carol, dave, erin y frank; es padre de: carol, alice, bob y frank.
El patrón es es padre de: carol, A, erin y frank.
¿Qué unifica: A?
```

Visible options in actor order: `dave`, `alice`, `carol`, `frank`, `bob`, `erin`.

Recorded prediction: **erin**. Gold answer: **dave**.

Semantic digest: `ef3e142717873945ac02d08454eb1fb5657f5c8592b67377a3203351c170246b`.

## 3. semantic_supervision · set_operations · symbols

Model seed 7, update 256, generator seed 610242; evaluation index 2, option shuffle seed 100009.

```text
{query: (select (and (not (color blue)) (shape sphere))), scene: [(obj o1 green sphere) (obj o0 blue sphere) (obj o3 blue cube) (obj o2 green cube)]}
```

Visible options in actor order: `o1`, `o0`, `o2`, `o3`.

Recorded prediction: **o3**. Gold answer: **o1**.

Semantic digest: `d3a3d2b6a89b8a7d1e01337ec405771ac0c4188bc4a817c3f0affe017f576fe0`.

## 4. multisurface_consistency · variable_binding · spanish

Model seed 7, update 256, generator seed 610240; evaluation index 0, option shuffle seed 100007.

```text
La sustitución es: D está ligado a alice; A está ligado a carol; B está ligado a erin; E está ligado a bob.
¿Qué valor: D?
```

Visible options in actor order: `erin`, `bob`, `dave`, `frank`, `carol`, `alice`.

Recorded prediction: **dave**. Gold answer: **alice**.

Semantic digest: `5117ede130ed8dae3e9510fc8a4555a4ba2dc583b1d8dc885d94fc967e6788a1`.

## 5. multisurface_consistency · unification · symbols

Model seed 7, update 256, generator seed 610244; evaluation index 4, option shuffle seed 100011.

```text
{facts: [(parent frank dave carol bob) (parent frank dave alice carol) (parent dave dave bob carol) (parent dave dave bob carol)], pattern: (parent frank dave B carol), query: (unify B)}
```

Visible options in actor order: `bob`, `erin`, `dave`, `frank`, `alice`, `carol`.

Recorded prediction: **dave**. Gold answer: **alice**.

Semantic digest: `a9b14d14699adbb055e89cc4b0fb25d22723fe672b019c394e38ca372fdc4b9c`.

## 6. multisurface_consistency · set_operations · unseen_lexical

Model seed 7, update 256, generator seed 610242; evaluation index 2, option shuffle seed 100009.

```text
In the scene: novelentity2 is a green sphere; novelentity3 is a blue sphere; novelentity1 is a blue cube; novelentity0 is a green cube.
What select does not colour blue and shape sphere?
```

Visible options in actor order: `novelentity2`, `novelentity3`, `novelentity0`, `novelentity1`.

Recorded prediction: **novelentity3**. Gold answer: **novelentity2**.

Semantic digest: `e51917226fd3fc1212f9360ecbfceed8b5010938c4b6e809bd7bb5785649360c`.

## Reproduction

Run `research/tools/reproduce-stage6-language-failures.py` with the frozen source on `PYTHONPATH`, passing the archived language directory and an output filename prefix. It regenerates only the six selected constructions, joins recorded predictions, verifies hashes and reproduces these JSON/Markdown files. It does not load checkpoints.
