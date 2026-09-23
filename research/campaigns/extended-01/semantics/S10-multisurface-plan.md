# Next semantic branch: one surface-augmentation intervention

**Planning only. No training, bulk generation, model inference, or reserved-cache inspection.** Begin only after known-renderer S12 confirmation identifies whether the acquired interface is worth extending. Historical Stage11 is unrelated to campaign S11; see the naming glossary.

## Pinned renderer and observability audit

Source: TCN `018c9ce9286fa4961292fe1e024b49fcd7e2dd7f`, adapter `tcn-fixed-construction-v2`. `tcn_data.build_tcn_example` constructs once in fixed English generation context, compiles one semantic graph, then renders that same Term into **english, spanish, symbols**. The underlying vendor registry mentions other languages, but the supported audited adapter deliberately rejects them; do not silently expand it.

| Surface | Existing availability | What changes / stays identifiable |
|---|---|---|
| English | Pinned grammar | Explicit facts/pattern/query blocks and ordered predicate arguments |
| Spanish | Pinned grammar | Translated labels/conjunctions/questions; ordered arguments, facts list, names/variables remain visible |
| Symbols | Pinned s-expression renderer | Explicit record fields and delimiters, ordered lists/arguments; a distinct notation whose correspondence is not automatically known to an English-only model |
| Reordered blocks | **Not an existing adapter renderer** | Requires separately versioned public-text transform; whole labeled blocks can move while preserving list and argument order |

A tiny audit uses eight new planning seeds, each original and injectively renamed, across three renderers: **16 cases, 48 surfaces**. All are visibly copyable, with no public-target or 64-bit lexical-feature collision in this limited sample, no truncation, nodes30–37, maximum ordered slot3. Token ranges: English54–65, Spanish62–75, symbols44–51. Fixed declared non-identity vocabulary (`parent`, `unify`, null) has no unknown targets. Vendor manifest verifies. This is an implementation/feasibility probe, not broad coverage, corpus-diversity proof, or learned transfer. Same Term/graph is shared across views; graph copies target **surface-specific first occurrence positions**, not a fixed English index.

**Important confound:** long alpha-renamings trigger the grammar's layout threshold: inline facts/pattern text becomes bulleted multiline text, changing punctuation/token count. This is not a pure lexical intervention. Any lexical-only control must either constrain names to preserve observed layout or report layout-changing and layout-preserving strata separately. Existing S12 rename rules remain a separate protocol.

The current `campaign_semantics_data.target` hardcodes English identifier forms. A multilingual cache must be separately versioned with an explicit renderer field and language-aware copy-target construction using inherited `semantic_scaling.targets(..., language=...)`. Do not feed renderer/gold graph metadata to the actor. Its forward remains public text only. Decoder first-token identity canonicalization is surface-token based and already language agnostic; verify exact target equivalence per renderer.

Potential unobservable distinctions to test, not assume: collapsed identifier forms; same visible text with different canonical graph; hidden generator metadata leaking into target; list order erased by prose; an uncopyable multiword identity; unknown values/slots; semantic train/test overlap. Current graph targets depend on the visible Term, not hidden answer metadata. Facts are an **ordered list** in this compiler: shuffling them without recomputing the term/target is invalid. Canonical scope/entity bookkeeping is supplied only in the separately labeled S10 secondary decoder.

## Preferred first comparison

Use the existing actor, objective, width1024, node capacity, vocabulary and a learning schedule fixed from S12 before new outcomes. **One intervention:** change training surfaces from English-only to a balanced English/Spanish rendering of the same semantic constructions. No architecture, latent consistency loss, new primitive or output metric change simultaneously.

Keep at least **8,192 distinct alpha-normalized TRAIN constructions**, not eight memorized examples. Split by semantic construction first; every rendering and renaming of a graph stays in its split. Reuse only allowed TRAIN constructions, with new alpha-disjoint development512 and confirmation1024 sets registered later; do not reuse inspected S12 confirmation as development. Generate each graph once, render independently, preserve stable semantic IDs. Audit the complete planned corpus before training.

Match initialization, construction batch order, updates, presentations, loss, and thresholds. Treatment alternates renderer per construction visit (seeded initial phase, content-independent), yielding balanced exposure on repeated visits; baseline always English. This keeps graph exposure matched but halves English-specific exposure in the treatment. Report that tradeoff, actual tokens, wall/GPU occupancy and per-graph/per-renderer visits. Spanish generally costs more tokens; equal presentations are not equal FLOPs. A future equal-English-dose control would be a separate compute comparison, not silently added here.

Start with one bounded development pair, then confirm the useful comparison with three paired seeds and fresh constructions. Freeze exact budgets after profiling; do not reserve or launch GPU from this planning document. Primary acquisition metrics: English and Spanish fresh-construction exact canonical recovery separately, typed/ordered F1, copy accuracy, graph-size stratification, and **both-surfaces-correct on the same graph**. This last metric measures functional cross-surface agreement, not latent vector similarity. Keep raw and one TRAIN-calibrated policy fixed across languages; do not tune renderer-specific test thresholds. Preserve learned primary versus S10 supplied-grammar secondary outcomes and the unchanged frequency control.

## Held-out surfaces and stopping rules

Before calling anything “renderer transfer,” reserve an identifiable held-out condition. A public-text-only field-block permutation transform is the cleanest prospective option: preserve exact block labels, internal fact order and ordered arguments, move intact facts/pattern/query blocks, and recompute copy positions from the resulting visible text. Audit both inline and bullet layouts; simplistic line reversal is unsafe. Version the transform, include inverse/round-trip tests, and register held-out permutations before model evaluation. Do not add this transform until its semantics are independently tested.

Symbols may be a **secondary unseen-notation diagnostic**, not a mandatory paraphrase gate: its untrained delimiters/type markers may lack learned correspondence. Spanish is a trained renderer in the proposed arm, so Spanish performance is multilingual acquisition, not zero-shot held-out-language transfer. These known-family tests still contain only the two inherited ordered tree motifs at depth3; new motifs/depth require a separate generator contract.

If training fails, inspect loss/copy acquisition before scaling. If known-renderer fresh graphs fail, investigate diversity/identity/decoder behavior before requiring unseen notation. If English degrades while Spanish improves, report the surface-exposure tradeoff instead of inventing a broad invariant-learning claim. No composition or supervision withdrawal follows from this branch automatically.

## Planning cost and artifacts

Tiny audit script and both receipts are retained under `s10-surface-planning`. Initial run used per-case finite vocabulary only for copy/layout feasibility; a second run verifies the same events against the fixed declared vocabulary. Neither uses a trained model. Total CPU wall is the sum of the two receipts (~2seconds); zero GPU. S12 analysis code preparation/test fixtures read no reserved targets.
