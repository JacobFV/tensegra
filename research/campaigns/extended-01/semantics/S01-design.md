# S01: diversity and acquisition before changing the actor

Development protocol, baseline `123299a`. Only the coordinator may release GPU work. Initial semantic allocation is three hours within the shared campaign budget; no run acquires that whole allocation automatically.

## Question and contracts

Stage 11 learned eight complete English unification graphs but recovered zero of 512 fresh graphs. Does the unchanged public-text actor acquire transferable canonical semantics after substantially broader procedural exposure? This is not a new test of structural attention: the current actor has no graph-biased attention. Public input is English text and its derived copy inventory; compiler graphs/order/type/edge labels are privileged targets only.

Audit nested TRAIN prefixes of 128, 1,024 and 8,192 unique alpha-normalized constructions, difficulty .5, unchanged pinned unification generator. Reserve 512 disjoint development constructions and 1,024 additional confirmation constructions; no confirmation prediction until a branch is selected and a new confirmation registry is frozen. Generate from seed 12000000, maximum 50000 attempts. Deduplicate full canonical alpha-key strings, not hashes alone. Public input collisions, vocabulary/copy coverage, lexical hash collisions, truncation, node/slot overflow, contradictory pair-slot targets, renderer exposure and graph-size distributions must be recorded. This family is not evidence about saturated variable-binding or set-operation families.

Build a lossless compact CPU cache: public text, stable generator seed, graph hash/alpha-key hash, categorical/copy targets and sparse typed ordered edges. This cache changes storage only. The actor never receives cached graph features. Record cache hashes and generator/compiler hashes. Report actual distinct visited graphs, presentation/token counts and presentation multiplicity per construction. Prefix audits are diversity descriptions, not three completed neural scaling arms.

## Reference model and training

Use the existing `SemanticCurriculumActor`: width 1024, eight workspace rows, four distinct blocks reused twice, 128 graph-node queries, same finite type/relation/slot vocabulary and public lexical features. Preserve sampled balanced edge loss and corrected edge-conditional slot supervision. Gold sampled pair queries select loss rows after node representations are formed. Predicted node presence and edges always determine deployment. No supplied-node features or gold masks enter inference.

Development initialization seed 201; shuffled full passes over 8,192 TRAIN graphs, batch eight, AdamW lr1e-4, clip norm1, BF16 dense operations and FP32 losses. Retain the known curriculum thresholds of 1000/2000 presentations, logging each component. Initial allocation request: 65,536 presentations (8,192 updates), with checkpoints at 0/8,192/32,768/65,536 presentations. This means eight actual epochs over 8,192 graphs, not thousands of repeats of eight surfaces.

At each checkpoint evaluate a fixed 128-graph TRAIN diagnostic/calibration subset and all 512 development graphs. Fit one relation threshold on TRAIN predicted-present pairs only, using the inherited minimum classification-error / lowest-tie rule. Freeze it for development. Raw zero-logit and calibrated metrics remain separate; presence threshold stays zero. Report complete canonical recovery, exact visible copying, entity equivalence, typed/ordered edge precision/recall/F1 and component errors. A coordinatewise frequency baseline is fit to the same full TRAIN mixture (English only), with no text features; its remaining fixed output-position priors are disclosed.

## Adaptive continuation rules

No confirmation selection uses reserved test predictions. After the initial curve, a separately recorded continuation may extend the same optimizer trajectory to 131,072, then at most 262,144 presentations if the last interval improves development complete accuracy by at least .02, or copy/ordered-edge F1 by at least .02. Substantial falling TRAIN loss with poor development can instead justify a diversity/regularization or frontend comparison; it does not count as transfer. A flat weak curve selects a new declared experiment rather than terminating the entire campaign.

The first alternative, if justified, is a conventional contextual-token encoder whose graph queries can read text directly, compared under equivalent public input/output contracts and exposures. That is a meaningful neural baseline and an intervention on the workspace bottleneck, not a new symbolic runtime. It requires its own source/config/profile and coordinator release. Never claim parameter parity implies FLOP parity. Three fresh paired confirmation seeds and at least 512, preferably 1,024, new examples are required for selected claims.

## Profile before launch

CPU audit/cache generation is allowed now, limited to two CPU threads. Profile 32 representative current-actor updates with distinct examples and a 32-example dense evaluation at width 1024, measuring model time, generation/cache time, serialization, CUDA allocated memory, process RSS, parameters and token allocation. Freeze the initial GPU cap after measurement. No main GPU run is currently released. Save every attempted recipe and branch decision, including failure.

## Primary methods read before any new mechanism

[Dozat and Manning, 2017, section 3.1](https://arxiv.org/html/1611.01734v3) motivates separate contextual representations and pair-dependent arc/label scoring. Our historical affine-factor edge scorer is not a faithful replication of that parser, and no dependency-tree constraint is supplied here.

[Dehghani et al., 2019, section 2.1](https://arxiv.org/html/1807.03819v3) refines contextual token states across shared recurrent depth, unlike compressing the entire input into eight workspace rows. This distinction motivates a potential later text-state comparison; it does not establish that the present workspace is intrinsically incapable. No adaptive halting or universality claim is adopted.

## Prospective matched-exposure amendment (before any main outcome)

Coordinator approved three development arms: TRAIN prefixes 128 / 1024 / 8192, all initialized with seed201 and trained for exactly65,536 presentations. All share the same512 development graphs, initialization architecture, optimizer/curriculum, and first128 TRAIN calibration examples. Their effective epochs are512 /64 /8. Prefix-derived categorical vocabularies are identical (parent/unify/null), so the shared output vocabulary introduces no additional class in a smaller arm. Each arm gets its own matched prefix-frequency baseline. Actual presentation/token counts and initialization hashes will verify the comparison. Run arms serially under separate queue releases; freeze profile-derived caps before launch. Extensions select only a useful branch under the previously declared curve rules. This replaces the original one-arm request, not any completed result.

Calibration exports use lossless compressed NumPy arrays for exact float32 scores, boolean targets, pair indices, and graph offsets, with hashes and graph metadata. Threshold computation is unchanged. This avoids JSON expansion of millions of entries when the untrained actor predicts many nodes; it is a storage optimization, not a numerical approximation.

## Measured resource freeze

Profile completed32 updates on256 distinct examples in2.33538 optimizer seconds; full process12.04809 seconds including64 development and64 TRAIN forwards across two checkpoints, exact calibration-array exports and checkpoint serialization. Parameter count57,853,781; peakCUDA allocation1,225,688,064 bytes; processmaxRSS2,909,528 KiB. Extrapolated optimizer time per8192-update arm is598 seconds; allow roughly120 seconds for target-cache/CPU preparation, four full512-development evaluations, calibration and export. Expected12 minutes perarm; requested hard15-minute perarm cap (45-minute total ladder ceiling). Each arm still requires its own coordinator queue release. Budget is based on runtime, not profile accuracy.
