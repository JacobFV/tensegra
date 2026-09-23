# A05 draft: superseded before any model run

**Status:** superseded-before-run after a generator-only coalescence audit. No profile or confirmation outcomes exist. Retained as a registered, rejected plan; see `a05-status.json` and `a05-coalescence.json`.

## Original proposed confirmation

Registered before A05 outcomes. A04 acquired the interface in one development seed; this confirmation retains its1000-update recipe and tests three fresh paired initializations401/402/403. No additional route labels or architecture are introduced. The common input coordinate system, graph grounding, regular-neighborhood contract and reverse schedule are supplied. Query/key code matching and value processing are learned.

## Primary question

Does soft graph bias improve exact payload retrieval relative to exact-address neighbor attention or the keyed neighborhood selector when fresh content matching is necessary? Primary condition: N128,depth32,K8. Primary contrasts: soft minus exact gather and soft minus keyed neighborhood selector. Retain the no-graph informational ablation as context. All models receive the same fresh graphs, instructions, continuous attribute codes and payloads. Exact gather/hard masked attention is one algebraic comparator; keyed context has a supplied hard address neighborhood and a known-key retrieval prior. Neither is misrepresented as an unconstrained transformer reading graph text.

Separate clean conditions cover N32/D4/K4; N128/D4/K4; N32/D32/K4; N128/D32/K8; reserved relation composition at N32/D4/K4; N64/D4/K8; N128/D4/K16; N64/D8/K4. The largest condition is a new joint size/depth/neighborhood shift, not simply depth alone. Every final cell has1024 fresh events; eight base groups are shared across arms/seeds. Initialization replicates do not multiply the number of independent events.

## Prospective interventions

On paired N32/D4/K4 events: change the first requested code and recompute targets; supply that change while retaining original targets; remove content scores; zero soft structural strength; supply wrong topology; mismatch graph identity correspondence. Instruction-swap metrics condition separately on changed terminal node and changed payload answer. Wrong-input metrics report agreement with both original and actually supplied facts. Zero strength is meaningful only in soft mode; other arms' duplicate cells are no-op checks.

Wrong topology permutes source rows independently per relation, preserving each relation's in/out-degree counts and one-neighbor-per-code support. Correspondence mismatch conjugates the graph by a within-attribute-group node permutation while leaving node memories fixed. This preserves degree multisets and the unique local matching contract. Both interventions are scored against clean targets and against the actually supplied graph. Missing/spurious controls are deliberately not reused: they would violate the fixed-K gather contract. Their earlier A03 results remain separate. Corruption can remove information needed for the clean answer; no recovery guarantee is asserted.

Consistent node permutation at N128/D32/K8 is restored before scoring. Report any intermediate argmax discrepancy rather than demanding bitwise equality after floating-point reordering.

A04's selected-node mass (.935 forK4, .870 forK8) motivates an additional **prospectively declared** frozen content-scale16 intervention at N128/D32/K8 and N32/D32/K4 for every arm. This changes normalized content QK sharpness, not graph strengthλ. No weights update and no test-based selection occurs. Primary comparisons retain the learned scale. A success shared by multiple interfaces is not a unique metric-programming advantage.

## Training, selection and data isolation

Four arms per seed: soft initial graph strength8, exact gather (`hard`), keyed neighborhood selector (`context`), and no graph (`none`). All use the unchanged1000-update recipe, batch16, N32/K4 and depths1–4, AdamW learning rate.0003 and weight decay1e-4. The same suffix-value supervision is privileged across arms. Training seeds71,000,000+initialization_seed×100,000+step. No graph-route targets enter the forward or loss.

Intermediate monitoring at0/10/25/50/100 uses separate data seed73,000,000,256 examples in two cells. Final-only confirmation at1000 uses81,000,000+base_group×100,000+offset. No checkpoint or threshold selection uses these final events. Preserve all prescribed seeds, including failures. A failed confirmation may motivate a new exploration with fresh data; it cannot be silently replaced.

## Metrics, budget and interpretation

Report task, all-node value, complete suffix trajectory, mean-head argmax path, selected-node mass, supplied-neighborhood mass, paired instruction effects, corruption scores, learning curves, parameters, actual forward/process cost, GPU allocation and RSS. Diagnostic paths are not a proof of causal multihead execution. Candidate correctness/calibration or autonomous execution are outside scope.

Bootstrap shared event indices jointly across fixed initialization seeds/arms; list per-seed differences. Distinguish event-sampling uncertainty from seed variation. Perfect1024-case agreement is not a rare-error or universal-equivalence guarantee.

A mechanical profile first uses an untrained seed499,12updates and64 largest-condition events. Its outcomes cannot select the recipe. Based on A04's63.01second four-arm development, provisional confirmation cap is300seconds per paired seed (four arms),900seconds total, including inference/export. Freeze or prospectively revise this estimate after the profile, before any confirmation outcome. Root owns GPU releases. No autonomous composition is authorized by this study.
