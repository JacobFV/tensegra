# C04 draft: three-replicate consumed-interface composition confirmation

Prospective CPU design only. Root requested this after C03 development; no fresh population has been generated, no model trained and no GPU allocation released. Final source/config/checkpoint hashes, namespaces, timing implementation and caps require root/reviewer freeze before any fresh outcomes. Historical C01/C02/C03 evidence stays intact. This document extends the pre-outcome C01-confirmation-cost-plan.md; it does not itself promote any interface.

## Pairing and proposed namespaces

Three independent new head/contextual initializations1601/1602/1603, with inherited component lineages mapped prospectively as below. Inherited components are reused frozen confirmation acquisitions, not new random initializations; distinguish that fact from fresh baseline heads and fresh constructions.

| Replicate | P01 inherited lowerer | R04/R05 backbone lineage | Train / clean calibration / shared test | Sampling / roles / rekey / public controls / swapped return |
|---|---|---|---|---|
|0|302|10|560000001 /560000002 /560000003|560000100 /560000101 /560000102 /560000200 /560000300|
|1|303|11|561000001 /561000002 /561000003|561000100 /561000101 /561000102 /561000200 /561000300|
|2|304|12|562000001 /562000002 /562000003|562000100 /562000101 /562000102 /562000200 /562000300|

All train/cal/test counts16384/2048/4096. Shared test means every arm in a replicate sees the exact same4096 complete public base constructions and their predeclared altered views. Replicates have independent constructions. Root must reserve these unused candidate namespaces; local configs/campaign search found no conflicting allocation. Model seeds alone are not a claim of independent inherited pretraining. Reference events/gold labels stay private and outside every forward/runtime path.

## Arms and fixed training

Retain all five arms, including whichever dominates:

- Frozen workspace hybrid: actual P01 proposal → actual predicted exact primitive/operands/destination → typed return → frozen corresponding R04 reconstruction/accessor → frozen corresponding learned R05 consumer.
- Supplied-copy engineering: SAME actual learned proposal/runtime return → supplied33-class one-hot scalar → corresponding frozen R05 oracle-return consumer. It executes one primitive and bypasses reconstruction explicitly; it is not a reconstruction success. Wrong lowering/refusals persist.
- N1 static: new head initialized with the paired inherited P01 lowerer, original public witness distribution; lowerer fine-tuned as in C01/C03.
- N1 roles: identical initial full N1 state and sample indices to static; independent50% public binary-record role swaps with actual-view supervision, as frozen C03. This preserves the stronger reversal baseline even if it wins and retains static's stronger clean performance.
- N2 rekey: new ContextualBaseline, original architecture/width1024/key32, public identity alpha-renaming independently per presentation as frozen C02. No original failed N2-static confirmation rerun; preserve its development failure and matched intervention evidence.

Every neural arm uses one LR.0003, AdamW weight_decay.01, batch256,4000updates, same summed answer/value/proposal losses. Per arm1,024,000 presentations, three neural arms×three replicates=9,216,000 total. N1 pair shares exact initial weights and base sample stream; N2 shares base events/sample indices but its different architecture is not claimed identical initialization. Code/role RNGs remain separate from sample/model RNG. No new primitives/layers, no increased width, no frozen-checkpoint alterations on the hybrid/copy paths, no hidden resampling or target repair.

Primary comparison endpoint remains fixed4000, matching C02/C03 registration. Secondary per-arm checkpoint uses only each replicate's CLEAN calibration answer count at0/250/500/1000/2000/3000/4000, earliest tie. Save all logits/models needed to audit selection; reverse calibration diagnostic never selects. The calibration-selected procedure is frozen before new data; do not pick the better endpoint on test or relabel the secondary as primary after outcomes. Retain both N1 recipes and N2 regardless of their fresh rankings. No extension/reseed/architecture change within confirmation. Save final optimizer/RNG/sample/view visitation for every arm.

## Semantic overlap audit and scope

Fresh IID construction seeds and opaque identities do not imply novel numerical computations. Preserve the declared IID finite-domain split and the separate role-shift population; do not force an unregistered numeric-combination OOD split. Before reporting, compute private audit signatures of the requested operation plus ordered typed numeric operands (including unary absence), and a second signature additionally including exact public threshold/polarity. Exclude nonce keys/names, destination identities and unrelated instructions. Preserve integer/float type tags and exact representable numeric values; do not conflate numerically equal differently typed operands.

For train/cal/test and clean/reversed views report event counts, distinct signatures, overlap counts and the fraction of test events whose signature occurred in the ACTUALLY visited training views. Distinguish unique-signature overlap from multiplicity-weighted event overlap. N1 role visitation determines which swapped signatures were seen; N2 rekey does not create additional numeric signatures. Also report cross-replicate signature overlap rather than treating fresh keys as independent new arithmetic combinations. Where inherited P01/R04/R05 training manifests allow exact regeneration, disclose corresponding numerical overlap; otherwise mark that lineage overlap unmeasured, not zero. These signatures are private audit metadata and must never enter model/runtime inputs, sample selection or posthoc filtering. No numerical-range extrapolation or unseen-computation claim follows from new seeds.

## Public populations and exact runtime

Clean events exactly recreate the registered retention mixture under correct lowering. Swapped-role views preserve numeric candidates/keys/query but change witness distribution; mark them as a distinct predeclared finite-domain robustness population, not as unchanged R04 history. Their targets come from the actual requested swapped operation and runtime-derived type. Any invalid generated view aborts preparation, rather than filtering by correctness or resampling. Wrong predicted proposals execute actual selected operands and retain destination provenance; refusals count in full denominators and never receive gold reset/clipping.

The same formal public schema and primitive contracts apply: add/sub/mul/neg/compare(<), integer/float operand semantics, finite half-unit result grid[-8,8], unary absent pointer canonicalized only from predicted primitive. Public complete-evidence schedule is supplied; no learned halting or readiness-probability claim.

## Gates, controls and reporting

Workspace primary consumed end-to-end gate: in EACH replicate, clean and reversed populations, distractors2 AND8, every delay0/1/2/4/8/16, at least4015/4096 correct JOINT ordered proposal AND answer. Report answer-only separately for fair neural comparisons. Never average away a failed seed/cell. This proposed expanded reversal gate is prospectively harder than original clean R04 acquisition; failure limits the claim rather than changing the population label. Exact-copy and neural answer gates: at least98% on both clean/reversed fresh views per replicate; also report proposal/value diagnostics and full raw outcomes. No baseline failure by itself demonstrates hybrid necessity.

Evaluate record order, inventory order, unrelated instructions and fresh-name(actor-input no-op) controls after selection on the shared test; reverse roles already has an explicit paired population. For workspace and copy, use causal2048-example subsets at8distractors and delays0/1/16 for BOTH clean/reversed populations: correct/drop/wrong/swapped returns, matched query-only consumer, oracle-lowering and oracle-return controls explicitly named. Wrong/swapped controls transform actual supplied returns, never private original labels. Report original-requested versus actually supplied labels, refusals, changed-answer subset counts and unchanged-answer errors. The copy path consumes the same manipulated actual scalar representation; drop is no supplied return, not a gold copy fallback.

Retain registered causal gates per seed/view/delay: clean−drop accuracy≥.15; normalized gain over query-only relative to exact-copy≥.8 when denominatorpositive; wrong AND swapped changed-answer supports≥256 and supplied-answer accuracy≥.90. A zero/negative normalization denominator is undefined/fail, not silently passed. Preserve all cells and marginal cases; causal support cannot be selected after outcomes.

Report per-seed counts, paired arm-only-correct/both-wrong outcomes, bootstrap event uncertainty and seed dispersion; do not treat repeated views/delays as independent examples. Provide primitive/type/query-gap strata, scalar reliability and correct-lowering conditional decision accuracy. Compare answer-only to answer-only and separately expose the stricter hybrid joint gate. No winner is erased when cost/accuracy tradeoffs differ across clean and role-shift populations.

## Cost and preliminary allocation

Use the frozen inference-cost contract in C01-confirmation-cost-plan.md: actual full paths, transfer/CPU runtime included, synchronized batches1/64 on first256 shared-test rows, delay0/16 workspace endpoints, warmup8 and3timedpasses. Report seconds per attempted/correct example, errors/refusals, executed/resident parameters, peak memory and recurrence. Exact-copy timing includes its own P01/runtime rather than amortizing a cached result. Profile the benchmark and source before allocating it; do not hide inference cost inside training occupancy or claim an efficiency win from a small accuracy gap.

Prepare a per-replicate inherited ledger for P01 selected exposure plus full search; Stage9/Stage11 backbone events/microsteps; R04 fit/capture; R05 learned and oracle consumers. Preserve measured timing scopes and unsuccessful development costs. N1 inherits P01, N2 starts scratch, hybrid/copy have different acquired components. Same current presentations do not equal total compute. N2 rekey adds nuisance-input diversity over the same16384 underlying examples; N1 roles adds two public view slots, not extra independent semantic graphs.

Provisional measured allocation (NOT approved/frozen caps): per replicate workspace+copy/control evaluation expected200–400s, cap600; N1static60–110s cap180; N1roles65–120s cap180; N2rekey250–350s cap600. Nine training phases plus three hybrid evaluation phases sum4680s hard caps if approved, expected1725–2940s before inference benchmark. Separate benchmark estimate/cap follows CPU implementation/profile; initial allowance180s/replicate would add540s, not silently borrowed from a training cap. Root may schedule phases separately and requires GPUFREE/fullprocess receipts before analysis. Current C03 one-replicate pair consumed156.908s; C02 rekey286.884s. Expanded hybrid reverse/causal coverage and independent full-path timing require a new mechanical profile, not an assumption of old117.99s cost.

No fresh source runner/config is frozen by this draft. After root/reviewer agreement, implement only the specified new campaign_composition modules/configs, verify CPU data/label/runtime and pairing invariants, fill exact checkpoint/source hashes, profile resource use, then freeze final caps and launch only on root release.
