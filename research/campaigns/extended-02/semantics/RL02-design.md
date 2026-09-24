# RL02: pointer node addressing with a supplied node boundary

Status: prospective development diagnostic. Implemented and CPU-tested only; this worker launched nothing.

**Question.** RL01 localized the omitted 3×4 failure to node addressing. For `field:pattern`, all 509 teacher-forced errors are target-node errors, and free decoding also emits out-of-range node references. Does replacing the ordinal node-ID source/target heads with a content-addressed pointer over already-emitted NODE records repair omitted-composition relations when the correct NODE prefix is supplied?

## Why the hypothesis is well-posed in this codec

An EDGE record `[EDGE, src, dst, role, slot]` is produced in **one decoder step**. All five fields are read from the same decoder state. In the historical actor, `src` and `dst` are classes 0..127 of the linear `source_head`/`target_head` over node **ordinals** (the i-th NODE record). They are not positions in the public text or in the output vocabulary. NODE records always come first; the codec rejects NODE after EDGE, and the S22 prefix controller preserves this order. Node ordinal *i* is therefore exactly the record read as input at decoder position *i*+1. So a pointer is well defined. At an EDGE step, the decoder state (query) scores the decoder states at input positions that hold NODE records (keys). It then scatters those scores into the same 128 ordinal classes. The loss, argmax, controller, codec, strict validity, and scoring code are unchanged. The only difference is where the source/target logits come from.

## Arms (paired, one inherited lineage)

Both arms load the S21 broad endpoint (file/tensor SHA as in RL01), reset AdamW, and use width 1024, batch 8, and 1024 updates. Checkpoints are at 0/256/1024. Both use the **historical eight-field all-record objective**, the RL01 schedule seed 220122, and the model seed 2201. They share one construction stream, and the runner checks it is identical to RL01's `eb3c0564…`.

- `all_records`: the unchanged `TypedRecordActor`. This is a rerun of the RL01 comparator recipe under the RL02 source snapshot, so the pair is clean. It should reproduce RL01 `all_records` up to GPU nondeterminism. Report both.
- `pointer`: `PointerRecordActor`, with `source` and `target` logits **replaced** by pointer logits. One shared key projection `W_k: 1024→256` and a query projection per field `W_q^f: 1024→256` are applied to the `decoder_norm` output. The score is `q·k/√256`. There are **787,200 new parameters**, initialized fresh from `init_seed` 220202 under a forked RNG. The arm has 63,464,515 parameters in total. The historical source/target heads are retained but unused; they receive no gradient. The pointer parameters use a separate AdamW group with learning rate **3e-4**, which was the parent's own peak LR. Historical parameters stay at 3e-5. This group LR is a disclosed deviation: fresh heads would barely move at 3e-5 in 1024 steps.

## Supplied versus learned

- **Supplied structural constraint:** the pointer classes are masked to NODE records present among the *causal previous inputs* at that step. During evaluation, these are the supplied correct prefix. During training, they are the teacher-forced gold records, exactly as in RL01. The mask never reads gold edges, edge counts, or future records, and it excludes non-NODE inputs. An out-of-range node reference is therefore impossible by construction whenever at least one NODE precedes an EDGE. Any change in the strict-validity rate is **not** a learned result. It must be reported separately from exact graphs. RL01 lost 122 of 512 all-record omitted outputs to out-of-range references.
- **Learned:** which admissible node each edge addresses (source and target). Also learned: tag, role, slot, EOS, and the backbone. These were trained as in RL01.
- **Not changed:** emitted indices are still re-embedded through the historical ordinal source/target embeddings. Keys contain the historical sinusoidal record positions. The pointer is therefore content-*capable*, not guaranteed position-free. NODE prefix and evaluation privileges are identical to RL01.
- **Initialization cost:** the fresh pointer makes the `pointer` u0 addressing essentially random, so the u0 exact counts are expected to be about 0. u0 is not a comparison point.

## Data

The RL01 `rl01-v1` TRAIN1024/known512/omitted-3×4-512 are reused unchanged, with the same hashes. Reuse keeps the pair on the same data as RL01 and needs no new generation. It is still **development reuse**: RL01's omitted-set outcome and its error localization motivated this design. The omitted 512 is therefore no longer untouched, and nothing here is a confirmation estimate.

## Pre-registered branch rule (fixed final endpoint u1024)

Promote to a **new, independently registered confirmation** with fresh alpha-disjoint constructions and new lineages only if all three hold:

1. `pointer` omitted-3×4 complete graphs ≥ **52/512**;
2. `pointer` − `all_records` omitted ≥ **+26/512**;
3. `pointer` known512 complete ≥ `all_records` known512 − **26**.

No best-checkpoint selection. Report every checkpoint. Also report, without using them for promotion:

- teacher-forced ordered-tuple accuracy per relation family (especially `field:pattern`/`field:query` target);
- per-field source/target accuracy;
- strict validity, labelled as partly constraint-induced;
- EOS/length diagnostics.

If the rule fails, classify pointer addressing under this parent and budget as not repairing the boundary. Do not automatically extend it; for example, do not add more updates, residual mode, or an unmasked pointer. `residual` mode exists in code only for tests. `verify_config` rejects it.

## Cost estimate

RL01 main took 251.8 s runner wall time. Of that, about 48 s per arm was optimizer time; the rest was evaluation, decoding, checkpointing, and parent load. RL01 profile took 22.2 s. RL02 has the same two-arm structure. The comparator arm matches RL01's per-arm cost. The pointer arm adds about 1.3% parameters and a `[T×S×256]` score per teacher-forced batch. It also adds one 256-dim query/key per cached decode step. Expect about **260–300 s main** and about 25 s profile. Keep the RL01 caps (profile 120 s, main 780 s; ≤900 GPU-s joint). If the profile projects above the cap, stop and register a revision.

## Commands (root scheduling only)

```
PYTHONPATH=src python -m topoformer.campaign02_relation_pointer source-bindings
PYTHONPATH=src python -m pytest -q tests/test_campaign02_relation_pointer.py tests/test_campaign02_relation.py
PYTHONPATH=<snapshot>/src <CUDA-python> -m topoformer.campaign02_relation_pointer run configs/campaign02/rl02-profile.json
PYTHONPATH=<snapshot>/src <CUDA-python> -m topoformer.campaign02_relation_pointer run configs/campaign02/rl02-main.json
```

`--output <dir>` overrides `output_dir`; the override is recorded in the written `config.json`. The configs bind RL01's 13 historical source hashes, which are unchanged, plus `campaign02_relation_pointer.py`. A test checks that the committed configs match the current sources and that the data, parent, seeds, and schedule are identical to `rl01-main.json`. `RL01-localization.py` hard-codes the RL01 arm names. For RL02 it needs the arm tuple changed to `('all_records','pointer')` before it is reused.
