# Stage 8 return interface: acquisition status

Timing only, not competence evidence. The default 1024-dimensional factorized persistent model has 55,854,360 parameters. On NVIDIA GB10 / Torch 2.14 CUDA13, two updates with batch2 and zero/one-step evaluations completed in 1.59 seconds including checkpoint serialization, peaking at 1,184,365,056 allocated CUDA bytes. This establishes feasibility; four optimizer examples cannot evaluate acquisition or generalization.

Six remote mechanical tests pass. Compressed concat and facet-token arms have identical parameter counts and paired initial tensors. Mixed full-width additive encoding has a larger encoder; all common backbone tensors still initialize identically. Factorized memory allocates6144 coordinates (six1024-wide tokens) from1024 pre-mixer encoded coordinates; concat allocates1024.

No retention or downstream-use gate has passed. Main studies and composition remain unstarted.

Fixed-set acquisition (seed0) completed for all six arms. Each reconstructed all six fields jointly on32/32 training events at both zero and one recurrent microstep after300 updates. Each arm took6.5–8.3 seconds, peak~1.2GB CUDA allocation. This demonstrates fitting capacity only: joint validation accuracy on128 fresh events was0–3/128 across zero/one/four-step probes. It is not retention competence. All raw predictions/configs/hashes are preserved under `results/stage8/return-acquisition`.

The prespecified main follow-up uses fresh32-example minibatches for1000 updates, three paired initialization seeds, all six arms, train delays0/1/2/4, and512-example validation/test cells with2/8 distractors at delays0/1/2/4/8/16/32. Evaluation is chunked at64. Corruption/lifecycle interventions occur only at16. Retention gate excludes absent unary arg1 from required-operand accuracy; all six-field joint accuracy remains separately reported. No downstream-use experiment is authorized by this runner.
