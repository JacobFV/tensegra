# E04 frozen controller transfer (development)

Both endpoints trained on4800 easy2×2 worlds with an always-tool public teacher. Width1024; lightweight2.206M versus recurrent26.365M parameters. No training or checkpoint selection here. All five policies share each of512 worlds/cell and48-step caps; compute_price0, actual compute separately recorded. References receive richer raw publicJSON than learned27/67 encodings. One initializer per architecture; no replicationclaim.

|Condition|Arm|Success/512|Mean utility|
|---|---|---:|---:|
|easy|lightweight|512|0.9715|
|easy|recurrent|512|0.9715|
|easy|reference-cheap|452|0.8597|
|easy|reference-always_tool|512|0.9715|
|easy|reference-cheap_first|512|0.9754|
|medium|lightweight|512|0.9610|
|medium|recurrent|0|-0.0572|
|medium|reference-cheap|434|0.8122|
|medium|reference-always_tool|512|0.9610|
|medium|reference-cheap_first|512|0.9636|
|hard_feasible|lightweight|512|0.9446|
|hard_feasible|recurrent|0|-0.0640|
|hard_feasible|reference-cheap|445|0.8173|
|hard_feasible|reference-always_tool|512|0.9446|
|hard_feasible|reference-cheap_first|512|0.9483|
|work_tight|lightweight|0|-0.0662|
|work_tight|recurrent|0|-0.0640|
|work_tight|reference-cheap|422|0.7738|
|work_tight|reference-always_tool|0|-0.0441|
|work_tight|reference-cheap_first|131|0.2129|
|expensive_tools|lightweight|512|0.3686|
|expensive_tools|recurrent|0|-0.0640|
|expensive_tools|reference-cheap|436|0.8008|
|expensive_tools|reference-always_tool|512|0.3686|
|expensive_tools|reference-cheap_first|512|0.9136|
|expensive_travel|lightweight|512|0.7354|
|expensive_travel|recurrent|0|-0.0573|
|expensive_travel|reference-cheap|422|0.4838|
|expensive_travel|reference-always_tool|512|0.7354|
|expensive_travel|reference-cheap_first|512|0.7373|
|obstacle|lightweight|309|0.5567|
|obstacle|recurrent|0|-0.0572|
|obstacle|reference-cheap|430|0.8041|
|obstacle|reference-always_tool|512|0.9583|
|obstacle|reference-cheap_first|512|0.9607|

Lightweight reproduces the expensive always-tool strategy on larger standardworlds. It does not adapt spending under tight/highpriceconditions. Recurrent zero size-transfer requires trajectory diagnosis; this is not a universal recurrenceclaim. Obstacle failures separate recovery from baseline execution.

Next exploratory intervention: mixed public-resource curriculum, repaired remainingbudgetmenu, stronger supplied bootstrap, then on-policy verifiedutility. PBT advantage requires matched independent search and three independent processes on fresh confirmation.

Fullprocess occupancy372.107sGPU/399.439CPU-coreseconds including inference/export. All raw trajectories/specs/hashes in research/results/campaign-02/e04-transfer; independentaudit pending.
