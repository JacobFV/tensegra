# A02 — finite structural mass versus learned updates

Exploratory, registered after A01development and before A02outcomes. Source extends only strength control/evaluation grouping; architecture unchanged. A01context/message/hard hit1.0 all five devcells by10updates; soft4 finalN32D8=.246. No advantage over strongest baseline. We do not weaken those baselines.

Question: does soft4 fail because finite structural mass dilutes under size/depth, or are learned updates themselves unreliable? In the equal-content mathematical limit, a one-successor graph has edge probability exp(λ)/(exp(λ)+N−1). This does not predict learned task accuracy. With permutation graphs and equal content scores, the stochastic operator is αP+(1−α)J, α=(expλ−1)/(expλ+N−1); composition contracts destination-specific signal as α^D. This is a mechanistic fixture, not an independence assumption about learned errors.

Fresh devseed102, train12020000, validation12030000, same500updates/8000generatedgraphs and five256examplecells. Three paired arms: soft4reference, soft8, soft4+log(N/16) size correction. The correction uses public node count, preserves λatN16, and is a programmed score prior, not learned size generalization. Samefinal500endpoint and10/25/50/100/250curves.

Separately evaluate the frozen A01soft4seed101 checkpoint on the same new validation events: unchanged, λoverride8, and λ+log(N/16). These are posthoc development interventions, not independently trained models or a confirmation. Values and all learned weights are unchanged. Compare clean task, mean-head argmaxpath diagnostic, and clean-edge attention mass. Prior strength controls do not carry candidate-correctness probabilities.

Budget:120seconds hardprocess cap, estimated30–60seconds. Selection: use best soft setting by average taskaccuracy across the four IID/size/depth/jointcells atfixed500, tie preferfixedλ simplicity. Confirm strongest soft plus context/message/hard with3pairedfreshseeds regardless of whether softwins. If no softvariantapproaches competitivecontext, retainnegativefinding and diagnose instead of indefinite tuning. Allarms/developmentfailures remain archived. No confirmation data inspected.
