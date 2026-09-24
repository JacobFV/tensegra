# R11 mechanical profile: exact replay succeeds

The frozen profile completes in4.30seconds of full external process occupancy. It replays the historicalR10mechanical endpoint exactly: weights, normalization, complete training-loss sequence, visited membership, and every archived reference prediction/target/event hash. Both100-update forks have identical sampled-index order hashes and final RNG hashes. This verifies the restart mechanism on the small profile, not main competence.

Child phases: load/preparation1.7774s; exact100-update replay.3115s; two100-update forks.2966s; final evaluation.2052s. Total measured child-internal work2.8885s; outer4.30s includes Python/driver startup, checks, export and shutdown.

The main frozen-cache run has32times as many fitting contexts and36times as many updates, but does not recapture or re-export the feature bank. A conservative estimate is70–120seconds, dominated by verification/decompression/preparation plus fixed replay/fitting. Proposed main hard cap180seconds. Main data, endpoint and thresholds remain exactly the preregisteredR11protocol; profile results do not select a learning rate or checkpoint.

Immutable main source candidate is`a7f99f05`. Compact profile artifacts/receipts are committed under`research/results/campaign-01/returns/r11-profile`; full fitted readouts, AdamW/RNG checkpoints and logits remain at`/home/brandonin/topoformer-campaign-01/return-tails/r11-profile`. Request independent profile audit before a separate root main release. No main experiment has launched.
