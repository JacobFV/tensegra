# Track C main resource amendment

The initial frozen main config (commit55c2b43) requested every intervention at every delay. Before inspecting any main evaluation results, timing-only monitoring found 1000 training updates took17.30s and approximately100 of840 evaluation rows took another17s. Estimated nine-run wall time was about24min, above the root's15min feasibility check.

Root approved a resource-only amendment: clean retention still covers0/1/2/4/8/16/32 updates; all nine corruptions/lifecycle controls and frozen persistent-drop comparator are evaluated at16 updates. Sample size512, three independent initializations, three validation and three untouched test data seeds, both2/8 distractor conditions and strict all-validation GateC remain unchanged. No outcomes informed selection, and no budget or model changes were made.

The original process was stopped with SIGTERM; partial output remains `/tmp/stage7-c-main.OEtscN/outputs/c-main-truncated-timing-only`. It is not included in scientific main results. Amended execution uses a fresh directory/output and committed source/config. Main has fresh generated training batches; acquisition results cannot authorize C2. C2 remains blocked for root audit.
