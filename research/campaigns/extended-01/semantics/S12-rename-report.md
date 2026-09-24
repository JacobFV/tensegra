# Fixed lexical rename diagnostic

All six frozen endpoints score **0/1024 complete renamed graphs**, raw and calibrated, with their original TRAIN thresholds. The same 1024 semantic constructions and exact copy targets were retained under the audited case-preserving unseen-spelling bijection; no renamed fitting occurred.

The three constant arms remain 0→0. Calibrated decay successes change 66→0, 131→0, and 139→0; every originally correct graph becomes incorrect, with no incorrect→correct transitions. Raw decay successes change 2→0, 6→0, and 11→0. Exact identity-copy accuracy is zero for every complete example under every endpoint; average per-example copy accuracy is .174–.183. Per-endpoint typed/ordered micro F1 and all transition counts are preserved in `s12-renamed-inference/analysis.json`, reproduced by `S12-rename-analyze.py` with artifact hashes, semantic pairing, and threshold equality assertions.

This diagnostic demonstrates fragility to unseen lexical feature codes despite preserved syntax, equality patterns, token count, and visible variable case. It does not test lexical translation or prove that an explicit public equality/schema decoder would fail. It does not alter the primary S12 paired advantage or rescue its failed all-seed competence criterion. Main wrapper exit 0; full occupancy 140.144956836 seconds within cap 420, GPU idle verified before archive.
