# S13 fixed paired renderer acquisition

The mixed English/Spanish continuation fails all three registered promotion criteria. Both arms score0/512 complete Spanish graphs, raw and calibrated. Mixed training substantially improves Spanish components but reduces English complete graphs:18 calibrated versus135 for English-only (−117/512), exceeding the allowed loss26. Spanish complete graphs remain zero from2048 to4096, so the registered extra-tranche improvement trigger (at least13) also fails. No extension or confirmation is warranted by the fixed rule.

| Endpoint | English raw / calibrated exact | Spanish raw / calibrated exact | Spanish exact-copy graphs | Spanish mean copy | Spanish calibrated ordered F1 |
|---|---:|---:|---:|---:|---:|
| English-only |41 /135|0 /0|0|.0123|.4359|
| Mixed |0 /18|0 /0|67|.8970|.7809|

All graph counts use512 examples. Both begin from identical S11 model/AdamW,119 calibrated English and0 Spanish complete graphs on the fresh DEV population. Mixed Spanish mean copy rises .0141→.7900→.8474→.8970 at the four fixed curves; ordered F1 rises .4307→.5327→.6268→.7809. This is partial component acquisition, not absence of learning, but it does not meet the complete-graph criterion. English calibrated curves are119→98→125→135 for English-only and119→27→29→18 for mixed. Endpoint English-copy accuracy stays high (.9971 vs .9955), so the complete-graph decline cannot be summarized as wholesale English copying failure.

Every prescribed curve, calibration tensor, raw prediction and paired transition is retained. `S13-analyze.py` verifies identical initial states, inherited optimizer steps, graph sequence, negative-pair sequence, first batch and renderer phases; exactly four presentations per construction, mixed two per renderer; and exact Spanish reuse of each checkpoint's English TRAIN128 thresholds. Endpoint English paired transitions are121 correct→wrong,4 wrong→correct,14 both correct,373 both wrong for calibrated outputs. All512 Spanish examples are wrong under both arms. Full component precision/recall/F1, graph-size outcomes and within-example renderer transitions are in `s13-paired-analysis.json`.

The comparison used32768 added presentations per arm. English-only processed1,936,072 tokens and247.7374 optimizer seconds; mixed2,083,804 tokens and256.9394 optimizer seconds. Full occupancies were405.664757941 and416.063828895 seconds, both exit0 within600-second caps. Equal presentations do not mean equal token exposure or compute. This is one development pair on two historical motifs, not independent confirmation, broad language competence, or new-shape transfer. S15's separately frozen shape intervention remains unchanged by these outcomes.
