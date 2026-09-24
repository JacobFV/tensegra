# A07: all-edge record attention, initial acquisition screen

Exploratory seed701,1000updates,16,000fresh training graph presentations. This is a graph-record key/value attention baseline, not unconstrained language parsing. It receives all public edge records with no query-dependent neighborhood mask; two soft address reads and aligned source/relation initial coordinates are supplied. Historical A06 remains unchanged.

| Final development condition | Exact task | Mean-head path | Complete suffix |
|---|---:|---:|---:|
| N32/D4/K4 |40.82%|53.52%|3.13%|
| N64/D8/K4 |18.95%|43.95%|0%|
| N128/D32/K8 |10.35%|1.17%|0%|
| Paired IID instruction swap |43.36%|52.34%|3.71%|
| IID zero attribute content |15.23%|.78%|0%|
| IID wrong topology |18.16%|0%|0%|
| Joint shift, doubled record logits and address16 |11.52%|1.37%|0%|

Each final cell has512fresh development events. The supplied graph carries enough information, but this recipe has not learned the all-record selection interface. Unlike A06's finite averaging diagnosis, route maxima themselves are often wrong. Sharpening cannot generally repair selection of the wrong record. No intrinsic impossibility or definitive attention-superiority claim follows from one underacquired comparator.

Learning remains active: IID monitoring accuracy progresses7.81%→16.41%→18.36%→22.66% at0/50/250/500updates; final fresh IID reaches40.82%. N64/D8 grows4.69%→11.72% by500 then18.95% on the final fresh population. Last100training losses average.9118 versus1.2119 at updates701–800, a24.8% reduction. The registered extension trigger (≥10%loss improvement and≥5point task improvement) is met. Note the1000endpoint uses a different fresh population from intermediate monitoring; the accuracy difference is not a paired-event improvement estimate.

A new2000-update recipe may be registered, with explicit replay of the initial1000updates because optimizer states were not archived. This would test greater exposure with unchanged architecture, not silently overwrite the failed1000-update acquisition result. No extension has run at report creation.

Full process50.29seconds, runner48.19seconds; complete compact raw/source/config and checkpoint hashes in `research/results/campaign-01/attention/a07-development/`. Durable checkpoint `~/topoformer-campaign01/attention/a07-development/checkpoint.pt` on GB10. Width1024; allocated4.575M versus2.413Mparticipating parameters. This is not active-parameter matching to A06. Dense all-edge memory cost remains reported in manifests. Independent metric audit requested. No GPU job remains.
