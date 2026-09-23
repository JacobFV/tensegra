# Attention development: explicit routing comparison

Exploratory seed101, not confirmation. The strongest equal-information baselines dominate softλ4 at combined size/depth shift. We retain them.

| Interface | N16/D4 | N32/D4 | N16/D8 | N32/D8 | Heldout22pair |
|---|---:|---:|---:|---:|---:|
| Softλ4 | 1.000 | .988 | .895 | .246 | 1.000 |
| Structured address context | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Message passing | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hard attention | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| No graph | .160 | .168 | .156 | .102 | .188 |

Final500updates,8000generatedgraph presentations,256freshvalidationgraphs/cell. Samewidth1024, initialstate, data and updatecount. All arms allocate4,241,449parameters but not all parameters participate in every arm. Message passing avoids contentQK and is cheaper; allocated parameter matching does not equate activecapacity orFLOPs. Hard/message routing are identical on functional graphs. Their agreement is not independent replication.

Context/message/hard reach1.0 in everycell atfirst10updatecheckpoint. Soft retains1.0 mean-head-argmax fullpointer-path accuracy in allcells, despite .246valueaccuracy atN32/D8. Atthatcell clean-edge massis.626soft versus.982context and1.0hard/message. Correct argmaxidentity doesnot guarantee useful weightedpayload propagation. This supports investigating dilution without attributing failure to identitybinding.

The reverse ordered schedule, explicitnodegrounding, and unique outgoingtypededge are supplied. Hard/message and known-keyprior context receive powerful routingpriors; the shared payloadembedding/update/classifier is learned. This experiment doesnot establish learnedplanning, surfaceunderstanding or advantageouslatentgrounding. Under correctsingle-successor graphhardrouting exactly selectsneighbors; this is not learnedpointerexecution.

[Learningcurves](a01-curves.svg) separate taskvalue, mean-headargmaxpath andedge mass. The pointerdiagnostic isnot a causalproof ofneural computation. Softstrength and normalization can alterweightedmixtures withoutchanging itsargmax.

A02is registered beforeitsoutcomes: strongerfixedλ, publicsizecorrection, andfrozenweight scoreinterventions. It willtest whether mass, ratherthan learnedstateupdate, explains themismatch. No tuningofbaselines orconfirmationselectionyet.

Processoccupancy36.30seconds includes5processinitializations/training/evaluation/export. Perarm soft7.07s/context7.49s/message5.29s/hard7.20s/none9.21s. These wholetrainingprocess times arenot pureinference latency. Originalprofile2.54s ischargedseparately. CPUmechanicaltests8passed afterA02instrumentation; primaryexperimentalwidth1024.
