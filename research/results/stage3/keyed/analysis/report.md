# Stage 3 artifact report

9 runs; seeds [0, 1, 2]. Values are seed means ± sample SD. No significance claim is made.

Task accuracy is primary. Exact path completion means all pre-step and post-step grounding argmaxes match their intended nodes, including the destination. Exact attention path completion separately requires every step’s mean-head attention argmax to select the clean next-node token. Exact pre-step grounding instead checks current-node grounding argmaxes and excludes the destination. Both are independent of answer accuracy. Pre-step next-node mass measures grounding before retrieval; structural next-node mass propagates grounding through the supplied relation.

Computation is recurrent and externally scheduled by the supplied relation path. Deeper paths receive more computation. The deterministic exact-routing oracle is an algorithmic control, not a learned model. `known` uses a fixed identity matcher; `graph_input` uses the same fixed cosine soft identity matcher, a privileged prior rather than exact identity routing. `hard` argmax blocks gradients to its identity-initialized grounder. These controls are not parameter-identical learning mechanisms. In `graph_input`, relation-message values already contain successor information, so attending the current entity can retrieve the next value. Its next-token attention and attention-path diagnostics are therefore not mechanism-equivalent to direct structural-attention routing; low values do not establish failed message passing. Identity-initialized soft grounding and randomly initialized soft grounding must be interpreted separately. A high task score alone does not establish graph use or traversal. Identity generalization here concerns unseen random continuous keys and opaque numeric IDs; it is not a lexical-language test.

## Accuracy and grounding

| Condition | Depth | Nodes | Distractors | Composition | Corruption | Variant | Task accuracy | Grounding accuracy | Complete grounding trajectory | Complete attention trajectory | Exact pre-step grounding | Clean-next-token attention |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| heldout_composition | 4 | 16 | 4 | heldout | 0.0 | graph_input_keyed | 0.9948 ± 0.0090 | 0.9844 ± 0.0147 | 0.9271 ± 0.0651 | 0.0000 ± 0.0000 | 0.9453 ± 0.0512 | 0.0702 ± 0.0108 |
| heldout_composition | 4 | 16 | 4 | heldout | 0.0 | none_keyed | 0.1745 ± 0.0296 | 0.2858 ± 0.0081 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0606 ± 0.0031 |
| heldout_composition | 4 | 16 | 4 | heldout | 0.0 | soft_keyed | 0.4609 ± 0.1022 | 0.6204 ± 0.1089 | 0.2135 ± 0.1117 | 0.3568 ± 0.1596 | 0.3333 ± 0.1495 | 0.6362 ± 0.1063 |
| corruption_0.1 | 4 | 16 | 4 | train | 0.1 | graph_input_keyed | 0.7682 ± 0.0239 | 0.8717 ± 0.0235 | 0.6432 ± 0.0554 | 0.0026 ± 0.0045 | 0.7292 ± 0.0502 | 0.0761 ± 0.0112 |
| corruption_0.1 | 4 | 16 | 4 | train | 0.1 | none_keyed | 0.1719 ± 0.0312 | 0.2832 ± 0.0020 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0078 ± 0.0000 | 0.0613 ± 0.0041 |
| corruption_0.1 | 4 | 16 | 4 | train | 0.1 | soft_keyed | 0.4141 ± 0.1213 | 0.5697 ± 0.0764 | 0.1667 ± 0.1020 | 0.2500 ± 0.1257 | 0.2370 ± 0.1211 | 0.5507 ± 0.0583 |
| corruption_0.25 | 4 | 16 | 4 | train | 0.25 | graph_input_keyed | 0.4635 ± 0.0045 | 0.6882 ± 0.0325 | 0.3151 ± 0.0477 | 0.0026 ± 0.0045 | 0.4115 ± 0.0636 | 0.0750 ± 0.0156 |
| corruption_0.25 | 4 | 16 | 4 | train | 0.25 | none_keyed | 0.1719 ± 0.0312 | 0.2832 ± 0.0020 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0078 ± 0.0000 | 0.0613 ± 0.0041 |
| corruption_0.25 | 4 | 16 | 4 | train | 0.25 | soft_keyed | 0.3385 ± 0.0430 | 0.4941 ± 0.0653 | 0.0990 ± 0.0636 | 0.1406 ± 0.0668 | 0.1458 ± 0.0989 | 0.4150 ± 0.0493 |
| corruption_0.5 | 4 | 16 | 4 | train | 0.5 | graph_input_keyed | 0.2396 ± 0.0180 | 0.4746 ± 0.0167 | 0.0729 ± 0.0251 | 0.0026 ± 0.0045 | 0.1172 ± 0.0391 | 0.0676 ± 0.0115 |
| corruption_0.5 | 4 | 16 | 4 | train | 0.5 | none_keyed | 0.1719 ± 0.0312 | 0.2832 ± 0.0020 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0078 ± 0.0000 | 0.0613 ± 0.0041 |
| corruption_0.5 | 4 | 16 | 4 | train | 0.5 | soft_keyed | 0.2370 ± 0.0119 | 0.3984 ± 0.0085 | 0.0156 ± 0.0135 | 0.0312 ± 0.0207 | 0.0443 ± 0.0296 | 0.2304 ± 0.0121 |
| deep_large | 32 | 64 | 4 | train | 0.0 | graph_input_keyed | 0.2682 ± 0.0916 | 0.3579 ± 0.0968 | 0.0208 ± 0.0239 | 0.0000 ± 0.0000 | 0.0260 ± 0.0325 | 0.0207 ± 0.0005 |
| deep_large | 32 | 64 | 4 | train | 0.0 | none_keyed | 0.1432 ± 0.0119 | 0.0432 ± 0.0028 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0141 ± 0.0012 |
| deep_large | 32 | 64 | 4 | train | 0.0 | soft_keyed | 0.1979 ± 0.0163 | 0.0952 ± 0.0268 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1078 ± 0.0272 |
| depth_1 | 1 | 16 | 4 | train | 0.0 | graph_input_keyed | 0.9974 ± 0.0045 | 1.0000 ± 0.0000 | 0.9922 ± 0.0078 | 0.0625 ± 0.0000 | 1.0000 ± 0.0000 | 0.0618 ± 0.0007 |
| depth_1 | 1 | 16 | 4 | train | 0.0 | none_keyed | 0.1875 ± 0.0135 | 1.0000 ± 0.0000 | 0.0599 ± 0.0045 | 0.0625 ± 0.0000 | 1.0000 ± 0.0000 | 0.0616 ± 0.0003 |
| depth_1 | 1 | 16 | 4 | train | 0.0 | soft_keyed | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.6562 ± 0.1335 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9980 ± 0.0003 |
| depth_16 | 16 | 16 | 4 | train | 0.0 | graph_input_keyed | 0.9792 ± 0.0163 | 0.9653 ± 0.0115 | 0.7917 ± 0.1170 | 0.0000 ± 0.0000 | 0.7969 ± 0.1127 | 0.0890 ± 0.0034 |
| depth_16 | 16 | 16 | 4 | train | 0.0 | none_keyed | 0.1641 ± 0.0590 | 0.1066 ± 0.0041 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0535 ± 0.0029 |
| depth_16 | 16 | 16 | 4 | train | 0.0 | soft_keyed | 0.3698 ± 0.0700 | 0.3517 ± 0.1092 | 0.0495 ± 0.0325 | 0.0625 ± 0.0488 | 0.0547 ± 0.0358 | 0.3916 ± 0.1044 |
| depth_2 | 2 | 16 | 4 | train | 0.0 | graph_input_keyed | 0.9974 ± 0.0045 | 0.9961 ± 0.0039 | 0.9714 ± 0.0251 | 0.0104 ± 0.0045 | 0.9922 ± 0.0078 | 0.0685 ± 0.0076 |
| depth_2 | 2 | 16 | 4 | train | 0.0 | none_keyed | 0.1901 ± 0.0163 | 0.5286 ± 0.0023 | 0.0026 ± 0.0045 | 0.0104 ± 0.0045 | 0.0573 ± 0.0045 | 0.0606 ± 0.0004 |
| depth_2 | 2 | 16 | 4 | train | 0.0 | soft_keyed | 0.7188 ± 0.0812 | 0.8255 ± 0.0430 | 0.4688 ± 0.1151 | 0.6693 ± 0.0759 | 0.6510 ± 0.0861 | 0.8330 ± 0.0375 |
| depth_32 | 32 | 16 | 4 | train | 0.0 | graph_input_keyed | 0.9714 ± 0.0197 | 0.9557 ± 0.0339 | 0.7083 ± 0.1683 | 0.0000 ± 0.0000 | 0.7083 ± 0.1683 | 0.0968 ± 0.0050 |
| depth_32 | 32 | 16 | 4 | train | 0.0 | none_keyed | 0.1484 ± 0.0282 | 0.0811 ± 0.0049 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0544 ± 0.0019 |
| depth_32 | 32 | 16 | 4 | train | 0.0 | soft_keyed | 0.3620 ± 0.0705 | 0.2721 ± 0.0954 | 0.0234 ± 0.0282 | 0.0312 ± 0.0413 | 0.0260 ± 0.0325 | 0.3166 ± 0.0931 |
| depth_4 | 4 | 16 | 4 | train | 0.0 | graph_input_keyed | 0.9974 ± 0.0045 | 0.9811 ± 0.0166 | 0.9245 ± 0.0592 | 0.0026 ± 0.0045 | 0.9375 ± 0.0547 | 0.0829 ± 0.0081 |
| depth_4 | 4 | 16 | 4 | train | 0.0 | none_keyed | 0.1719 ± 0.0312 | 0.2832 ± 0.0020 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0078 ± 0.0000 | 0.0613 ± 0.0041 |
| depth_4 | 4 | 16 | 4 | train | 0.0 | soft_keyed | 0.4740 ± 0.1890 | 0.6159 ± 0.1095 | 0.2083 ± 0.1345 | 0.3385 ± 0.1846 | 0.2995 ± 0.1714 | 0.6443 ± 0.1074 |
| depth_8 | 8 | 16 | 4 | train | 0.0 | graph_input_keyed | 0.9948 ± 0.0090 | 0.9775 ± 0.0137 | 0.8802 ± 0.0700 | 0.0000 ± 0.0000 | 0.8932 ± 0.0586 | 0.0891 ± 0.0147 |
| depth_8 | 8 | 16 | 4 | train | 0.0 | none_keyed | 0.2031 ± 0.0341 | 0.1725 ± 0.0071 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0617 ± 0.0093 |
| depth_8 | 8 | 16 | 4 | train | 0.0 | soft_keyed | 0.4010 ± 0.0783 | 0.4606 ± 0.0888 | 0.1068 ± 0.0636 | 0.1615 ± 0.0783 | 0.1328 ± 0.0769 | 0.4952 ± 0.0845 |
| distractors_0 | 4 | 16 | 0 | train | 0.0 | graph_input_keyed | 0.9974 ± 0.0045 | 0.9876 ± 0.0056 | 0.9401 ± 0.0316 | 0.0182 ± 0.0090 | 0.9531 ± 0.0207 | 0.0866 ± 0.0234 |
| distractors_0 | 4 | 16 | 0 | train | 0.0 | none_keyed | 0.1849 ± 0.0705 | 0.2884 ± 0.0081 | 0.0026 ± 0.0045 | 0.0078 ± 0.0078 | 0.0026 ± 0.0045 | 0.0708 ± 0.0046 |
| distractors_0 | 4 | 16 | 0 | train | 0.0 | soft_keyed | 0.5130 ± 0.0942 | 0.6556 ± 0.0846 | 0.2865 ± 0.1246 | 0.4115 ± 0.1152 | 0.3672 ± 0.1182 | 0.6809 ± 0.0741 |
| distractors_16 | 4 | 16 | 16 | train | 0.0 | graph_input_keyed | 0.9922 ± 0.0078 | 0.9824 ± 0.0186 | 0.9167 ± 0.0674 | 0.0026 ± 0.0045 | 0.9375 ± 0.0625 | 0.0710 ± 0.0132 |
| distractors_16 | 4 | 16 | 16 | train | 0.0 | none_keyed | 0.1797 ± 0.0135 | 0.2728 ± 0.0079 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0428 ± 0.0063 |
| distractors_16 | 4 | 16 | 16 | train | 0.0 | soft_keyed | 0.5703 ± 0.0610 | 0.6810 ± 0.0752 | 0.3307 ± 0.1095 | 0.4401 ± 0.0983 | 0.4036 ± 0.1095 | 0.7006 ± 0.0665 |
| size_32 | 4 | 32 | 4 | train | 0.0 | graph_input_keyed | 0.9922 ± 0.0078 | 0.9707 ± 0.0090 | 0.8802 ± 0.0430 | 0.0078 ± 0.0078 | 0.9089 ± 0.0251 | 0.0520 ± 0.0068 |
| size_32 | 4 | 32 | 4 | train | 0.0 | none_keyed | 0.1276 ± 0.0393 | 0.2676 ± 0.0103 | 0.0026 ± 0.0045 | 0.0026 ± 0.0045 | 0.0052 ± 0.0090 | 0.0319 ± 0.0112 |
| size_32 | 4 | 32 | 4 | train | 0.0 | soft_keyed | 0.4323 ± 0.1403 | 0.6048 ± 0.1045 | 0.1875 ± 0.1151 | 0.3177 ± 0.1551 | 0.2969 ± 0.1421 | 0.6153 ± 0.1046 |
| size_64 | 4 | 64 | 4 | train | 0.0 | graph_input_keyed | 0.9219 ± 0.0512 | 0.9121 ± 0.0332 | 0.6432 ± 0.1340 | 0.0000 ± 0.0000 | 0.7474 ± 0.1097 | 0.0180 ± 0.0062 |
| size_64 | 4 | 64 | 4 | train | 0.0 | none_keyed | 0.1094 ± 0.0207 | 0.2565 ± 0.0030 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0131 ± 0.0030 |
| size_64 | 4 | 64 | 4 | train | 0.0 | soft_keyed | 0.3047 ± 0.0827 | 0.5117 ± 0.0959 | 0.1198 ± 0.0861 | 0.1979 ± 0.0939 | 0.1927 ± 0.0939 | 0.5168 ± 0.0979 |

The `_keyed` supplement gives all three variants a shared, graph-free cosine identity content bias (β=8). Its soft structural strength starts at 16 to compete with attraction to the current identity. This is a separate comparison family, not a matched comparison to the primary runs. Paired effects compare `soft_keyed` only with `none_keyed` and `graph_input_keyed`; source/configuration mixing remains prohibited. The graph-input attention diagnostic caveat also applies to `graph_input_keyed`.

## Exact routing controls

Direct graph traversal and exact one-hot induced attention route the supplied graph. Under corruption their clean-answer accuracy may fall. Distinct path nodes quantify revisiting; instruction depth is not the number of unique entities visited.

| Condition | Depth | Nodes | Direct oracle accuracy | Attention oracle accuracy | Attention oracle exact path | Distinct path nodes |
|---|---:|---:|---:|---:|---:|---:|
| heldout_composition | 4 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 4.3516 ± 0.0207 |
| corruption_0.1 | 4 | 16 | 0.7682 ± 0.0239 | 0.7682 ± 0.0239 | 0.6901 ± 0.0369 | 4.2552 ± 0.0163 |
| corruption_0.25 | 4 | 16 | 0.4635 ± 0.0045 | 0.4635 ± 0.0045 | 0.3359 ± 0.0341 | 4.2552 ± 0.0163 |
| corruption_0.5 | 4 | 16 | 0.2396 ± 0.0180 | 0.2396 ± 0.0180 | 0.0755 ± 0.0239 | 4.2552 ± 0.0163 |
| deep_large | 32 | 64 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 23.3073 ± 0.3157 |
| depth_1 | 1 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.9375 ± 0.0000 |
| depth_16 | 16 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 9.0651 ± 0.0651 |
| depth_2 | 2 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 2.8073 ± 0.0163 |
| depth_32 | 32 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 11.5677 ± 0.1641 |
| depth_4 | 4 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 4.2552 ± 0.0163 |
| depth_8 | 8 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 6.5208 ± 0.0813 |
| distractors_0 | 4 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 4.3333 ± 0.0664 |
| distractors_16 | 4 | 16 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 4.3073 ± 0.0502 |
| size_32 | 4 | 32 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 4.5911 ± 0.0430 |
| size_64 | 4 | 64 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 4.8125 ± 0.0358 |

## Paired task-accuracy differences

Treatment minus control; positive favors the soft treatment. Pairs have identical evaluation data and training schedules. Initialization matching is reported separately because grounding ablations can deliberately change parameters.

| Condition | Depth | Nodes | Treatment | Control | Difference | Common initialization verified |
|---|---:|---:|---|---|---:|---|
| heldout_composition | 4 | 16 | soft_keyed | graph_input_keyed | -0.5339 ± 0.0989 | True |
| heldout_composition | 4 | 16 | soft_keyed | none_keyed | 0.2865 ± 0.0916 | True |
| corruption_0.1 | 4 | 16 | soft_keyed | graph_input_keyed | -0.3542 ± 0.1299 | True |
| corruption_0.1 | 4 | 16 | soft_keyed | none_keyed | 0.2422 ± 0.1074 | True |
| corruption_0.25 | 4 | 16 | soft_keyed | graph_input_keyed | -0.1250 ± 0.0413 | True |
| corruption_0.25 | 4 | 16 | soft_keyed | none_keyed | 0.1667 ± 0.0508 | True |
| corruption_0.5 | 4 | 16 | soft_keyed | graph_input_keyed | -0.0026 ± 0.0119 | True |
| corruption_0.5 | 4 | 16 | soft_keyed | none_keyed | 0.0651 ± 0.0197 | True |
| deep_large | 32 | 64 | soft_keyed | graph_input_keyed | -0.0703 ± 0.1016 | True |
| deep_large | 32 | 64 | soft_keyed | none_keyed | 0.0547 ± 0.0282 | True |
| depth_1 | 1 | 16 | soft_keyed | graph_input_keyed | 0.0026 ± 0.0045 | True |
| depth_1 | 1 | 16 | soft_keyed | none_keyed | 0.8125 ± 0.0135 | True |
| depth_16 | 16 | 16 | soft_keyed | graph_input_keyed | -0.6094 ± 0.0827 | True |
| depth_16 | 16 | 16 | soft_keyed | none_keyed | 0.2057 ± 0.0989 | True |
| depth_2 | 2 | 16 | soft_keyed | graph_input_keyed | -0.2786 ± 0.0835 | True |
| depth_2 | 2 | 16 | soft_keyed | none_keyed | 0.5286 ± 0.0871 | True |
| depth_32 | 32 | 16 | soft_keyed | graph_input_keyed | -0.6094 ± 0.0620 | True |
| depth_32 | 32 | 16 | soft_keyed | none_keyed | 0.2135 ± 0.0532 | True |
| depth_4 | 4 | 16 | soft_keyed | graph_input_keyed | -0.5234 ± 0.1877 | True |
| depth_4 | 4 | 16 | soft_keyed | none_keyed | 0.3021 ± 0.1671 | True |
| depth_8 | 8 | 16 | soft_keyed | graph_input_keyed | -0.5938 ± 0.0816 | True |
| depth_8 | 8 | 16 | soft_keyed | none_keyed | 0.1979 ± 0.0444 | True |
| distractors_0 | 4 | 16 | soft_keyed | graph_input_keyed | -0.4844 ± 0.0901 | True |
| distractors_0 | 4 | 16 | soft_keyed | none_keyed | 0.3281 ± 0.1362 | True |
| distractors_16 | 4 | 16 | soft_keyed | graph_input_keyed | -0.4219 ± 0.0547 | True |
| distractors_16 | 4 | 16 | soft_keyed | none_keyed | 0.3906 ± 0.0745 | True |
| size_32 | 4 | 32 | soft_keyed | graph_input_keyed | -0.5599 ± 0.1477 | True |
| size_32 | 4 | 32 | soft_keyed | none_keyed | 0.3047 ± 0.1426 | True |
| size_64 | 4 | 64 | soft_keyed | graph_input_keyed | -0.6172 ± 0.0947 | True |
| size_64 | 4 | 64 | soft_keyed | none_keyed | 0.1953 ± 0.0620 | True |

## Null capacity and binding diagnostics

Null mass and distractor-null accuracy diagnose forced binding; key accuracy diagnoses memory alignment. Entropy uses natural logarithms. These diagnostics do not replace task performance.

| Condition | Depth | Nodes | Variant | Entropy | Null mass | Distractor null accuracy | Key grounding accuracy | Pre-step next-node mass | Structural next-node mass | Supplied-relation attention |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| heldout_composition | 4 | 16 | graph_input_keyed | 0.1368 ± 0.0479 | 0.0395 ± 0.0201 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0689 ± 0.0123 | 0.9532 ± 0.0240 | 0.0702 ± 0.0108 |
| heldout_composition | 4 | 16 | none_keyed | 0.2525 ± 0.0108 | 0.2321 ± 0.0429 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0529 ± 0.0061 | 0.3155 ± 0.0060 | 0.0606 ± 0.0031 |
| heldout_composition | 4 | 16 | soft_keyed | 0.1577 ± 0.0186 | 0.0313 ± 0.0141 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0689 ± 0.0055 | 0.6322 ± 0.1051 | 0.6362 ± 0.1063 |
| corruption_0.1 | 4 | 16 | graph_input_keyed | 0.1389 ± 0.0489 | 0.0398 ± 0.0196 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0752 ± 0.0109 | 0.7839 ± 0.0285 | 0.0762 ± 0.0093 |
| corruption_0.1 | 4 | 16 | none_keyed | 0.2523 ± 0.0098 | 0.2316 ± 0.0431 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0524 ± 0.0051 | 0.2864 ± 0.0063 | 0.0596 ± 0.0073 |
| corruption_0.1 | 4 | 16 | soft_keyed | 0.1617 ± 0.0290 | 0.0329 ± 0.0137 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0803 ± 0.0187 | 0.5436 ± 0.0612 | 0.6014 ± 0.0731 |
| corruption_0.25 | 4 | 16 | graph_input_keyed | 0.1399 ± 0.0474 | 0.0411 ± 0.0201 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0730 ± 0.0150 | 0.5291 ± 0.0350 | 0.0819 ± 0.0153 |
| corruption_0.25 | 4 | 16 | none_keyed | 0.2523 ± 0.0098 | 0.2316 ± 0.0431 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0524 ± 0.0051 | 0.2404 ± 0.0110 | 0.0561 ± 0.0051 |
| corruption_0.25 | 4 | 16 | soft_keyed | 0.1698 ± 0.0207 | 0.0312 ± 0.0153 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0768 ± 0.0223 | 0.4081 ± 0.0545 | 0.5332 ± 0.0567 |
| corruption_0.5 | 4 | 16 | graph_input_keyed | 0.1320 ± 0.0412 | 0.0361 ± 0.0156 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0660 ± 0.0107 | 0.2530 ± 0.0177 | 0.0813 ± 0.0135 |
| corruption_0.5 | 4 | 16 | none_keyed | 0.2523 ± 0.0098 | 0.2316 ± 0.0431 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0524 ± 0.0051 | 0.1641 ± 0.0100 | 0.0531 ± 0.0155 |
| corruption_0.5 | 4 | 16 | soft_keyed | 0.1596 ± 0.0238 | 0.0302 ± 0.0145 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0698 ± 0.0147 | 0.2277 ± 0.0134 | 0.4315 ± 0.0068 |
| deep_large | 32 | 64 | graph_input_keyed | 0.6440 ± 0.0949 | 0.2914 ± 0.0732 | 0.8483 ± 0.0130 | 1.0000 ± 0.0000 | 0.0171 ± 0.0012 | 0.3364 ± 0.0928 | 0.0207 ± 0.0005 |
| deep_large | 32 | 64 | none_keyed | 0.4661 ± 0.0086 | 0.1817 ± 0.0621 | 0.8483 ± 0.0130 | 1.0000 ± 0.0000 | 0.0127 ± 0.0029 | 0.0546 ± 0.0035 | 0.0141 ± 0.0012 |
| deep_large | 32 | 64 | soft_keyed | 0.3816 ± 0.0396 | 0.0184 ± 0.0084 | 0.9310 ± 0.0126 | 1.0000 ± 0.0000 | 0.0204 ± 0.0016 | 0.1069 ± 0.0270 | 0.1078 ± 0.0272 |
| depth_1 | 1 | 16 | graph_input_keyed | 0.0090 ± 0.0010 | 0.0009 ± 0.0000 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0624 ± 0.0000 | 0.9989 ± 0.0002 | 0.0618 ± 0.0007 |
| depth_1 | 1 | 16 | none_keyed | 0.0090 ± 0.0010 | 0.0009 ± 0.0000 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0624 ± 0.0000 | 0.9989 ± 0.0002 | 0.0616 ± 0.0003 |
| depth_1 | 1 | 16 | soft_keyed | 0.0044 ± 0.0021 | 0.0003 ± 0.0002 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0625 ± 0.0000 | 0.9995 ± 0.0003 | 0.9980 ± 0.0003 |
| depth_16 | 16 | 16 | graph_input_keyed | 0.1872 ± 0.0565 | 0.0607 ± 0.0225 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0879 ± 0.0050 | 0.9235 ± 0.0236 | 0.0890 ± 0.0034 |
| depth_16 | 16 | 16 | none_keyed | 0.2496 ± 0.0246 | 0.2197 ± 0.0506 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0490 ± 0.0067 | 0.1483 ± 0.0075 | 0.0535 ± 0.0029 |
| depth_16 | 16 | 16 | soft_keyed | 0.2002 ± 0.0336 | 0.0394 ± 0.0111 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0813 ± 0.0036 | 0.3839 ± 0.1032 | 0.3916 ± 0.1044 |
| depth_2 | 2 | 16 | graph_input_keyed | 0.0833 ± 0.0238 | 0.0192 ± 0.0055 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0670 ± 0.0087 | 0.9777 ± 0.0075 | 0.0685 ± 0.0076 |
| depth_2 | 2 | 16 | none_keyed | 0.1580 ± 0.0208 | 0.1037 ± 0.0161 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0552 ± 0.0022 | 0.5488 ± 0.0070 | 0.0606 ± 0.0004 |
| depth_2 | 2 | 16 | soft_keyed | 0.1042 ± 0.0146 | 0.0171 ± 0.0041 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0590 ± 0.0034 | 0.8309 ± 0.0396 | 0.8330 ± 0.0375 |
| depth_32 | 32 | 16 | graph_input_keyed | 0.1901 ± 0.0623 | 0.0596 ± 0.0248 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0955 ± 0.0057 | 0.9147 ± 0.0438 | 0.0968 ± 0.0050 |
| depth_32 | 32 | 16 | none_keyed | 0.2291 ± 0.0253 | 0.1950 ± 0.0498 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0524 ± 0.0040 | 0.1278 ± 0.0111 | 0.0544 ± 0.0019 |
| depth_32 | 32 | 16 | soft_keyed | 0.2110 ± 0.0357 | 0.0442 ± 0.0148 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0827 ± 0.0153 | 0.3096 ± 0.0899 | 0.3166 ± 0.0931 |
| depth_4 | 4 | 16 | graph_input_keyed | 0.1401 ± 0.0482 | 0.0400 ± 0.0191 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0820 ± 0.0094 | 0.9510 ± 0.0245 | 0.0829 ± 0.0081 |
| depth_4 | 4 | 16 | none_keyed | 0.2523 ± 0.0098 | 0.2316 ± 0.0431 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0524 ± 0.0051 | 0.3173 ± 0.0108 | 0.0613 ± 0.0041 |
| depth_4 | 4 | 16 | soft_keyed | 0.1682 ± 0.0299 | 0.0360 ± 0.0140 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0825 ± 0.0192 | 0.6347 ± 0.1090 | 0.6443 ± 0.1074 |
| depth_8 | 8 | 16 | graph_input_keyed | 0.1699 ± 0.0509 | 0.0511 ± 0.0191 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0879 ± 0.0148 | 0.9379 ± 0.0250 | 0.0891 ± 0.0147 |
| depth_8 | 8 | 16 | none_keyed | 0.2755 ± 0.0165 | 0.2493 ± 0.0474 | 0.9518 ± 0.0063 | 1.0000 ± 0.0000 | 0.0550 ± 0.0072 | 0.2068 ± 0.0071 | 0.0617 ± 0.0093 |
| depth_8 | 8 | 16 | soft_keyed | 0.1862 ± 0.0190 | 0.0423 ± 0.0159 | 0.9766 ± 0.0085 | 1.0000 ± 0.0000 | 0.0780 ± 0.0126 | 0.4889 ± 0.0836 | 0.4952 ± 0.0845 |
| distractors_0 | 4 | 16 | graph_input_keyed | 0.1322 ± 0.0383 | 0.0359 ± 0.0147 | N/A | 1.0000 ± 0.0000 | 0.0844 ± 0.0245 | 0.9572 ± 0.0174 | 0.0866 ± 0.0234 |
| distractors_0 | 4 | 16 | none_keyed | 0.2533 ± 0.0226 | 0.1810 ± 0.0408 | N/A | 1.0000 ± 0.0000 | 0.0591 ± 0.0084 | 0.3284 ± 0.0113 | 0.0708 ± 0.0046 |
| distractors_0 | 4 | 16 | soft_keyed | 0.1659 ± 0.0367 | 0.0267 ± 0.0127 | N/A | 1.0000 ± 0.0000 | 0.0845 ± 0.0256 | 0.6713 ± 0.0775 | 0.6809 ± 0.0741 |
| distractors_16 | 4 | 16 | graph_input_keyed | 0.1454 ± 0.0489 | 0.0432 ± 0.0201 | 0.9562 ± 0.0057 | 1.0000 ± 0.0000 | 0.0731 ± 0.0149 | 0.9500 ± 0.0243 | 0.0710 ± 0.0132 |
| distractors_16 | 4 | 16 | none_keyed | 0.2425 ± 0.0134 | 0.2814 ± 0.0477 | 0.9562 ± 0.0057 | 1.0000 ± 0.0000 | 0.0393 ± 0.0093 | 0.3011 ± 0.0093 | 0.0428 ± 0.0063 |
| distractors_16 | 4 | 16 | soft_keyed | 0.1467 ± 0.0414 | 0.0317 ± 0.0130 | 0.9801 ± 0.0054 | 1.0000 ± 0.0000 | 0.0766 ± 0.0047 | 0.6954 ± 0.0703 | 0.7006 ± 0.0665 |
| size_32 | 4 | 32 | graph_input_keyed | 0.1778 ± 0.0539 | 0.0489 ± 0.0185 | 0.9375 ± 0.0034 | 1.0000 ± 0.0000 | 0.0516 ± 0.0079 | 0.9329 ± 0.0235 | 0.0520 ± 0.0068 |
| size_32 | 4 | 32 | none_keyed | 0.3480 ± 0.0079 | 0.2484 ± 0.0373 | 0.9375 ± 0.0034 | 1.0000 ± 0.0000 | 0.0266 ± 0.0126 | 0.2873 ± 0.0120 | 0.0319 ± 0.0112 |
| size_32 | 4 | 32 | soft_keyed | 0.2073 ± 0.0354 | 0.0196 ± 0.0106 | 0.9720 ± 0.0127 | 1.0000 ± 0.0000 | 0.0390 ± 0.0053 | 0.6054 ± 0.1017 | 0.6153 ± 0.1046 |
| size_64 | 4 | 64 | graph_input_keyed | 0.3144 ± 0.0695 | 0.0925 ± 0.0297 | 0.8483 ± 0.0130 | 1.0000 ± 0.0000 | 0.0179 ± 0.0063 | 0.8556 ± 0.0440 | 0.0180 ± 0.0062 |
| size_64 | 4 | 64 | none_keyed | 0.4728 ± 0.0406 | 0.2672 ± 0.0711 | 0.8483 ± 0.0130 | 1.0000 ± 0.0000 | 0.0096 ± 0.0056 | 0.2647 ± 0.0028 | 0.0131 ± 0.0030 |
| size_64 | 4 | 64 | soft_keyed | 0.2891 ± 0.0517 | 0.0094 ± 0.0016 | 0.9310 ± 0.0126 | 1.0000 ± 0.0000 | 0.0193 ± 0.0051 | 0.5111 ± 0.0939 | 0.5168 ± 0.0979 |

## Example trajectories

Examples are predetermined first evaluation examples from the first seed, not selected for success. Node numbers index the current runtime graph and are not stable training labels.

### soft_keyed: {'composition': 'train', 'condition': 'depth_4', 'corruption': 0.0, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9998 | 0.0019 | 0.9995 |
| 1 | 0 | 0 | 11 | 0.9992 | 0.0070 | 1.0000 |
| 2 | 11 | 11 | 14 | 0.9971 | 0.0203 | 1.0000 |
| 3 | 14 | 16 | 12 | 0.7369 | 0.7786 | 0.0613 |

### soft_keyed: {'composition': 'train', 'condition': 'depth_16', 'corruption': 0.0, 'depth': 16, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9998 | 0.0019 | 0.9995 |
| 1 | 0 | 0 | 11 | 0.9992 | 0.0070 | 1.0000 |
| 2 | 11 | 11 | 14 | 0.9971 | 0.0203 | 1.0000 |
| 3 | 14 | 16 | 12 | 0.7369 | 0.7786 | 0.0613 |
| 4 | 12 | 15 | 10 | 0.7039 | 0.7606 | 0.0001 |
| 5 | 10 | 5 | 5 | 0.9930 | 0.0475 | 0.0002 |
| 6 | 5 | 3 | 3 | 0.9980 | 0.0154 | 0.0000 |
| 7 | 3 | 7 | 8 | 0.9996 | 0.0040 | 0.0000 |
| 8 | 8 | 9 | 11 | 1.0000 | 0.0001 | 0.0000 |
| 9 | 11 | 0 | 14 | 0.9974 | 0.0202 | 0.0000 |
| 10 | 14 | 8 | 12 | 0.7561 | 0.7076 | 0.0000 |
| 11 | 12 | 0 | 14 | 0.9633 | 0.1621 | 0.0000 |
| 12 | 14 | 11 | 7 | 0.9990 | 0.0084 | 0.0000 |
| 13 | 7 | 6 | 9 | 1.0000 | 0.0000 | 0.0000 |
| 14 | 9 | 11 | 0 | 0.9999 | 0.0012 | 0.0000 |
| 15 | 0 | 16 | 15 | 0.7332 | 0.8272 | 0.2093 |

### soft_keyed: {'composition': 'train', 'condition': 'depth_32', 'corruption': 0.0, 'depth': 32, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9998 | 0.0019 | 0.9995 |
| 1 | 0 | 0 | 11 | 0.9992 | 0.0070 | 1.0000 |
| 2 | 11 | 11 | 14 | 0.9971 | 0.0203 | 1.0000 |
| 3 | 14 | 16 | 12 | 0.7369 | 0.7786 | 0.0613 |
| 4 | 12 | 15 | 10 | 0.7039 | 0.7606 | 0.0001 |
| 5 | 10 | 5 | 5 | 0.9930 | 0.0475 | 0.0002 |
| 6 | 5 | 3 | 3 | 0.9980 | 0.0154 | 0.0000 |
| 7 | 3 | 7 | 8 | 0.9996 | 0.0040 | 0.0000 |
| 8 | 8 | 9 | 11 | 1.0000 | 0.0001 | 0.0000 |
| 9 | 11 | 0 | 14 | 0.9974 | 0.0202 | 0.0000 |
| 10 | 14 | 8 | 6 | 0.7561 | 0.7076 | 0.0000 |
| 11 | 6 | 11 | 12 | 1.0000 | 0.0005 | 0.0000 |
| 12 | 12 | 1 | 14 | 0.9839 | 0.0985 | 0.0000 |
| 13 | 14 | 9 | 6 | 0.9998 | 0.0022 | 0.0000 |
| 14 | 6 | 0 | 11 | 0.9895 | 0.0681 | 0.0000 |
| 15 | 11 | 8 | 1 | 0.7081 | 0.8040 | 0.0011 |
| 16 | 1 | 0 | 0 | 0.9601 | 0.1727 | 0.0000 |
| 17 | 0 | 11 | 4 | 0.5511 | 0.8786 | 0.0001 |
| 18 | 4 | 11 | 15 | 1.0000 | 0.0005 | 0.0000 |
| 19 | 15 | 1 | 11 | 0.9844 | 0.0956 | 0.0000 |
| 20 | 11 | 0 | 14 | 0.9763 | 0.1268 | 0.0000 |
| 21 | 14 | 8 | 6 | 0.5580 | 0.8479 | 0.0000 |
| 22 | 6 | 11 | 12 | 1.0000 | 0.0005 | 0.0000 |
| 23 | 12 | 1 | 3 | 0.9841 | 0.0970 | 0.0000 |
| 24 | 3 | 0 | 8 | 0.9766 | 0.1257 | 0.0000 |
| 25 | 8 | 8 | 0 | 0.5576 | 0.8482 | 0.8146 |
| 26 | 0 | 0 | 4 | 0.4711 | 0.9429 | 0.9200 |
| 27 | 4 | 11 | 15 | 0.5177 | 0.7985 | 0.0007 |
| 28 | 15 | 1 | 0 | 0.4795 | 1.1856 | 0.0000 |
| 29 | 0 | 9 | 4 | 0.9995 | 0.0042 | 0.0000 |
| 30 | 4 | 0 | 15 | 0.9908 | 0.0612 | 1.0000 |
| 31 | 15 | 15 | 5 | 0.9985 | 0.0124 | 0.9999 |

### soft_keyed: {'composition': 'train', 'condition': 'size_32', 'corruption': 0.0, 'depth': 4, 'distractors': 4, 'nodes': 32}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 20 | 20 | 0 | 0.9995 | 0.0046 | 0.9964 |
| 1 | 0 | 7 | 28 | 0.9994 | 0.0063 | 0.0000 |
| 2 | 28 | 6 | 17 | 0.9765 | 0.1229 | 0.0000 |
| 3 | 17 | 15 | 3 | 0.9945 | 0.0365 | 0.0000 |

### soft_keyed: {'composition': 'train', 'condition': 'size_64', 'corruption': 0.0, 'depth': 4, 'distractors': 4, 'nodes': 64}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 28 | 28 | 46 | 0.9995 | 0.0042 | 0.9985 |
| 1 | 46 | 46 | 43 | 0.7636 | 0.8102 | 0.9626 |
| 2 | 43 | 43 | 9 | 0.8342 | 0.5587 | 0.9996 |
| 3 | 9 | 15 | 62 | 0.3956 | 1.2099 | 0.2335 |

### soft_keyed: {'composition': 'heldout', 'condition': 'heldout_composition', 'corruption': 0.0, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9998 | 0.0019 | 0.9995 |
| 1 | 0 | 0 | 4 | 0.9992 | 0.0070 | 0.9999 |
| 2 | 4 | 11 | 12 | 0.5272 | 0.9582 | 0.0000 |
| 3 | 12 | 6 | 14 | 1.0000 | 0.0000 | 0.0000 |

### soft_keyed: {'composition': 'train', 'condition': 'deep_large', 'corruption': 0.0, 'depth': 32, 'distractors': 4, 'nodes': 64}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 28 | 28 | 46 | 0.9995 | 0.0042 | 0.9985 |
| 1 | 46 | 46 | 43 | 0.7636 | 0.8102 | 0.9626 |
| 2 | 43 | 43 | 9 | 0.8342 | 0.5587 | 0.9996 |
| 3 | 9 | 15 | 62 | 0.3956 | 1.2099 | 0.2335 |
| 4 | 62 | 20 | 23 | 0.5305 | 1.2448 | 0.0000 |
| 5 | 23 | 13 | 57 | 1.0000 | 0.0005 | 0.0000 |
| 6 | 57 | 46 | 33 | 0.5711 | 1.0575 | 0.0003 |
| 7 | 33 | 23 | 3 | 0.8328 | 0.5364 | 0.0004 |
| 8 | 3 | 57 | 22 | 0.7900 | 0.5608 | 0.0000 |
| 9 | 22 | 58 | 52 | 0.9505 | 0.2043 | 0.0004 |
| 10 | 52 | 36 | 19 | 0.7354 | 0.5859 | 0.0000 |
| 11 | 19 | 1 | 39 | 0.9944 | 0.0434 | 0.0000 |
| 12 | 39 | 9 | 49 | 0.6020 | 1.1580 | 0.0000 |
| 13 | 49 | 19 | 54 | 0.9812 | 0.1090 | 0.0000 |
| 14 | 54 | 28 | 50 | 0.3533 | 1.1001 | 0.0010 |
| 15 | 50 | 15 | 12 | 0.8819 | 0.5210 | 0.0000 |
| 16 | 12 | 40 | 62 | 0.9994 | 0.0059 | 0.0000 |
| 17 | 62 | 34 | 23 | 0.9690 | 0.1508 | 0.0000 |
| 18 | 23 | 30 | 57 | 0.6958 | 0.7218 | 0.0003 |
| 19 | 57 | 58 | 61 | 0.5266 | 1.0114 | 0.0185 |
| 20 | 61 | 19 | 33 | 0.9150 | 0.3509 | 0.0001 |
| 21 | 33 | 39 | 0 | 1.0000 | 0.0000 | 0.0000 |
| 22 | 0 | 30 | 9 | 0.9906 | 0.0720 | 0.0000 |
| 23 | 9 | 43 | 62 | 0.9470 | 0.2090 | 0.0000 |
| 24 | 62 | 9 | 47 | 0.5439 | 0.9578 | 0.0000 |
| 25 | 47 | 19 | 27 | 0.9513 | 0.2342 | 0.0000 |
| 26 | 27 | 28 | 50 | 0.3558 | 1.0992 | 0.0004 |
| 27 | 50 | 17 | 12 | 0.9921 | 0.0582 | 0.0000 |
| 28 | 12 | 15 | 7 | 0.9983 | 0.0127 | 0.0000 |
| 29 | 7 | 40 | 12 | 0.9997 | 0.0030 | 0.0000 |
| 30 | 12 | 34 | 62 | 0.9801 | 0.1057 | 0.0000 |
| 31 | 62 | 30 | 47 | 0.7604 | 0.6554 | 0.0000 |

### soft_keyed: {'composition': 'train', 'condition': 'corruption_0.1', 'corruption': 0.1, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9998 | 0.0019 | 0.9995 |
| 1 | 0 | 0 | 11 | 0.9992 | 0.0070 | 1.0000 |
| 2 | 11 | 11 | 14 | 0.9971 | 0.0203 | 1.0000 |
| 3 | 14 | 16 | 12 | 0.7369 | 0.7786 | 0.0613 |

### soft_keyed: {'composition': 'train', 'condition': 'corruption_0.25', 'corruption': 0.25, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9998 | 0.0019 | 0.0000 |
| 1 | 0 | 4 | 11 | 0.7598 | 0.7796 | 0.9899 |
| 2 | 11 | 11 | 14 | 0.9949 | 0.0345 | 1.0000 |
| 3 | 14 | 16 | 12 | 0.5004 | 0.9940 | 0.0024 |

### soft_keyed: {'composition': 'train', 'condition': 'corruption_0.5', 'corruption': 0.5, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9998 | 0.0019 | 0.0000 |
| 1 | 0 | 4 | 11 | 0.7598 | 0.7796 | 0.9899 |
| 2 | 11 | 11 | 14 | 0.9949 | 0.0345 | 1.0000 |
| 3 | 14 | 16 | 12 | 0.5004 | 0.9940 | 0.0024 |

### soft_keyed: {'composition': 'train', 'condition': 'distractors_0', 'corruption': 0.0, 'depth': 4, 'distractors': 0, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 4 | 4 | 15 | 0.9998 | 0.0016 | 1.0000 |
| 1 | 15 | 15 | 5 | 0.9992 | 0.0073 | 0.9998 |
| 2 | 5 | 5 | 6 | 0.7600 | 0.5837 | 0.9999 |
| 3 | 6 | 6 | 12 | 1.0000 | 0.0000 | 0.9991 |

### soft_keyed: {'composition': 'train', 'condition': 'distractors_16', 'corruption': 0.0, 'depth': 4, 'distractors': 16, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 15 | 15 | 0 | 0.9998 | 0.0023 | 0.9999 |
| 1 | 0 | 0 | 4 | 0.9995 | 0.0046 | 0.9999 |
| 2 | 4 | 11 | 15 | 0.6134 | 0.9510 | 0.0003 |
| 3 | 15 | 1 | 0 | 0.9930 | 0.0476 | 0.0000 |

### graph_input_keyed: {'composition': 'train', 'condition': 'depth_4', 'corruption': 0.0, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9991 | 0.0073 | 0.0002 |
| 1 | 0 | 0 | 11 | 0.9843 | 0.0936 | 0.0540 |
| 2 | 11 | 11 | 14 | 0.9450 | 0.2310 | 0.0044 |
| 3 | 14 | 14 | 12 | 0.9752 | 0.1205 | 0.0092 |

### graph_input_keyed: {'composition': 'train', 'condition': 'depth_16', 'corruption': 0.0, 'depth': 16, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9991 | 0.0073 | 0.0002 |
| 1 | 0 | 0 | 11 | 0.9843 | 0.0936 | 0.0540 |
| 2 | 11 | 11 | 14 | 0.9450 | 0.2310 | 0.0044 |
| 3 | 14 | 14 | 12 | 0.9752 | 0.1205 | 0.0092 |
| 4 | 12 | 12 | 10 | 0.9951 | 0.0323 | 0.0037 |
| 5 | 10 | 10 | 5 | 0.8543 | 0.4748 | 0.0001 |
| 6 | 5 | 5 | 3 | 0.9794 | 0.1008 | 0.0001 |
| 7 | 3 | 3 | 8 | 0.9078 | 0.3091 | 0.0002 |
| 8 | 8 | 8 | 11 | 0.9870 | 0.0695 | 0.0010 |
| 9 | 11 | 11 | 14 | 0.9703 | 0.1474 | 0.0026 |
| 10 | 14 | 14 | 12 | 0.9528 | 0.2006 | 0.0152 |
| 11 | 12 | 12 | 14 | 0.9951 | 0.0323 | 0.0072 |
| 12 | 14 | 14 | 7 | 0.9486 | 0.2288 | 0.0000 |
| 13 | 7 | 7 | 9 | 0.9533 | 0.2024 | 0.0083 |
| 14 | 9 | 9 | 0 | 0.9875 | 0.0716 | 0.0023 |
| 15 | 0 | 0 | 15 | 0.9888 | 0.0706 | 0.0042 |

### graph_input_keyed: {'composition': 'train', 'condition': 'depth_32', 'corruption': 0.0, 'depth': 32, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9991 | 0.0073 | 0.0002 |
| 1 | 0 | 0 | 11 | 0.9843 | 0.0936 | 0.0540 |
| 2 | 11 | 11 | 14 | 0.9450 | 0.2310 | 0.0044 |
| 3 | 14 | 14 | 12 | 0.9752 | 0.1205 | 0.0092 |
| 4 | 12 | 12 | 10 | 0.9951 | 0.0323 | 0.0037 |
| 5 | 10 | 10 | 5 | 0.8543 | 0.4748 | 0.0001 |
| 6 | 5 | 5 | 3 | 0.9794 | 0.1008 | 0.0001 |
| 7 | 3 | 3 | 8 | 0.9078 | 0.3091 | 0.0002 |
| 8 | 8 | 8 | 11 | 0.9870 | 0.0695 | 0.0010 |
| 9 | 11 | 11 | 14 | 0.9703 | 0.1474 | 0.0026 |
| 10 | 14 | 14 | 6 | 0.9528 | 0.2006 | 0.0001 |
| 11 | 6 | 6 | 12 | 0.9729 | 0.1248 | 0.0000 |
| 12 | 12 | 12 | 14 | 0.9916 | 0.0509 | 0.0021 |
| 13 | 14 | 14 | 6 | 0.9249 | 0.3116 | 0.0000 |
| 14 | 6 | 6 | 11 | 0.9623 | 0.1616 | 0.0022 |
| 15 | 11 | 11 | 1 | 0.9584 | 0.1918 | 0.0003 |
| 16 | 1 | 1 | 0 | 0.9379 | 0.2339 | 0.0000 |
| 17 | 0 | 0 | 4 | 0.9539 | 0.2315 | 0.0005 |
| 18 | 4 | 16 | 15 | 0.7075 | 0.8290 | 0.0298 |
| 19 | 15 | 15 | 11 | 0.8389 | 0.5206 | 0.0602 |
| 20 | 11 | 11 | 14 | 0.8814 | 0.4064 | 0.0115 |
| 21 | 14 | 14 | 6 | 0.9729 | 0.1293 | 0.0001 |
| 22 | 6 | 6 | 12 | 0.9775 | 0.1078 | 0.0000 |
| 23 | 12 | 12 | 3 | 0.9918 | 0.0499 | 0.0013 |
| 24 | 3 | 3 | 8 | 0.9687 | 0.1398 | 0.0003 |
| 25 | 8 | 8 | 0 | 0.9906 | 0.0534 | 0.0001 |
| 26 | 0 | 0 | 4 | 0.9846 | 0.0899 | 0.0008 |
| 27 | 4 | 16 | 15 | 0.5203 | 1.0099 | 0.0182 |
| 28 | 15 | 15 | 0 | 0.9505 | 0.2156 | 0.0127 |
| 29 | 0 | 0 | 4 | 0.9737 | 0.1446 | 0.0007 |
| 30 | 4 | 16 | 15 | 0.5767 | 0.9879 | 0.0251 |
| 31 | 15 | 15 | 5 | 0.9415 | 0.2447 | 0.0013 |

### graph_input_keyed: {'composition': 'train', 'condition': 'size_32', 'corruption': 0.0, 'depth': 4, 'distractors': 4, 'nodes': 32}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 20 | 20 | 0 | 0.9991 | 0.0074 | 0.0000 |
| 1 | 0 | 0 | 28 | 0.9637 | 0.1756 | 0.0021 |
| 2 | 28 | 28 | 17 | 0.9787 | 0.1061 | 0.0126 |
| 3 | 17 | 17 | 3 | 0.9860 | 0.0823 | 0.0003 |

### graph_input_keyed: {'composition': 'train', 'condition': 'size_64', 'corruption': 0.0, 'depth': 4, 'distractors': 4, 'nodes': 64}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 28 | 28 | 46 | 0.9982 | 0.0147 | 0.0001 |
| 1 | 46 | 46 | 43 | 0.4977 | 0.8104 | 0.0021 |
| 2 | 43 | 12 | 9 | 0.8872 | 0.4404 | 0.0028 |
| 3 | 9 | 7 | 62 | 0.9014 | 0.3805 | 0.0006 |

### graph_input_keyed: {'composition': 'heldout', 'condition': 'heldout_composition', 'corruption': 0.0, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9991 | 0.0073 | 0.0002 |
| 1 | 0 | 0 | 4 | 0.9843 | 0.0936 | 0.0007 |
| 2 | 4 | 16 | 12 | 0.5025 | 1.0193 | 0.0027 |
| 3 | 12 | 12 | 14 | 0.9902 | 0.0622 | 0.0015 |

### graph_input_keyed: {'composition': 'train', 'condition': 'deep_large', 'corruption': 0.0, 'depth': 32, 'distractors': 4, 'nodes': 64}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 28 | 28 | 46 | 0.9982 | 0.0147 | 0.0001 |
| 1 | 46 | 46 | 43 | 0.4977 | 0.8104 | 0.0021 |
| 2 | 43 | 12 | 9 | 0.8872 | 0.4404 | 0.0028 |
| 3 | 9 | 7 | 62 | 0.9014 | 0.3805 | 0.0006 |
| 4 | 62 | 7 | 23 | 0.9931 | 0.0437 | 0.0002 |
| 5 | 23 | 12 | 57 | 0.9787 | 0.1267 | 0.0101 |
| 6 | 57 | 62 | 33 | 0.9786 | 0.1166 | 0.0018 |
| 7 | 33 | 16 | 3 | 0.9658 | 0.1830 | 0.0002 |
| 8 | 3 | 6 | 22 | 0.6978 | 0.8582 | 0.0013 |
| 9 | 22 | 53 | 52 | 0.9336 | 0.2737 | 0.0010 |
| 10 | 52 | 47 | 19 | 0.9500 | 0.2496 | 0.0002 |
| 11 | 19 | 30 | 39 | 0.8266 | 0.6109 | 0.0052 |
| 12 | 39 | 64 | 49 | 0.7158 | 0.7363 | 0.0026 |
| 13 | 49 | 7 | 54 | 0.9631 | 0.1677 | 0.0010 |
| 14 | 54 | 12 | 50 | 0.9609 | 0.2027 | 0.0000 |
| 15 | 50 | 7 | 12 | 0.9864 | 0.0800 | 0.0004 |
| 16 | 12 | 7 | 62 | 0.9957 | 0.0297 | 0.0001 |
| 17 | 62 | 12 | 23 | 0.9799 | 0.1206 | 0.0103 |
| 18 | 23 | 62 | 57 | 0.9787 | 0.1159 | 0.0039 |
| 19 | 57 | 23 | 61 | 0.8052 | 0.5699 | 0.0011 |
| 20 | 61 | 64 | 33 | 0.8679 | 0.5065 | 0.0036 |
| 21 | 33 | 7 | 0 | 0.7289 | 0.6543 | 0.0060 |
| 22 | 0 | 7 | 9 | 0.9913 | 0.0537 | 0.0017 |
| 23 | 9 | 7 | 62 | 0.9961 | 0.0275 | 0.0001 |
| 24 | 62 | 7 | 47 | 0.9962 | 0.0266 | 0.0001 |
| 25 | 47 | 7 | 27 | 0.9962 | 0.0266 | 0.0014 |
| 26 | 27 | 12 | 50 | 0.9795 | 0.1222 | 0.0000 |
| 27 | 50 | 62 | 12 | 0.9785 | 0.1171 | 0.0112 |
| 28 | 12 | 47 | 7 | 0.9800 | 0.1188 | 0.0001 |
| 29 | 7 | 30 | 12 | 0.8016 | 0.6363 | 0.0020 |
| 30 | 12 | 16 | 62 | 0.6093 | 1.0920 | 0.0081 |
| 31 | 62 | 27 | 47 | 0.9172 | 0.3580 | 0.0007 |

### graph_input_keyed: {'composition': 'train', 'condition': 'corruption_0.1', 'corruption': 0.1, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9991 | 0.0073 | 0.0002 |
| 1 | 0 | 0 | 11 | 0.9844 | 0.0933 | 0.0539 |
| 2 | 11 | 11 | 14 | 0.9450 | 0.2312 | 0.0044 |
| 3 | 14 | 14 | 12 | 0.9751 | 0.1212 | 0.0092 |

### graph_input_keyed: {'composition': 'train', 'condition': 'corruption_0.25', 'corruption': 0.25, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9991 | 0.0073 | 0.0002 |
| 1 | 0 | 2 | 11 | 0.9341 | 0.2467 | 0.0002 |
| 2 | 11 | 7 | 14 | 0.9962 | 0.0254 | 0.0001 |
| 3 | 14 | 9 | 12 | 0.9917 | 0.0509 | 0.0138 |

### graph_input_keyed: {'composition': 'train', 'condition': 'corruption_0.5', 'corruption': 0.5, 'depth': 4, 'distractors': 4, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 8 | 8 | 0 | 0.9991 | 0.0073 | 0.0002 |
| 1 | 0 | 2 | 11 | 0.9341 | 0.2467 | 0.0002 |
| 2 | 11 | 7 | 14 | 0.9962 | 0.0255 | 0.0001 |
| 3 | 14 | 9 | 12 | 0.9917 | 0.0509 | 0.0138 |

### graph_input_keyed: {'composition': 'train', 'condition': 'distractors_0', 'corruption': 0.0, 'depth': 4, 'distractors': 0, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 4 | 4 | 15 | 0.9991 | 0.0075 | 0.0179 |
| 1 | 15 | 15 | 5 | 0.9867 | 0.0760 | 0.0003 |
| 2 | 5 | 5 | 6 | 0.9960 | 0.0260 | 0.0030 |
| 3 | 6 | 6 | 12 | 0.9622 | 0.1624 | 0.0000 |

### graph_input_keyed: {'composition': 'train', 'condition': 'distractors_16', 'corruption': 0.0, 'depth': 4, 'distractors': 16, 'nodes': 16}

| Step | True pre-step node | Most likely grounded node | Actual next node | Grounding max probability | Entropy | Next-token attention |
|---:|---|---|---|---:|---:|---:|
| 0 | 15 | 15 | 0 | 0.9991 | 0.0076 | 0.0018 |
| 1 | 0 | 0 | 4 | 0.9875 | 0.0778 | 0.0006 |
| 2 | 4 | 16 | 15 | 0.5477 | 1.0046 | 0.0240 |
| 3 | 15 | 15 | 0 | 0.9143 | 0.3255 | 0.0185 |

## Resource and model records

Raw per-run records below preserve resource units, model parameters, learned strengths and temperatures. Resource comparisons include the runner’s stated instrumentation; no asymptotic scaling law is inferred.

Process RSS is a high-water mark, not per-model allocated memory. Plain inference timing excludes diagnostic extraction; total run time includes evaluation and instrumentation.

| Variant | Total seconds | Training seconds | Plain inference batch seconds | Process peak RSS MiB |
|---|---:|---:|---:|---:|
| graph_input_keyed | 3.5926 ± 0.2100 | 1.9144 ± 0.1726 | 0.0036 ± 0.0004 | 420.5781 ± 0.0000 |
| none_keyed | 2.9215 ± 0.0133 | 1.3111 ± 0.0111 | 0.0021 ± 0.0001 | 420.5781 ± 0.0000 |
| soft_keyed | 5.0728 ± 1.2640 | 3.0818 ± 1.0030 | 0.0045 ± 0.0018 | 420.5013 ± 0.1331 |

### soft_keyed, seed 0

```json
{
  "model": {
    "parameters": 9303,
    "strengths": [
      [
        [
          15.999725341796875,
          16.002933502197266,
          16.010910034179688
        ],
        [
          15.998520851135254,
          15.998358726501465,
          15.999744415283203
        ],
        [
          16.053159713745117,
          16.036535263061523,
          16.014389038085938
        ],
        [
          16.070493698120117,
          16.068904876708984,
          16.03765869140625
        ]
      ]
    ],
    "temperatures": {
      "grounders.0": 0.04999999701976776
    }
  },
  "resources": {
    "device": "cpu",
    "inference_batch_seconds": 0.00660529350861907,
    "inference_batch_size": 32,
    "inference_depth": 4,
    "inference_repeats": 10,
    "peak_rss_mib": 420.34765625,
    "seconds": 4.955719172954559,
    "threads": 2,
    "torch_version": "2.14.0+cpu",
    "train_seconds": 2.86309964209795
  }
}
```

### graph_input_keyed, seed 0

```json
{
  "model": {
    "parameters": 9303,
    "strengths": [
      [
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ]
      ]
    ],
    "temperatures": {
      "grounders.0": 0.04999999701976776
    }
  },
  "resources": {
    "device": "cpu",
    "inference_batch_seconds": 0.0041111208032816645,
    "inference_batch_size": 32,
    "inference_depth": 4,
    "inference_repeats": 10,
    "peak_rss_mib": 420.578125,
    "seconds": 3.7958973329514265,
    "threads": 2,
    "torch_version": "2.14.0+cpu",
    "train_seconds": 2.085211447440088
  }
}
```

### none_keyed, seed 0

```json
{
  "model": {
    "parameters": 9303,
    "strengths": [
      [
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ]
      ]
    ],
    "temperatures": {
      "grounders.0": 0.04999999701976776
    }
  },
  "resources": {
    "device": "cpu",
    "inference_batch_seconds": 0.0022445017006248237,
    "inference_batch_size": 32,
    "inference_depth": 4,
    "inference_repeats": 10,
    "peak_rss_mib": 420.578125,
    "seconds": 2.906160688959062,
    "threads": 2,
    "torch_version": "2.14.0+cpu",
    "train_seconds": 1.2984250858426094
  }
}
```

### soft_keyed, seed 1

```json
{
  "model": {
    "parameters": 9303,
    "strengths": [
      [
        [
          16.041839599609375,
          15.991293907165527,
          16.024677276611328
        ],
        [
          16.03767967224121,
          15.979802131652832,
          16.002059936523438
        ],
        [
          16.012557983398438,
          16.059877395629883,
          16.063932418823242
        ],
        [
          16.063941955566406,
          16.072195053100586,
          16.08579444885254
        ]
      ]
    ],
    "temperatures": {
      "grounders.0": 0.04999999701976776
    }
  },
  "resources": {
    "device": "cpu",
    "inference_batch_seconds": 0.003541586408391595,
    "inference_batch_size": 32,
    "inference_depth": 4,
    "inference_repeats": 10,
    "peak_rss_mib": 420.578125,
    "seconds": 3.8714814512059093,
    "threads": 2,
    "torch_version": "2.14.0+cpu",
    "train_seconds": 2.2061855224892497
  }
}
```

### graph_input_keyed, seed 1

```json
{
  "model": {
    "parameters": 9303,
    "strengths": [
      [
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ]
      ]
    ],
    "temperatures": {
      "grounders.0": 0.04999999701976776
    }
  },
  "resources": {
    "device": "cpu",
    "inference_batch_seconds": 0.0034045275300741197,
    "inference_batch_size": 32,
    "inference_depth": 4,
    "inference_repeats": 10,
    "peak_rss_mib": 420.578125,
    "seconds": 3.6053715338930488,
    "threads": 2,
    "torch_version": "2.14.0+cpu",
    "train_seconds": 1.9180963337421417
  }
}
```

### none_keyed, seed 1

```json
{
  "model": {
    "parameters": 9303,
    "strengths": [
      [
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ]
      ]
    ],
    "temperatures": {
      "grounders.0": 0.04999999701976776
    }
  },
  "resources": {
    "device": "cpu",
    "inference_batch_seconds": 0.0019651309587061403,
    "inference_batch_size": 32,
    "inference_depth": 4,
    "inference_repeats": 10,
    "peak_rss_mib": 420.578125,
    "seconds": 2.929497722070664,
    "threads": 2,
    "torch_version": "2.14.0+cpu",
    "train_seconds": 1.3160636969842017
  }
}
```

### soft_keyed, seed 2

```json
{
  "model": {
    "parameters": 9303,
    "strengths": [
      [
        [
          16.0139102935791,
          15.991893768310547,
          15.977259635925293
        ],
        [
          16.019664764404297,
          15.996842384338379,
          15.967046737670898
        ],
        [
          16.012680053710938,
          16.024808883666992,
          16.027681350708008
        ],
        [
          16.06916618347168,
          16.061321258544922,
          16.046836853027344
        ]
      ]
    ],
    "temperatures": {
      "grounders.0": 0.04999999701976776
    }
  },
  "resources": {
    "device": "cpu",
    "inference_batch_seconds": 0.003386597614735365,
    "inference_batch_size": 32,
    "inference_depth": 4,
    "inference_repeats": 10,
    "peak_rss_mib": 420.578125,
    "seconds": 6.391343955881894,
    "threads": 2,
    "torch_version": "2.14.0+cpu",
    "train_seconds": 4.176082438789308
  }
}
```

### graph_input_keyed, seed 2

```json
{
  "model": {
    "parameters": 9303,
    "strengths": [
      [
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ]
      ]
    ],
    "temperatures": {
      "grounders.0": 0.04999999701976776
    }
  },
  "resources": {
    "device": "cpu",
    "inference_batch_seconds": 0.0033622717019170523,
    "inference_batch_size": 32,
    "inference_depth": 4,
    "inference_repeats": 10,
    "peak_rss_mib": 420.578125,
    "seconds": 3.3764091371558607,
    "threads": 2,
    "torch_version": "2.14.0+cpu",
    "train_seconds": 1.7399948467500508
  }
}
```

### none_keyed, seed 2

```json
{
  "model": {
    "parameters": 9303,
    "strengths": [
      [
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ],
        [
          16.0,
          16.0,
          16.0
        ]
      ]
    ],
    "temperatures": {
      "grounders.0": 0.04999999701976776
    }
  },
  "resources": {
    "device": "cpu",
    "inference_batch_seconds": 0.002082074899226427,
    "inference_batch_size": 32,
    "inference_depth": 4,
    "inference_repeats": 10,
    "peak_rss_mib": 420.578125,
    "seconds": 2.9287198297679424,
    "threads": 2,
    "torch_version": "2.14.0+cpu",
    "train_seconds": 1.3188225659541786
  }
}
```

## Provenance

```json
{
  "analysis_artifact": {
    "analyzer_file": "src/topoformer/grounding_analysis.py",
    "analyzer_sha256": "5f9725b42562edadff60170392d65216767433ed44fd3929b4f95ff855bb700a",
    "metrics_sha256": "b57301e889ee98229e1ace0278b1a8ef379975df228a3b753c6e4daebf10ee49"
  },
  "config_hash": "1a38fd07724e23501b5ee101e4f4e8e0d47a6d9caafc733d61078a15cad71e74",
  "source": {
    "commit": "27ca0a2c1f6f77f0072eba69be513cd985603280",
    "dirty": false,
    "files": {
      "src/topoformer/__init__.py": "a597933984bcde7b51e2dcc92f39749bc2c7600623265647e60156eab7265730",
      "src/topoformer/attention.py": "ac1486c8f5398bba9e084e2ddda85ba402ed9633638e68029c54149fe0152854",
      "src/topoformer/data.py": "c4ce845820c7efb26a701c0dfcd2f6ecbcb7e38b045f3918e61416e17b387b21",
      "src/topoformer/evaluation.py": "c5fe751c5e6cb55d3ee88a2c3f1f8b08e8513438a76f11eeeb39aeb34a86ee11",
      "src/topoformer/experiment.py": "fa34e5fe18132ad1a4d59bfc42fc74949aeabe88977877a8b67b087757f3d763",
      "src/topoformer/graphs.py": "16ce61cc40a48d9e294fa215f6ec512c4fdc2d82273867b59146f75e112d9a07",
      "src/topoformer/grounding.py": "2296c904aa8e997c219e67a7e0d4f9791723a4fc05d168668537731dde449ae7",
      "src/topoformer/grounding_analysis.py": "2653ea13b14e52f88df78d0c8d67e4dc7540b69c6058ef13b362d2956e3fc9e2",
      "src/topoformer/grounding_study.py": "f2f51cfa2801332aab2edad6a28069a8890afa2a3b2c373aa31f8c14e6820927",
      "src/topoformer/model.py": "3eeb52c71fb6a23dcee7a7064fb198b17a8ccbc7eae05ec918075c5a3ca5237b",
      "src/topoformer/runtime_graph.py": "c7c9884febf8dc15009ddb337ae4c885058186d0faa7741c38247697e9e3eefe",
      "src/topoformer/study.py": "27c6667a7c0ea68cbd2b33796b95ef3bc3083b399078946336e80c62352fe71b",
      "src/topoformer/study_analysis.py": "b012d9edb9756fb641a92f6331377e01bb2e9f84d638c44cc764051b8fe461a3",
      "src/topoformer/study_data.py": "59d3293c7114598ff14d533f0cf040184068d67c77f36418286431cf276a1b1d",
      "src/topoformer/study_model.py": "455ed9b4e8eea422cb1da5db36dd525319b1bae81743bc407586b1565170e94f",
      "src/topoformer/training.py": "697cfd27c0db940bcc489f02e25b7504aaff80305a46dc24dc465cbd11176a43",
      "src/topoformer/traversal_data.py": "714cf6d2cecf03bccc3ad7dd17cb092fa233b6599d8e0a702ef62fc579a874a6",
      "src/topoformer/traversal_model.py": "3bce439db58f4482b7760ae174a103eda151958e3a44f281d675aa72634afeb6",
      "src/topoformer/traversal_oracle.py": "d690fe5d4b10ce8accf5dd8d1157166194a5dd6fde80ff4d122c1e52cf3bcb66"
    }
  }
}
```

## Figures

- [depth-f814afda.png](figures/depth-f814afda.png)
- [depth-f814afda.svg](figures/depth-f814afda.svg)
- [depth-9c4cee73.png](figures/depth-9c4cee73.png)
- [depth-9c4cee73.svg](figures/depth-9c4cee73.svg)
- [nodes-171c914e.png](figures/nodes-171c914e.png)
- [nodes-171c914e.svg](figures/nodes-171c914e.svg)
- [nodes-d54076bf.png](figures/nodes-d54076bf.png)
- [nodes-d54076bf.svg](figures/nodes-d54076bf.svg)
- [corruption-55ff183a.png](figures/corruption-55ff183a.png)
- [corruption-55ff183a.svg](figures/corruption-55ff183a.svg)
- [distractors-2da157d3.png](figures/distractors-2da157d3.png)
- [distractors-2da157d3.svg](figures/distractors-2da157d3.svg)
- [diagnostics-a464ec74.png](figures/diagnostics-a464ec74.png)
- [diagnostics-a464ec74.svg](figures/diagnostics-a464ec74.svg)
- [diagnostics-b9e54006.png](figures/diagnostics-b9e54006.png)
- [diagnostics-b9e54006.svg](figures/diagnostics-b9e54006.svg)
- [diagnostics-a39b6ef8.png](figures/diagnostics-a39b6ef8.png)
- [diagnostics-a39b6ef8.svg](figures/diagnostics-a39b6ef8.svg)
- [training-curves-keyed.png](figures/training-curves-keyed.png)
- [training-curves-keyed.svg](figures/training-curves-keyed.svg)
