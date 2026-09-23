# Proposed next direct-attention task: content selection within typed neighborhoods

Design only. No implementation, training, or promotion before the full A03 confirmation is reviewed.

A03 deliberately makes the selected relation functional: the graph already identifies the sole next node. A more informative next question gives content matching a real role while preserving an auditable computation.

## Candidate benchmark

Each graph has three directed relation types. Nodes also carry one of K public attribute codes. For every source and relation, exactly one outgoing neighbor is sampled from each attribute group, giving outdegree K. The group codes are fresh normalized continuous vectors per graph, not a fixed vocabulary of memorized names. Node identifiers and payload classes are independently randomized. Start with N32 and K4, so eight different nodes share each attribute code.

Each public instruction is an ordered pair `(relation, desired_attribute_code)`. The intended next node is the unique neighbor under that relation whose attribute code matches the instruction. Several global nodes share that code, so the terminal attribute never uniquely reveals the final entity or value. A supplied ordered instruction sequence determines the path. The graph and content are both necessary; neither alone determines the answer.

An exact intersection selector provides the programmed reference. Models instead learn query/key matching between instruction attributes and immutable node attributes, then propagate payload representations. The same public reverse schedule from A03 may be reused for all-node suffix computation. It remains a supplied execution schedule, not learned planning. All models receive enough recurrent steps for the full supplied path.

## Strong comparisons

- Soft structural attention: learned attribute QK compatibility plus trainable typed adjacency bias.
- Hard structural attention: the same learned QK selector, restricted to supplied neighbors.
- Competitive neighbor-attention/message passing: learned content-dependent weights over neighbors, rather than an intentionally inadequate uniform mean. If implemented with the same QK scorer, it is algebraically the hard-attention comparator; report that equality rather than invent independent evidence.
- Structured address context: a lossless list of neighbor addresses, with a strong known-key retrieval prior, followed by the same learned attribute selector. Report its additional retrieval compute. Avoid making it weak through unnecessary address dilution.
- No-graph and no-content controls: information ablations, not the main equal-information comparisons.

Workspace width stays1024. Initially change only the generator and content-selector inputs of the existing shared value-processing cell. The query/key selector starts without an identity-matching initialization; attribute matching must be acquired. Preserve the successful payload encoder/update/readout architecture. Match supervised value targets across methods and disclose that intermediate suffix targets are privileged. A direct route-label auxiliary would be a separately named intervention if answer/value supervision fails.

## What would be learned

The intended learned interface is equality/similarity matching for fresh continuous attribute codes in conjunction with supplied typed adjacency. Graph wiring, the unique-match generator contract, the schedule, and the exact reference are programmed. This is neither language grounding nor newly induced algorithm planning.

First test acquisition and fresh graphs/codes under N32,K4,D≤4. Then separate larger N, longer D, new relation/selector compositions, and larger K within representational support. Retain exact identity-path diagnostics alongside payload accuracy because different nodes can share payloads. Use matched graph and attribute inputs for every strong comparator.

Wrong or missing graph edges can remove the information required to distinguish several nodes with the same code. Such corruption measures reliance and uncertainty, not guaranteed recovery from absent information. Do not use corruption as an automatic argument for soft topology. A later recovery task would require legitimate redundant public information and its own protocol.

Before promotion: freeze a numerical development budget and learning-curve decision rule after profiling; predeclare the strongest baselines; then use fresh three-seed confirmation only for an informative acquired mechanism. A03 may already be sufficient to settle the narrower supplied-functional-routing question.
