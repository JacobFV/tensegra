# Stage 8 return memory

Default workspace width is 1024. Six encoders split exactly that many informative coordinates among value, type, operation, ordered argument identities and provenance. Compressed encoding concatenates those channels into one token; factorized encoding supplies six padded tokens. Both use the same learned mixer and identical parameter sets. Factorization allocates six times as many KV coordinates; this cost is explicitly reported, not called memory-matched.

Availability is independently once/persistent. Initial access is an identical learned cross-attention read in both arms. Zero microsteps measures acquisition after that read, without a recurrent cell. Every microstep reuses the existing four-block current-state-only workspace cell. Heads consume only six workspace rows, never event fields directly. Exact stored event integrity is architectural, separate from learned prediction.

Reuse Stage 7 valid primitive return events and nonce identities. Release/overwrite modifies only external storage, never clears free workspace. Dropped/wrong value/type/provenance and unrelated activity are interventions. Lifecycle stale reconstruction is diagnostic, not required magical forgetting. The first study measures zero/one-step acquisition; long retention and downstream usage remain gate-dependent. No task-use head, runtime scheduler, or composition is added.

Timing config remains width1024, batch2 and two CPU threads. Unit tests explicitly use a tiny width as test-only mechanical fixtures. No training authorized yet.

A third `mixed` arm uses six full-width field encoders summed before mixing, analogous to Stage 7's undifferentiated additive encoder at width1024. It has more encoder parameters and 6144 pre-sum coordinates; report this explicitly. The `compressed` arm is specifically *disjoint concatenation*, already factorized before compression. Only compressed-versus-factorized tokenization has exactly matched parameters/encoder coordinates; neither is labeled an unfactorized Stage 7 replication.
