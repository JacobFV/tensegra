# Future lossless export compactness

Read-only inspection of immutable A08 `public.npz` (74,615,588bytes) found64 arrays but only28 unique exact `.npy` byte streams by SHA256. Duplicate entries contribute37,314,921compressed bytes; unique streams occupy37,292,261compressed bytes. A canonical-array archive plus alias manifest would therefore halve this artifact with negligible manifest overhead. No current artifact or frozen A10 export was modified.

All public arrays already useuint8 and ZIP DEFLATE. The main duplication is clean gold/start/relation/successor/adjacency repeated across four fractions at each shape, plus fraction0 supplied arrays identical to clean arrays. ZIP compression does not deduplicate across separate members.

For a future versioned exporter, store each exact array stream once under its SHA256 and map logical field names to those hashes in a schema-versioned JSON manifest. Preserve shape/dtype and checksums; the loader can reconstruct every original logical array byte-for-byte. This avoids changing metric definitions or relying on regeneration seeds. Add a round-trip equality test before adopting it. Shape-level shared clean arrays with fraction-specific supplied fields are a simpler domain-specific alternative.

Adjacency bit-packing or fixed-degree destination lists could reduce uncompressed memory further, but compression gains were not measured and need not complicate the first change. Current uint8 node indices fitN≤128; any future larger-node export must choose a wider index dtype rather than silently truncating. None of these recommendations authorizes rewriting existing history or altering the A10 source/config freeze.
