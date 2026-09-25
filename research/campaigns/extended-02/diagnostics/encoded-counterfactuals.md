# Encoded-input counterfactual audit (post research-audit)

This audit implements the external research audit's first recommendation: check what the policy's *encoded* inputs can distinguish before interpreting provenance or recovery failures. It builds paired public states and compares the exact observation and candidate vectors under m1, m2 and m3.
- Script: `research/tools/campaign02_encoded_counterfactuals.py`
- Output: `encoded-counterfactuals.json`
- Cost: about 2 CPU-s, no GPU, no remote host.

| Paired states | m1 | m2 | m3 |
|---|---|---|---|
| `use_return` on own current-draft CSP result vs same-type foreign (prior) CSP result; same status, validity and retrieval | **identical** | differ (coords 76, 77) | differ |
| `use_return` on own current result vs own *stale* result (same draft handle, called before constraints were added) | **identical** | differ (coord 77, snapshot match) | differ |
| After a commit rejected for **capacity** vs one rejected for **funds** (observation + `commit_pending` candidate) | **identical** | **identical** | **identical** |
| "Commit rejected, then one other action" vs "never committed" (step-matched) | **identical** | **identical** | differ |

**Implications.**
- **E19.** The E19 collapse under same-type distractors was an **interface limit**: the m1 controller received bit-identical inputs for the right and wrong return. It is not evidence that provenance cannot be learned.
- **E20.** m2's success is an input-contract correction, and m2 also covers the stale-draft case.
- **E22.** The rejection history is erased from m1/m2 inputs after any intervening action, which supports E22's memory-gap reading at the input level.
- **Rejection reason.** No feature version encodes the reason (capacity vs funds), and the `commit_pending` candidate carries no pending weight/price sums. A controller cannot tell *why* a direct commit failed except through other candidates' features.
- **Not yet tested.** Old-but-still-valid reuse (row 4 of the audit's applicability table) and randomized, draft-like names for foreign records (to remove m2's `prior_i` vs `problem_n` near-oracle) remain untested.
