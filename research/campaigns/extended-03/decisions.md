# Decisions (extended-03)

- 2026-09-25T05:31Z: User instructed full ownership, including managing subagents. Declared default ceilings, the same as extended-02: 48 CPU core-h, 12 GPU device-h (device-occupancy accounting, accepted by the user in extended-02), 24 h wall (deadline 2026-09-26T05:31Z), ~20% reserved for confirmation and closure. No paid services. Machine verified idle; branch campaign/extended-03 from main e80ddacb.
- Execution host and wrapper as in extended-02 (gb10-direct, campaign02_job.py receipts). Only root launches jobs; subagents implement, test and audit, and never launch remote jobs.
- Stage A: depworld-v1 per world-spec.md, delegated to a builder subagent. Stage B preflight observability audit to be done by a *different* subagent (independence from the encoder author). Training is gated on Stage A leverage and the Stage B preflight.
