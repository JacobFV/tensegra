# S16 prospective main freeze

Root accepted a480-second whole-wrapper cap after the timing-only profile. `S16-main-frozen-v1.json` SHA256 is `1e95311a823bf753d7496852a954d9c2d39e212687970adb420641e2cba60496`. It binds the profile cost proposal, profile manifest and lifecycle receipt by SHA256 through the new budget-prepared configuration. The original uncapped prepared file is preserved. Source hashes are exactly identical to the executed profile-v2 source c63c4d6c; model/checkpoint/threshold/data/normalization policy is unchanged. All six fixed models process all1,024 original inputs; normalized-renamed outcomes remain an identical-input implication, not independent forwards. No outcome selection or tuning occurred.

The main freeze is prospective and grants no launch authority. Independent profile/budget review and a separate root release are required. From the separately staged immutable source snapshot, run only after release:

```sh
/home/brandonin/topoformer-stage8-cuda/bin/python \
  src/topoformer/campaign_semantics_s16_launch.py \
  research/campaigns/extended-01/semantics/S16-main-frozen-v1.json \
  --cap 480 --python /home/brandonin/topoformer-stage8-cuda/bin/python \
  --prefix /home/brandonin/topoformer-campaign01-semantics/results/s16-main-launch
```

Wrap the complete process in GNU time, retain command/hash and0.01-second precision, conservatively charge outer duration, and report GPUFREE before outcome analysis. Preserve partials and stop on failure/timeout; no automatic extension or omitted models.
