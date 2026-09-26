import gzip, json, sys, os
R = os.path.expanduser("~/tensegra-campaign04/results")
for p in ["f1b-f-boot-x1-r3/iid_f0-r_mask/f-boot-x1-r3.jsonl.gz", "f1b-f-boot-x1-r3/iid_f0-r_mask/worlds.jsonl.gz", "f1b-references/iid_f0-greedy/reference-dep_reuse.jsonl.gz"]:
    with gzip.open(os.path.join(R, p), "rt") as f:
        row = json.loads(f.readline())
    def show(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict): print(pre+k, "{dict}"); show(v, pre+"  ")
            elif isinstance(v, list): print(pre+k, "[list len %d]" % len(v), repr(v[:3])[:200])
            else: print(pre+k, repr(v)[:120])
    print("=====", p); show(row)
s = json.load(open(os.path.join(R, "f1b-f-boot-x1-r3/summary.json")))
print(json.dumps(s, indent=1)[:3000])
print(sorted(os.listdir(os.path.join(R, "f-boot-x1-r3"))))
