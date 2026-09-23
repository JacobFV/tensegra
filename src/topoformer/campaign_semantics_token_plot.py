"""Static matched-exposure S06 comparison from frozen development archives."""
import argparse,gzip,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('output');a=p.parse_args()
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for folder,label,color in [('s01-current-n8192-dev201','8-row workspace','#0072B2'),('s06-token-n8192-dev201','Public-token state','#D55E00')]:
    x=json.load(gzip.open(Path(a.results)/folder/'manifest.json.gz','rt'));curves=x['curves'];steps=[r['presentations']/1000 for r in curves]
    for ax,key,title in [(axes[0,0],'copy','Canonical identity copying'),(axes[0,1],'typed_edge_f1','Typed-edge F1, TRAIN-calibrated'),(axes[1,0],'ordered_edge_f1','Ordered-edge F1, TRAIN-calibrated'),(axes[1,1],'exact','Complete canonical graphs')]:
        scores=[r['dev_calibrated'][key] for r in curves]
        ax.plot(steps,scores,'o-',label=label,color=color);ax.set(title=title,xlabel='Optimizer presentations (thousands)',ylabel='DEV score',ylim=(-.02,1.02));ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('S06: same initialization, 8,192 TRAIN graphs and 512 DEV graphs\nOne development seed; 1,024 dimensions; attention-token allocation and compute differ')
fig.savefig(a.output,metadata={'Date':None})
