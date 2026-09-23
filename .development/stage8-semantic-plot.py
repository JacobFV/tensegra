"""Plot reported semantic curves; does not fit models or select thresholds."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('summary',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
data=json.loads(args.summary.read_text())['aggregate']
args.output.mkdir(parents=True,exist_ok=True)
colors={'semantic':'#1565c0','no_input':'#c62828','frequency':'#666666'}
for decoder,name in [('evaluation','raw'),('calibrated_evaluation','train-calibrated')]:
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharey=True,constrained_layout=True)
    for row,corpus in enumerate((1000,10000)):
        for col,surface in enumerate(('english','spanish','symbols','heldout_lexicon')):
            ax=axes[row,col]
            for arm in ('semantic','no_input','frequency'):
                points=sorted([r for r in data if r['decoder']==decoder and r['corpus']==corpus and r['arm']==arm and r['surface']==surface and r['lesson']=='all'],key=lambda r:r['presentations'])
                if not points:continue
                x=[p['presentations'] for p in points]
                y=[p['typed_edge_f1']['mean'] for p in points]
                if arm=='frequency':
                    ax.axhline(y[0],color=colors[arm],linestyle=':',label=arm)
                else:
                    ax.plot(x,y,'o-',color=colors[arm],label=arm)
                    ax.fill_between(x,[p['typed_edge_f1']['minimum'] for p in points],[p['typed_edge_f1']['maximum'] for p in points],alpha=.15,color=colors[arm])
            ax.set_xscale('symlog',linthresh=1000)
            ax.set_xticks([0,10016,100000],['0','10,016','100,000'])
            ax.set_ylim(0,1);ax.grid(alpha=.2)
            ax.set_title(f'N={corpus:,}; {surface}')
            if col==0:ax.set_ylabel('Typed-edge F1')
            if row==1:ax.set_xlabel('Optimizer presentations')
    axes[0,0].legend(loc='upper left',fontsize=8)
    fig.suptitle(f'Width1024 semantic acquisition — {name} decoder\nLines: seed means; shaded bands: observed seed ranges')
    fig.savefig(args.output/f'typed-edge-{name}.png',dpi=160)
    plt.close(fig)
