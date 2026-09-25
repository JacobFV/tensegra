"""Static matched-endpoint LR comparison from archived metrics only."""
import argparse,gzip,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(reference,intervention,output):
    old=json.load(gzip.open(reference,'rt'));new=json.load(gzip.open(intervention,'rt'))
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for name,m,color in [('constant LR 1e-4',old,'#777777'),('reduced LR 1e-5',new,'#1565c0')]:
        curves=[c for c in m['curves'] if 16384<=c['update']<=24576];x=[c['presentations']/1000 for c in curves]
        axes[0].plot(x,[c['dev_calibrated']['exact']/512 for c in curves],marker='o',color=color,label=name)
        axes[1].plot(x,[c['dev_calibrated']['ordered_edge_f1'] for c in curves],marker='o',color=color,label=name)
    axes[0].set_ylabel('Complete graphs / 512');axes[1].set_ylabel('Ordered-edge F1')
    for ax in axes:ax.set_xlabel('Cumulative training presentations (thousands)');ax.grid(alpha=.2)
    axes[0].set_ylim(0,.3);axes[1].set_ylim(.9,1);axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('One development seed; same parent, data and endpoint; TRAIN-calibrated decoding',fontsize=10)
    fig.tight_layout();fig.savefig(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('intervention');p.add_argument('output');a=p.parse_args();main(a.reference,a.intervention,a.output)
