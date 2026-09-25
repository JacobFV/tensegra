"""Static scientific plot from immutable S01 development metrics; no inference."""
import argparse,gzip,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('replacement');p.add_argument('output');a=p.parse_args()
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for count,color,marker in [(128,'#0072B2','o'),(1024,'#D55E00','s'),(8192,'#009E73','^')]:
    x=json.load(gzip.open(Path(a.results)/f's01-current-n{count}-dev201/manifest.json.gz','rt'))
    curves=x['curves'];steps=[r['presentations']/1000 for r in curves]
    for ax,key in [(axes[0,0],'copy'),(axes[0,1],'ordered_edge_f1')]:
        ax.plot(steps,[r['dev_calibrated'][key] for r in curves],marker=marker,color=color,label=f'{count:,} TRAIN graphs')
    axes[1,0].plot(steps,[r['dev_calibrated']['exact']/512 for r in curves],marker=marker,color=color,label=f'{count:,} TRAIN graphs',alpha=.75)
for ax,title in [(axes[0,0],'Visible identity copying (graph-macro)'),(axes[0,1],'Ordered-edge F1 (TRAIN-calibrated)'),(axes[1,0],'Complete canonical graph accuracy')]:
    ax.set(title=title,xlabel='Optimizer presentations (thousands)',ylabel='Development score',ylim=(-.025,1.025));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
r=json.load(open(a.replacement));r=next(row for row in r['results'] if row['split']=='development' and row['policy']=='calibrated')
labels=['Deployed','Gold edges','Gold edges\n+ slots','Gold node\nattributes'];values=[r['original_exact'],r['replacement_exact']['edges'],r['replacement_exact']['edges_and_slots'],r['replacement_exact']['all_node_attributes']]
ax=axes[1,1];bars=ax.bar(labels,[v/512 for v in values],color=['#777777','#CC79A7','#CC79A7','#CC79A7']);ax.set(title='N=8,192: privileged component replacements',ylabel='Complete graphs / 512',ylim=(0,.32));ax.grid(axis='y',alpha=.2)
for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,b.get_height()+.01,f'{v}/512',ha='center',fontsize=10)
fig.suptitle('S01: matched 65,536-presentation diversity ladder\nOne development initialization (201); same 512 fresh graphs; no confirmation claim',fontsize=13)
fig.savefig(a.output,metadata={'Date':None})
