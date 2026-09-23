"""Static S01/S03/S04 development exposure curve from archived summaries."""
import argparse,gzip,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('output');a=p.parse_args();curves={}
for name in ['s01-current-n8192-dev201','s03-current-n8192-131k-dev201','s04-current-n8192-262k-dev201']:
    for row in json.load(gzip.open(Path(a.results)/name/'manifest.json.gz','rt'))['curves']:curves[row['presentations']]=row
rows=[curves[k] for k in sorted(curves) if k>=8192];steps=[r['presentations']/1000 for r in rows]
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,key,title in [(axs[0,0],'copy','Visible identity copying (graph-macro)'),(axs[0,1],'typed_edge_f1','Typed-edge F1'),(axs[1,0],'ordered_edge_f1','Ordered-edge F1'),(axs[1,1],'exact','Complete canonical graph accuracy')]:
    for field,color,style,label,denom in [('train_calibrated','#D55E00','--','TRAIN128 diagnostic, calibrated',128),('dev_calibrated','#0072B2','-','DEV512, TRAIN-calibrated',512)]:
        vals=[r[field][key]/denom if key=='exact' else r[field][key] for r in rows];ax.plot(steps,vals,style,marker='o',color=color,label=label)
    if key in ('typed_edge_f1','ordered_edge_f1'):ax.plot(steps,[r['dev_raw'][key] for r in rows],':',marker='s',color='#777777',label='DEV512, raw')
    ax.set(title=title,xlabel='Cumulative presentations (thousands)',ylabel='Score',ylim=(-.02,1.02));ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right' if key!='exact' else 'upper right')
    for boundary in (65.536,131.072):ax.axvline(boundary,color='black',alpha=.2,linewidth=.7)
fig.suptitle('Current actor: same optimizer trajectory, 8,192 distinct TRAIN graphs\nOne development initialization; final262k endpoint retained despite regression',fontsize=12)
fig.savefig(a.output,metadata={'Date':None})
