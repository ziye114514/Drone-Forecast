"""Static scientific plot of the measured common-target errors."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/optimization_resume_20261026'
d=json.loads((WORK/'output/single_model_optimized_RAW_20261026/scores.json').read_text('utf-8'))
fig,axes=plt.subplots(1,2,figsize=(11.5,4.5));horizons=['0.1','0.2','0.4'];x=np.arange(3);width=.34
for ax,case,title in zip(axes,['side','train2'],['Side view, same targets','Second recording, same targets']):
    rows=d['results'][case]['scores']
    old=[rows[h]['metrics']['original_model_replay']['mean'] for h in horizons]
    new=[rows[h]['metrics']['merged_scale']['mean'] for h in horizons]
    ax.bar(x-width/2,old,width,label='Original, offline same-frame reference',color='#8794a5')
    ax.bar(x+width/2,new,width,label='Single merged model + scale',color='#347ab7')
    for i,h in enumerate(horizons):ax.text(i,max(old[i],new[i])+2,'n='+str(rows[h]['metrics']['merged_scale']['n']),ha='center',fontsize=10)
    ax.set_xticks(x,horizons);ax.set_xlabel('Future horizon from actual publication (s)');ax.set_ylabel('Mean center endpoint error (px)')
    ax.set_title(title);ax.set_ylim(0,max(max(old),max(new))*1.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for spine in ['top','right']:ax.spines[spine].set_visible(False)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
fig.suptitle('Small development gains; held-out motion did not improve',fontsize=13)
fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(OUT/'mean_errors.png',dpi=160,bbox_inches='tight');plt.close(fig)
print('Saved mean_errors.png; no uncertainty/significance claim')
