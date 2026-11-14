"""Scientific learning/availability trace; no metric or model changes."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(source,target,title):
    with Path(source).open(encoding='utf-8') as f:rows=[json.loads(s) for s in f]
    t=np.array([r['t'] for r in rows]);d=[r['diagnostics'] for r in rows]
    fig,axes=plt.subplots(4,1,figsize=(11,8),sharex=True)
    axes[0].plot(t,[q['event_gain'] for q in d],label='Effective image gain');axes[0].set_ylabel('px/s2 per feature');axes[0].legend(loc='upper left')
    axes[1].plot(t,[q['event_gain_updates'] for q in d],label='Matured gain updates');axes[1].axhline(10,color='#bb4444',linestyle='--',label='Existing enable gate:10');axes[1].set_ylabel('Update count');axes[1].legend(loc='upper left')
    axes[2].plot(t,[np.nan if q['periodic']['ratio'] is None else q['periodic']['ratio'] for q in d]);axes[2].set_ylabel('Relative f-squared');axes[2].axhline(1,color='#999999',linestyle=':')
    axes[3].step(t,[q['fresh'] for q in d],where='post',label='Published physical support')
    axes[3].step(t,[.5*bool(q['frequency']['current']) for q in d],where='post',alpha=.6,label='Current frequency input (height0.5)')
    axes[3].set_yticks([0,.5,1]);axes[3].set_ylabel('Availability');axes[3].set_xlabel('Native source time (s)');axes[3].legend(loc='upper left')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(title+'\nGain is an image-domain coefficient, not mass, wind or calibrated thrust')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(target,dpi=155);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--target',required=True);p.add_argument('--title',required=True);a=p.parse_args();plot(a.source,a.target,a.title)
