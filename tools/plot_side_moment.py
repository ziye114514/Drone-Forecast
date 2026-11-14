"""Visualize frozen spatial statistics, without new inference or tuning."""
from pathlib import Path
import json,numpy as np
from scipy.signal import lombscargle
ROOT=Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT.parent/'output/fred_transfer_20261029/side_spatial_moment'
    data=json.loads((out/'summary.json').read_text())
    rows=[next(r for r in data['samples'] if r['clip']==name and 'series' in r) for name in ('side','clear')]
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':155})
    fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
    for i,row in enumerate(rows):
        s=row['series'];keep=np.asarray(s['valid']);t=np.asarray(s['times'])[keep];y=np.asarray(s['variance'])[keep]
        t=t-t[0];A=np.column_stack((np.ones(len(t)),t));signal=y-A@np.linalg.lstsq(A,y,rcond=None)[0]
        window=t<=.04;axes[i,0].plot(t[window]*1000,y[window],'.-',ms=2,lw=1)
        axes[i,0].set_xlabel('160ms历史窗起点之后 / ms（展示前40ms）');axes[i,0].set_ylabel('空间方差 / px²')
        label='侧视' if i==0 else '清晰四桨'
        axes[i,0].set_title(f"{label}，截止{row['t']:.6f}s")
        grid=np.arange(60.,600.,3.125);power=lombscargle(t,signal,2*np.pi*grid,normalize='normalize',floating_mean=True)
        axes[i,1].plot(grid,power,label='空间方差周期谱')
        if i==1:
            for j,f in enumerate((303.253372,230.210526,256.483368,284.432022)):
                axes[i,1].axvline(f,ls='--',lw=.8,color='#b97a25',label='同帧四桨SDK频率' if j==0 else None)
        axes[i,1].set_xlabel('事件周期分量 / Hz（未验证机械RPM）');axes[i,1].set_ylabel('归一化谱功率');axes[i,1].legend(fontsize=9)
    fig.savefig(out/'moment_signal.png');plt.close(fig)
    print(str(out/'moment_signal.png'))


if __name__=='__main__':main()
