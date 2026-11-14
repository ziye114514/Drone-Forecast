"""Check translational smear with the existing past-only event warp.

No velocity search, labels, new period detector or forecast rerun. Geometric
counts only; the source native gate is applied before moving event locations.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools'),str(ROOT/'src')]
from native_event_alignment import past_warp
from pure_visibility170 import window
from fast_structure_front import count_pair
from curvature_visibility import recover,masks,contours
from isolated_rim_visibility import recover_isolated_rims
from fast_geometry_compute import fit_rotor_geometry_fast
from research.stage146.branches146 import three_affine_hypotheses
from run_rotor_only import serial
OUT=ROOT.parent/'output/alignment178_20261107'


def candidates(a,b,offset):
    dense,_=masks(a,b);rotors=[]
    for c in contours(dense)[:6]:
        g=fit_rotor_geometry_fast(c,max_major=min(500.,1.5*max(a.shape)),allow_arc=False)
        if g is not None:
            p=np.asarray(g.pop('center'))+offset
            rotors.append(dict(g,cx=float(p[0]),cy=float(p[1]),current_observation=True,fov_clipped=False))
    if rotors:
        major=max(r['major'] for r in rotors);rotors=[r for r in rotors if r['major']>=major/2.5]
    rotors,split=recover(a,b,rotors,offset,reconsider_small_cores=True)
    rotors,isolated=recover_isolated_rims(a,b,rotors,offset)
    hyp=three_affine_hypotheses([dict(v,mode='ellipse',inferred_shape=False) for v in rotors])
    return rotors,dict(split=split,isolated=isolated,three=hyp)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);cv2.setNumThreads(2)
    D=ROOT.parent/'output/rim174_20261107';records=json.loads((D/'results.json').read_text('utf-8'))
    rows=[json.loads(s) for s in (ROOT.parent/'output/layout176_20261107/integrated/train2_fixed.jsonl').read_text('utf-8').splitlines()]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,2,figsize=(13,9));out=[]
    for i,rec in enumerate(records):
        t=rec['t'];q=round(t,1);z=np.load(D/f'{q:.1f}.npz');keep=z['geometry_native'];bounds=np.array(rec['roi']);x,y,X,Y=bounds
        old=max((r for r in rows if r['t']<=t-.02),key=lambda r:r['t']);m=old['motion'];shift=m.get('estimate')
        if shift is None or shift.get('peak_at_search_boundary') or not old['body'].get('detected'):
            raise RuntimeError('No qualified past native velocity at '+str(t))
        velocity=np.array([shift['dx'],shift['dy']])/m['dt_s']
        e=window(ROOT.parent/'output/native_plane165_20261104/cache_train2_inside',t)
        selected=e[keep[e['y'],e['x']]]
        warped,provenance=past_warp(selected,t,velocity,old['t'],window_us=20000)
        ua,ub=count_pair(selected,bounds,t);wa,wb=count_pair(warped,bounds,t)
        saved=[np.where(keep[y:Y,x:X],z[k][y:Y,x:X],0) for k in ('on','off')]
        control_delta=max(float(np.max(np.abs(a-b))) for a,b in zip(saved,[ua,ub]))
        entries={}
        for j,(arm,a,b) in enumerate([('unwarped',ua,ub),('past_warp',wa,wb)]):
            rs,info=candidates(a,b,bounds[:2]);entries[arm]=dict(rotors=rs,audit=info)
            ax=axes[i,j];ax.imshow(np.log1p(np.minimum(a,b)),cmap='gray')
            for v in rs:
                ax.add_patch(Ellipse((v['cx']-x,v['cy']-y),v['major'],v['minor'],angle=v['angle'],fill=False,color='#00e0ff',lw=1.3))
            ax.set_title(f"{q:.1f}s "+('原始20ms' if j==0 else '仅过去速度对齐')+f"：{len(rs)}个候选",fontsize=10)
            ax.set_xlim(0,X-x);ax.set_ylim(Y-y,0)
        out.append(dict(t=t,previous_velocity_source_t=old['t'],alignment=provenance,
            raw_count_parity_max=control_delta,arms=entries))
        np.savez_compressed(OUT/f'{q:.1f}.npz',unwarped_on=ua,unwarped_off=ub,warped_on=wa,warped_off=wb)
        print(q,velocity,control_delta,[(k,len(v['rotors']),v['audit']['three']['reason']) for k,v in entries.items()])
    fig.suptitle('检查20ms平移拖影：只使用窗口开始前的原生位移速度\n没有按标注或清晰度搜索速度；未重算周期或宣称预测提升')
    fig.tight_layout(h_pad=3);fig.savefig(OUT/'past_motion_alignment.png',dpi=150);plt.close(fig)
    (OUT/'probe.json').write_text(json.dumps(dict(records=out,labels_used=False,forecast_test=False),default=serial,indent=2),'utf-8')
