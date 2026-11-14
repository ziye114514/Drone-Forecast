"""Mechanism and exact native map checks, with no coordinate labels."""
from pathlib import Path
import sys,json,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from isolated_rim_visibility import recover_isolated_rims
from research.stage146.branches146 import three_affine_hypotheses
from run_rotor_only import serial
OUT=ROOT.parent/'output/isolated177_20261107'


def seed(cx=65.,cy=50.,major=12.,minor=8.):
    return dict(cx=cx,cy=cy,major=major,minor=minor,angle=0.,ratio=minor/major,
        mode='ellipse',current_observation=True,fov_clipped=False,
        center_sigma_px=1.,axis_angle_sigma_deg=5.,geometry_quality=.6,
        inferred_shape=False)


def ellipse(ax=22,ay=10,value=2):
    a=np.zeros((100,130),np.float32)
    cv2.ellipse(a,(65,50),(ax,ay),0,0,360,value,-1)
    cv2.ellipse(a,(65,50),(6,4),0,0,360,16,-1)
    return a


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);cv2.setNumThreads(2)
    a=ellipse();b=a.copy();r0=seed();rs,au=recover_isolated_rims(a,b,[r0],np.zeros(2))
    assert au['accepted']==1 and len(rs)==1 and rs[0]['major']>30
    assert r0==seed() and np.array_equal(a,b)
    assert rs[0]['activity_core_center_xy']==[65.,50.] and not rs[0]['outline_is_periodicity_measurement']
    cov=np.asarray(rs[0]['center_covariance_px2']);assert np.allclose(cov,np.eye(2)*rs[0]['center_sigma_px']**2)
    empty,ea=recover_isolated_rims(a,b,[],np.zeros(2));assert not empty and ea['accepted']==0
    pair,pa=recover_isolated_rims(a,b,[r0,seed(70,50)],np.zeros(2));assert len(pair)==2 and pa['accepted']==0
    flat=ellipse(22,2);_,fa=recover_isolated_rims(flat,flat,[r0],np.zeros(2));assert fa['accepted']==0
    wrong=ellipse();wrong[:]=0;_,wa=recover_isolated_rims(wrong,wrong,[r0],np.zeros(2));assert wa['accepted']==0
    tests=dict(core_expansion=True,no_unseeded_disk=True,no_multi_core_collapse=True,
        flat_not_replaced=True,no_empty_input_detection=True,covariance_matches_new_fit=True,
        input_not_mutated=True,outline_not_period_measurement=True)
    D=ROOT.parent/'output/rim174_20261107'
    source=json.loads((D/'supported_rim_recovery.json').read_text('utf-8'))['records']
    records=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,2,figsize=(13,8))
    for row,rec in enumerate(source):
        t=round(rec['t'],1);z=np.load(D/f'{t:.1f}.npz');x,y,X,Y=rec['roi'];k=z['geometry_native'][y:Y,x:X]
        on=np.where(k,z['on'][y:Y,x:X],0);off=np.where(k,z['off'][y:Y,x:X],0)
        old=rec['new'];new,audit=recover_isolated_rims(on,off,old,[x,y])
        metric=three_affine_hypotheses([dict(v,mode='ellipse',inferred_shape=False) for v in new])
        records.append(dict(t=rec['t'],roi=rec['roi'],old=old,new=new,audit=audit,three=metric))
        for col,rotors in enumerate([old,new]):
            ax=axes[row,col];ax.imshow(np.log1p(np.minimum(on,off)),cmap='gray',origin='upper')
            for v in rotors:
                refit=v.get('native_rim_refit',False);color='#00e0ff' if refit else '#ffbd50'
                ax.add_patch(Ellipse((v['cx']-x,v['cy']-y),v['major'],v['minor'],angle=v['angle'],fill=False,color=color,lw=1.6))
                ax.plot(v['cx']-x,v['cy']-y,'+',color=color)
                ax.text(v['cx']-x,v['cy']-y-12,f"{v['major']:.1f}px",color=color,fontsize=8)
            ax.set_title(f"{t:.1f}s · "+('174双弧候选' if col==0 else f"177孤立轮廓补入口，接受{audit['accepted']}个"),fontsize=10)
            ax.set_xlim(0,X-x);ax.set_ylim(Y-y,0)
    fig.suptitle('原生20ms ON/OFF：黄色为已有候选，青色为孤立原轮廓重拟合\n候选轮廓恢复不等于已确认桨盘，也不等于预测提升',fontsize=12)
    fig.tight_layout(h_pad=3);fig.savefig(OUT/'native_outlines.png',dpi=150);plt.close(fig)
    (OUT/'mechanism.json').write_text(json.dumps(dict(tests=tests,records=records,labels_used=False),default=serial,indent=2),'utf-8')
    print(json.dumps(dict(tests=tests,frames=[dict(t=r['t'],accepted=r['audit']['accepted'],three=r['three']['reason']) for r in records])))
