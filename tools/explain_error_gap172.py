"""Post-inference error attribution. No inference, fitting or GPU experiment."""
from pathlib import Path
import sys,json,bisect
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from compare_final_pipeline import reference_at
from physical_publication import evaluate
from adaptive_force_memory import kernel
OUT=ROOT.parent/'output/motion172_20261107'

def load(p):return json.loads(p.read_text('utf-8'))
def reference_box(truth,t):
    i=bisect.bisect_left([r['t'] for r in truth],t)
    if i<len(truth) and abs(truth[i]['t']-t)<1e-7:return np.array(truth[i]['box'],float)
    if not 0<i<len(truth):return None
    a,b=truth[i-1:i+1]
    if b['t']-a['t']>.1500001 or any(r['state']!='present' or not r['box'] for r in (a,b)):return None
    f=(t-a['t'])/(b['t']-a['t']);return (1-f)*np.array(a['box'])+f*np.array(b['box'])

if __name__=='__main__':
    s=load(OUT/'final_summary.json');result=dict(scope='post-run historical development-reference diagnostic; labels not runtime inputs',
        decomposition='prediction-target = (estimated_position-current_reference) + h*(estimated_velocity-true_future_average_velocity) + acceleration_rollout',
        scalar_norms_do_not_add=True,normalization='each query error divided by current manual box diagonal; then mean',cases={})
    for case,source,labels,scored in [('side','final172/side_fixed.jsonl','dense_labels_side_110_1195',s['side']['fixed']['combined']),
        ('train2','final_regional172/train2_fixed.jsonl','dense_labels_train2_40_46',s['train2']['fixed'])]:
        rows=[json.loads(r) for r in (ROOT.parent/'output/geometry170_20261106'/source).read_text('utf-8').splitlines()]
        truth=sorted(load(ROOT.parent/'output'/labels/'labels.json')['records'].values(),key=lambda r:r['t']);records=[]
        for z in scored['records']:
            if not z['valid']:continue
            q=z['t'];r=next(r for r in rows if r['t']==z['source_t']);p=np.array(r['coefficient']['position']);v=np.array(r['coefficient']['velocity'])
            h=q+.4-r['t'];now=reference_at(truth,r['t']);future=np.array(z['target']);avg=(future-now)/h
            pred=np.array(evaluate(r,[h])[str(float(h))]);acc=pred-p-h*v;e0=p-now;ev=h*(v-avg)
            assert abs(np.linalg.norm(e0+ev+acc)-z['error_px'])<1e-6
            box=reference_box(truth,q);diag=float(np.linalg.norm(box[2:]-box[:2]));past=reference_at(truth,r['t']-.1)
            known=np.zeros(2) if not r['force'].get('active') else float(kernel(-1/.35,h))*r['force']['effective_gain_px_s2']*np.array(r['force']['input_u'])
            records.append(dict(t=q,source_t=r['t'],h=h,error_px=z['error_px'],position_error_px=float(np.linalg.norm(e0)),
                current_velocity_px_s=v.tolist(),past_100ms_reference_average_velocity_px_s=None if past is None else ((now-past)/.1).tolist(),
                true_future_average_velocity_px_s=avg.tolist(),position_error_xy=e0.tolist(),velocity_error_displacement_xy=ev.tolist(),
                acceleration_rollout_xy=acc.tolist(),acceleration_rollout_norm_px=float(np.linalg.norm(acc)),
                position_velocity_only_error_px=float(np.linalg.norm(p+h*v-future)),box_diagonal_px=diag,error_over_box_diagonal=z['error_px']/diag,
                force_active=r['force']['active'],gain_ever_learned=r['force'].get('gain_ever_learned'),
                current_known_term_rollout_norm_px=float(np.linalg.norm(known)),
                known_term_decomposition_not_counterfactual_ablation=True,
                mode=r['structure']['mode'],view=r['structure'].get('view_evidence')))
        result['cases'][case]=dict(summary=dict(n=len(records),mean_error_px=float(np.mean([r['error_px'] for r in records])),
            mean_position_error_px=float(np.mean([r['position_error_px'] for r in records])),
            mean_box_diagonal_px=float(np.mean([r['box_diagonal_px'] for r in records])),
            mean_error_over_box_diagonal=float(np.mean([r['error_over_box_diagonal'] for r in records])),
            mean_position_velocity_only_error_px=float(np.mean([r['position_velocity_only_error_px'] for r in records])),
            max_error_px=max(r['error_px'] for r in records)),records=records)
        if case=='train2':plot_rows=rows;plot_truth=truth
    (OUT/'error_gap_breakdown.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),'utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(1,3,figsize=(13,4.6),constrained_layout=True)
    for ax,q in zip(axs,[43.6,43.8,44.2]):
        z=min(result['cases']['train2']['records'],key=lambda z:abs(z['t']-q));r=next(r for r in plot_rows if r['t']==z['source_t'])
        origin=reference_at(plot_truth,r['t']);past=np.array([reference_at(plot_truth,r['t']+h) for h in np.linspace(-.2,0,9)])-origin
        future=np.array([reference_at(plot_truth,r['t']+h) for h in np.linspace(0,z['h'],17)])-origin
        hs=np.linspace(0,z['h'],17);pred=evaluate(r,hs);pred=np.array([pred[str(float(h))] for h in hs])-origin
        ax.plot(past[:,0],past[:,1],color='#5379b8',label='过去0.2s参考')
        ax.plot(future[:,0],future[:,1],color='#188650',lw=2,label='随后0.4s参考')
        ax.plot(pred[:,0],pred[:,1],'--',color='#d87418',lw=2,label='模型外推')
        ax.plot(0,0,'o',color='#188650');ax.plot(future[-1,0],future[-1,1],'*',color='#188650',ms=10)
        ax.plot(pred[-1,0],pred[-1,1],'x',color='#d87418',ms=8)
        ax.plot([future[-1,0],pred[-1,0]],[future[-1,1],pred[-1,1]],':',color='#bb4d4d',label='0.4s误差')
        ax.set_title(f'{q:.1f}s起点：误差{z["error_px"]:.1f}px');ax.set_aspect('equal',adjustable='datalim');ax.invert_yaxis()
        ax.set_xlabel('相对当前真实参考的 x / px');ax.set_ylabel('y / px（图像向下）');ax.grid(alpha=.2)
    axs[0].legend(fontsize=8,loc='best')
    fig.suptitle('同一录像也会时好时坏：转弯、暂时吻合、旧加速度过度延续\n人工框轨迹仅用于事后分析，没有输入预测器',fontsize=12)
    fig.savefig(OUT/'why_error_spikes.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v['summary'] for k,v in result['cases'].items()},ensure_ascii=False))
