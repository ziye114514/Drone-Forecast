"""Offline visualization of frozen predictions; never called by inference."""
from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from compare_final_pipeline import reference_at


def main(run,label,video=False):
    path=Path(run);rows=[json.loads(s) for s in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
    truth=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    a=json.loads((path/'shared_motion_accuracy.json').read_text('utf-8'));scores=a['scores']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(3,1,figsize=(10,11),constrained_layout=True)
    manual=[r for r in truth if rows[0]['t']<=r['t']<=rows[-1]['t'] and r.get('state')=='present' and r.get('box')]
    points=np.array([(np.asarray(r['box'][:2])+r['box'][2:])/2 for r in manual])
    positions=np.array([r['coefficient']['position'] if r['valid'] and r['body'].get('detected') else [np.nan,np.nan] for r in rows])
    ax=axes[0];ax.plot(points[:,0],points[:,1],'.',color='#327d3d',label='人工参考（开发片段）')
    ax.plot(positions[:,0],positions[:,1],color='#7a8090',alpha=.6,label='当前位置状态')
    for j,t in enumerate([110.43,112.23,113.23,116.465] if rows[-1]['t']>116.5 else [110.43,111.03,112.23,113.23]):
        r=min(rows,key=lambda r:abs(r['t']-t))
        if not r['valid']:continue
        p=np.asarray(r['coefficient']['position']);q=np.asarray(r['physical']['0.4']);target=reference_at(truth,r['t']+.4)
        ax.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='->',color='#d87a19',lw=1.8))
        ax.scatter(*q,color='#d87a19',s=20,label='0.4秒预测' if j==0 else None)
        if target is not None:ax.scatter(*target,color='#a12a46',marker='x',s=40,label='对应未来参考（仅离线显示）' if j==0 else None)
    ax.invert_yaxis();ax.set(xlabel='原生画面 x / px',ylabel='原生画面 y / px',title='当前轨迹和若干0.4秒预测；参考从未进入预测器')
    ax.legend(loc='best',fontsize=8,ncol=2);ax.grid(alpha=.2)
    ax=axes[1]
    for name,label_,color in [('same_input_no_motion','同输入二力控制／未校正速度','#7a8090'),('RAW_current_motion','本轮候选','#146d9a')]:
        data=scores[name]['future04_detail'];times=[];errors=[]
        for j,r in enumerate(data):
            if j and r['t']-data[j-1]['t']>.25:times.append(np.nan);errors.append(np.nan)
            times.append(r['t']);errors.append(r['error_px'])
        ax.plot(times,errors,'.-',label=label_,color=color)
    ax.axhline(50,color='#bd3c3c',ls='--',label='50px目标')
    stat=scores['RAW_current_motion']['physical_future_error_px']['0.4'];fraction=scores['RAW_current_motion']['within50_fraction']
    ax.set(xlabel='预测起点的源时间 / s',ylabel='0.4秒欧氏误差 / px',title=f'同{stat["n"]}个目标：均值 {stat["mean"]:.2f}px，P95 {stat["p95"]:.2f}px；50px内{fraction:.1%}')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[2];ax.plot([r['t'] for r in rows],[r['cost']['completion_lag_ms'] for r in rows],color='#6e4d94')
    ax.axhline(1000/30,color='#bd3c3c',ls='--',label='一个30Hz源周期')
    summary=json.loads((path/'summary.json').read_text('utf-8'))
    extra=path/'extension_summary.json'
    title=f'处理能力 {summary["steady_capacity_fps"]:.2f}FPS，当前观测 {summary["paced_fresh_observation_rate"]:.2f}Hz；初期仍积压'
    if extra.exists():
        extension=json.loads(extra.read_text('utf-8'));capacity=extension['scopes']['fixed110_114']['capacity_fps']
        title=f'原4秒有目标段处理 {capacity:.2f}FPS；全段当前观测 {summary["paced_fresh_observation_rate"]:.2f}Hz（含出画）'
    ax.set(xlabel='源时间 / s',ylabel='状态完成延迟 / ms',title=title)
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('侧视完整RAW小试：源时刻、输出资格和评分范围保留',fontsize=14)
    fig.savefig(path/'prediction_and_latency.png',dpi=160);plt.close(fig)
    if video:
        import cv2
        from live_decode import native_batches
        from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
        from metavision_sdk_core import BaseFrameGenerationAlgorithm
        protocol=json.loads((path/'protocol.json').read_text('utf-8'));raw=Path(protocol['raw']);before=raw.stat()
        noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer();frame=np.empty((720,1280,3),np.uint8)
        writer=cv2.VideoWriter(str(path/'prediction_preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
        if not writer.isOpened():raise RuntimeError('Cannot create preview')
        n=0
        try:
            for (t,e),r in zip(native_batches(raw,protocol['actual_iterator_start'],protocol['actual_duration'],False,protocol['native_step_us']),rows):
                assert abs(t-r['t'])<1e-8 and len(e)==r['raw_events']
                noise.process_events(e,buffer);clean=buffer.numpy();clean=clean[clean['t']>=round(t*1e6)-20000]
                BaseFrameGenerationAlgorithm.generate_frame(clean,frame)
                body=r['body'];color=(30,210,50) if not body.get('low_confidence') else (160,160,160)
                if body.get('detected'):
                    b=np.rint(body['bbox']).astype(int);cv2.rectangle(frame,tuple(b[:2]),tuple(b[2:]),color,2)
                if r['valid']:
                    trace=np.rint([r['coefficient']['position']]+[r['physical'][str(h)] for h in [.1,.2,.4]]).astype(np.int32)
                    cv2.polylines(frame,[trace],False,(0,200,255),2);cv2.circle(frame,tuple(trace[-1]),5,(0,200,255),2)
                    ref=reference_at(truth,t+.4)
                    if ref is not None:cv2.drawMarker(frame,tuple(np.rint(ref).astype(int)),(80,70,240),cv2.MARKER_CROSS,16,2)
                axis=r['structure'].get('axis_xy')
                if axis is not None and body.get('detected'):
                    p=np.asarray(body['center']);cv2.arrowedLine(frame,tuple(np.rint(p).astype(int)),tuple(np.rint(p+70*np.asarray(axis)).astype(int)),(230,130,20),2)
                for y,text in [(25,f'{t:.3f}s  mode={r["structure"]["mode"]}  prediction={r["valid"]}'),(50,'yellow=forecast  red=future manual reference (OFFLINE ONLY)'),(75,'Native ON/OFF; video playback30Hz, performance measured separately')]:
                    cv2.putText(frame,text,(15,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,220,40),1,cv2.LINE_AA)
                writer.write(frame);n+=1
        finally:writer.release()
        assert n==len(rows) and raw.stat().st_size==before.st_size and raw.stat().st_mtime_ns==before.st_mtime_ns
        (path/'preview_scope.json').write_text(json.dumps(dict(frames=n,frozen_predictions=True,extra_inference=False,
            reference_only_for_offline_render=True,playback_FPS_is_not_processing_FPS=True,RAW_unchanged=True),indent=2),'utf-8')
    print(json.dumps(dict(image=str(path/'prediction_and_latency.png'),video=video)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('labels');p.add_argument('--video',action='store_true')
    a=p.parse_args();main(a.run,a.labels,a.video)
