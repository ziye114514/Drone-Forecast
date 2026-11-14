"""Publish stage159 actual results and unresolved limits, including negatives."""
from pathlib import Path
import ast,hashlib,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from prepare_fred_transfer import parse
from score_fred_transfer import reference


def read(p):return json.loads(Path(p).read_text('utf-8'))
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]


def main(output):
    out=Path(output);previous=ROOT.parent/'output/rpm_recall_pilot_20261031'
    specs={'FRED0':('FRED0_clean_scores.json','RAW_FRED0_acquire4s'),
           'FRED7':('FRED7_new33_scores.json','RAW_FRED7_new33_4s'),
           'FRED9':('FRED9_budgeted_scores.json','RAW_FRED9_budgeted4s')}
    table=[];summary={};n=good=0
    for name,(score,folder) in specs.items():
        s=read(out/score);r=read(out/folder/'summary.json');p=read(out/folder/'protocol.json')
        count=round(s['IoU50_fraction_including_misses']*s['annotation_source_ticks']);n+=s['annotation_source_ticks'];good+=count
        summary[name]=dict(scores={k:v for k,v in s.items() if k not in ['anchors','targets']},runtime=r,protocol=p)
        table.append(f"<tr><td>{name}</td><td>{count}/{s['annotation_source_ticks']} = {100*s['IoU50_fraction_including_misses']:.1f}%</td><td>{s['current_outputs']}/{s['annotation_source_ticks']}</td><td>{s['error_04s_px']['mean']:.2f} / {s['error_04s_px']['p95']:.2f}px</td><td>{s['predicted_target_slots']}/{s['target_slots']}</td><td>{r['publication']['valid_outputs']}/{r['publication']['outputs']}</td></tr>")
    summary['pooled_positive_IoU50']=dict(qualified=good,eligible=n,fraction=good/n)
    wall=[]
    for name,filename in [('FRED7','FRED7_new_common_publication.json'),('FRED9','FRED9_budgeted_common_publication.json')]:
        d=read(out/filename);summary[name]['common_wall_clock']=d;m=d['matched']
        wall.append(f"<tr><td>{name}</td><td>{m['n']}</td><td>{m['control_error_px']['mean']:.2f} → {m['candidate_error_px']['mean']:.2f}px</td><td>{m['control_error_px']['p95']:.2f} → {m['candidate_error_px']['p95']:.2f}px</td><td>{d['statistics']['control']['valid']} → {d['statistics']['candidate']['valid']}</td></tr>")
    summary['new_training']=read(out/'native33_adaptation/export.json');summary['training_protocol']=read(out/'native33_adaptation/protocol.json')
    summary['negative_control']=read(out/'new33_scores.json')['empty_self5'];summary['geometry_budget']=read(out/'geometry_budget_checks.json')
    summary['limits']=['Per-clip90percent not universally achieved: FRED9 is106/118, one qualified box short',
        'Source-time FDE differs from actual-publication FDE','60-frame empty control has6 detector candidate frames',
        'Publications30Hz do not imply30 valid observations; latency tails remain','Not a blind or full official FRED benchmark',
        'Original truth labels unchanged despite visible misalignment examples']
    (out/'final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    # Same source-time current ON/OFF image and reference crop for both arms.
    a=rows(previous/'RAW_FRED9_recall_final4s/predictions.jsonl');b=rows(out/'RAW_FRED9_budgeted4s/predictions.jsonl')
    labels=parse((ROOT.parent/'output/stability_audit_20261031/holdout9/input/coordinates.txt').read_text('utf-8-sig'))
    video=cv2.VideoWriter(str(out/'FRED9_prediction_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(960,352))
    for k,(old,new) in enumerate(zip(a,b)):
        image=np.load(previous/'frames/FRED9'/f'{k:04d}.npz')['image'];gt=reference(labels,new['t']);future=reference(labels,new['t']+.4)
        center=(gt[:2]+gt[2:])/2 if gt is not None else np.array([640,360]);lo=np.clip((center-[150,110]).astype(int),[0,0],[980,500]);panels=[]
        for r,title in [(old,'Previous'),(new,'Native33 + joint state')]:
            im=image.copy()
            if gt is not None:cv2.rectangle(im,tuple(np.rint(gt[:2]).astype(int)),tuple(np.rint(gt[2:]).astype(int)),(0,255,0),1)
            if r['body'].get('detected'):
                box=np.rint(r['body']['bbox']).astype(int);cv2.rectangle(im,tuple(box[:2]),tuple(box[2:]),(255,0,255),1)
            if r['valid'] and r['physical']:
                points=[r['coefficient']['position']]+[r['physical'][str(h)] for h in [.1,.2,.4]]
                cv2.polylines(im,[np.rint(points).astype(np.int32)],False,(0,220,255),1)
                cv2.drawMarker(im,tuple(np.rint(points[-1]).astype(int)),(0,220,255),cv2.MARKER_TILTED_CROSS,8,1)
            if future is not None:cv2.drawMarker(im,tuple(np.rint((future[:2]+future[2:])/2).astype(int)),(0,255,0),cv2.MARKER_CROSS,9,1)
            crop=im[lo[1]:lo[1]+220,lo[0]:lo[0]+300];crop=cv2.resize(crop,(480,352),interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop,f'{title} {r["t"]:.3f}s',(8,21),0,.5,(0,220,255),1);panels.append(crop)
        frame=np.hstack(panels);video.write(frame)
        if k==72:cv2.imwrite(str(out/'prediction_preview.png'),frame)
    video.release()
    snapshot=out/'final_source_snapshot';snapshot.mkdir(exist_ok=True);manifest={}
    for p in list(ROOT.glob('*.py'))+list((ROOT/'tools').glob('*native33*.py'))+[ROOT/'tools/report_recall90_motion.py',ROOT/'tools/probe_singer_state.py',ROOT/'tools/probe_recall90_models.py',ROOT/'tools/calibrate_small_box_size.py']:
        relative=p.relative_to(ROOT);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);data=p.read_bytes();ast.parse(data.decode('utf-8-sig'));dest.write_bytes(data);manifest[relative.as_posix()]=hashlib.sha256(data).hexdigest()
    (snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2),'utf-8')
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>90%合格框与预测兑现</title><style>body{{font:17px/1.7 system-ui;max-width:1120px;margin:30px auto;color:#23364a;padding:0 24px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border:1px solid #ccd6df;padding:10px;text-align:left}}img,video{{max-width:100%}}pre,.note{{background:#edf3f7;padding:16px;white-space:pre-wrap}}.warning{{background:#fff0d8;padding:14px}}</style><h1>90%合格框与预测能力兑现</h1><p>2026-10-31 · 阶段159 · 小目标专用候选，原五圆／侧视入口保留。</p>
    <p class="note">三段合计{good}/{n}，即{100*good/n:.2f}%合格框。FRED0/7超过90%，FRED9为106/118=89.83%，仍差一个合格框，不能说所有场景均达到90%。预测已有实际输出时间上的改善，持续有效30Hz和25px仍未全面完成。</p>
    <h2>完整RAW结果</h2><table><tr><th>片段</th><th>IoU≥0.5／有真值时刻</th><th>当前观测覆盖</th><th>源0.4s均值／P95</th><th>源预测覆盖</th><th>有效发布／全部心跳</th></tr>{''.join(table)}</table><p>漏失计入分母；当前观测包括当前图像光流续接，非纯网络F1或全景AP。0/7/9使用同一新模型及共享状态；0/7未另跑几何降频，但其源几何均无有效结构，不用跨配置相邻FPS差冒充纯提速。全部原标签保留。</p>
    <h2>与上一阶段相同实际查询和目标</h2><table><tr><th>片段</th><th>共同查询</th><th>平均误差</th><th>P95</th><th>可用查询数</th></tr>{''.join(wall)}</table><p>只用查询前已计算完成的状态，目标为实际查询＋0.4秒。源39px不等于实际发布39px。FRED9共同均值62.73→57.04px，说明本轮改善已部分兑现，仍有长尾。<a href="final_summary.json">完整机器可读结果</a></p>
    <h2>改动对应了哪些损失</h2><ol><li>复用已有FRED小目标模型，再固定5轮匹配33.333ms原生事件窗口。训练3/4/5、验证6，保留原自采1/4弱标签；评分0/7/8/9/10不参与选权重。单模型单次NN，旧模型不覆盖。</li><li>失锁时直接测当前帧，修复“网络已经看到目标，但延迟框因小目标光流不足而被丢弃”的获取循环。</li><li>单一Singer共享状态联合估计p/v/未知净加速度，继续同一力学积分；不是额外并行轨迹，也不把净加速度叫作桨叶推力。原转速、五圆和二力分工保留。</li><li>当前NN关联门拒绝约400px的目标身份跳变，输出仍需当前图像证据。FRED10两帧错配曾产生1514/1012px预测；门控后源均值174.13→47.29px，相同16个实际查询91.63→59.02px，当前119/120。</li><li>仅在9次更新都无任何有效结构、目标小且没有模型时，以10Hz重新搜索几何；NN、运动和所有原生周期输入保留30Hz。FRED9跳过74次无效重复拟合，120个当前框和所有源预测完全相同；处理能力30.83→35.12FPS，滞后P95147.37→91.26ms，仍不能叫持续有效30Hz。</li></ol>
    <h2>预测视频</h2><p>紫框：当前定位；绿框／绿十字：当前位置／未来0.4s真值；黄色：预测。视口用真值居中仅用于显示，预测不读取真值。此视频按源时刻展示，实际发布分数见上表。</p><video controls src="FRED9_prediction_comparison.mp4" poster="prediction_preview.png"></video>
    <h2>限制与未采用路径</h2><p class="warning">空场5号60个完整窗口中，新模型出现6帧检测候选；这是待解决的误报风险，不发布“全面90%”或全场景默认升级。新模型与配置只作为小目标候选，原realtime30仍保留。FRED8旧小目标模型配置已达90%，FRED10门控配置84.2%，不能把不同模型配置混成一个统一90%结果。</p><p>FRED9部分框偏离可见事件主体；同时间戳官方PNG也存在该现象，不是本项目RAW渲染独有。原因尚未完全查清，没有改标签、删除难例或给模型喂真值偏移。</p><details><summary>官方帧与本机RAW同时间戳核验</summary><img src="label_alignment/official_vs_raw.png"></details><p>ByteTrack现成适配在FRED10仅保留39/120当前输出，未采用。20ms验证帧拟合的尺寸比例不能直接迁移到33ms，分数变差，未部署。所有负结果保留，未继续扫参数。上游预训练可能覆盖整个FRED，这不是全系统盲测或官方排行榜。</p>
    <h2>运行小目标候选</h2><pre>python run_fast_structure.py --raw "events.raw" --output "新的目录" --start 32 --duration 4 --sdk-avx2 --structure-guided-nn --recall90 --detector-model "models/small_target_native33_v1.onnx" --detector-sha256 fc228a1b355a3b8396c05df63f7eb8beaf192bca758bf5089898e16e893a4255</pre><p>recall90是目标名称，不是准确率认证；组合当前获取、完整包NN、共享Singer状态、身份门控和合格后降频。可显式用--unresolved-geometry-stride1保留完整几何频率。原--realtime30／--legacy-realtime30仍可运行；大目标自采侧视继续优先原入口。</p><p>参考：<a href="https://github.com/FoundationVision/ByteTrack">ByteTrack官方</a>、<a href="https://stonesoup.readthedocs.io/en/latest/stonesoup.models.transition.html">Stone Soup Singer</a>、<a href="https://filterpy.readthedocs.io/en/latest/common/common.html">FilterPy Van Loan</a>。没有复制新的第三方实现，未改保护目录、RAW、原模型或旧发布ZIP，未上传GitHub。</p></html>'''
    (out/'index.html').write_text(text,'utf-8');print(json.dumps(summary['pooled_positive_IoU50']))


if __name__=='__main__':main(sys.argv[1])
