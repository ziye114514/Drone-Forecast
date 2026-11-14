import csv,json
import cv2
from study31 import ROOT,OUT,CASES,load,save,verify


def fmt(x):return '—' if x is None else f'{x:.2f}'


def main():
    verify();a=load(OUT/'analysis.json');summary=[]
    for case in CASES:
        complete=load(OUT/case/'completion.json');rows=list(map(json.loads,(OUT/case/'observations_forecasts.jsonl').read_text(encoding='utf-8').splitlines()))
        assert len(rows)==complete['frames'];cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(720,1280);count+=1
        cap.release();assert count==len(rows)
        with (OUT/case/'primary_forecasts.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['t','current_detection','forecast_valid','physics_active','x_.1','y_.1','x_.2','y_.2','x_.4','y_.4'])
            for r in rows:w.writerow([r['t'],r['observation']['detected'],r['forecasting']['valid'],r['forecasting']['feature_diagnostic']['active'],*[v for h in ['0.1','0.2','0.4'] for v in (r['primary_prediction'][h] if r['primary_prediction'] else [None,None])]])
        summary.append(dict(case=case,**complete,video_frames_verified=count))
    save(OUT/'runtime_summary.json',summary)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三十一阶段：跨录像检查</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1300px;background:white;margin:auto;padding:32px}h1{font-size:30px}.note{background:#edf6fb;border-left:5px solid #3787a7;padding:18px}.warn{background:#fff4dd;border-color:#b88726}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;white-space:nowrap;text-align:left;border-bottom:1px solid #dae4ee}th{background:#eef3f7}.scroll{overflow:auto}video{max-width:100%}a{color:#176c99}</style><main><h1>第三十一阶段：另一份自采RAW的完整预测检查</h1><p>2026-10-18 · train3 aligned 25–31s · 未修改阶段30参数</p><div class="note">直接运行只输入RAW的原入口，包含逐ID频率、姿态、有效推力、在线反馈和15个对照。所有预测保存后才用train3存档近似框中心评分，未使用未来检测输出。此段在阶段24曾作20ms定性/定位检查，故是跨录像检查，不称全球盲测。</div><h2>1. 视频与运行</h2>'''
    for s in summary:
        c=s['case'];text+=f'<video controls preload="metadata" src="{c}/forecasts.mp4"></video><p>时刻{s["frames"]}，当前发布{s["detected"]}，可预报{s["valid_forecasts"]}，物理生效{s["physics_active_ticks"]}；处理median/P95 {s["processing_median_ms"]:.2f}/{s["processing_p95_ms"]:.2f}ms，含读写{s["wall_including_decode_preview_s"]:.2f}s。100fps为源时间轴，不代表实时。</p><p><a href="{c}/observations_forecasts.jsonl">完整预测JSON</a> · <a href="{c}/primary_forecasts.csv">主预测CSV</a></p>'
    text+='<h2>2. 全部未来框端点</h2><p>每个名义跨度从目标时间减H之前最近锚点预报，实际再长0–25ms，参考整机粗框中心。必须同时查看有效/有机目标、最大误差，不能拿少数容易目标的均值称泛化成功。</p><div class="scroll"><table><tr><th>模型</th><th>名义s</th><th>有效/有机目标</th><th>均值FDEpx</th><th>中位px</th><th>最大px</th><th>联合通过</th></tr>'
    for r in a['table']:text+=f'<tr><td>{r["model"]}</td><td>{r["nominal_h"]}</td><td>{r["valid_forecasts"]}/{r["present_targets"]}</td><td>{fmt(r["mean_fde_px"])}</td><td>{fmt(r["median_fde_px"])}</td><td>{fmt(r["max_fde_px"])}</td><td>{r["joint_success"]}</td></tr>'
    text+='</table></div><div class="note warn">低覆盖或很少物理生效时，主要问题仍是当前桨盘/整机定位缺失，在线调参数不能凭空补出没有的观测。未给模型额外人工框，也未按本段得分调参。所有模型有效性相同，但有效样本少时误差均值不支持强结论。不存在真实姿态/RPM/风速真值。</div><p><a href="protocol.json">冻结协议</a>、<a href="analysis.json">逐目标评分</a>、<a href="runtime_summary.json">运行/视频核验</a>、<a href="阶段结论.md">结论</a>。无G/yolov10-main访问，原默认保持。</p></main></html>'
    (OUT/'第三十一阶段跨录像物理预测报告.html').write_text(text,encoding='utf-8')
    key=[r for r in a['table'] if r['nominal_h']==.4 and r['model'] in ['cv','trusted_acc','adaptive_axis_freq','adaptive_history_mix','adaptive_all_mix']]
    (OUT/'阶段结论.md').write_text('# 阶段31跨录像结果\n\n冻结原入口、另一份train3 aligned RAW25–31s。没有结果后调参，不是全球盲测。\n\n运行：\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n0.4秒全部有效目标对照：\n'+json.dumps(key,ensure_ascii=False,indent=2)+'\n\n所有预测先保存再读存档近似整机框，不能把少量有效目标的均值写成全片段准确率。无独立RPM/姿态真值；下一步优先提高小目标/部分桨盘的当前定位覆盖，再谈动力学泛化。\n',encoding='utf-8');print('Stage31 report rendered and all video frames verified',flush=True)


if __name__=='__main__':main()
