import csv,json
import cv2,numpy as np
from study30 import ROOT,OUT,CASES,load,save,verify


def main():
    verify();summary=load(OUT/'analysis.json');images=[];video_checks=[]
    for s in summary:
        case=s['case'];rows=list(map(json.loads,(OUT/case/'observations_forecasts.jsonl').read_text(encoding='utf-8').splitlines()))
        with (OUT/case/'primary_forecasts.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['t','current_detection','forecast_valid','physics_active','primary','x_.1','y_.1','x_.2','y_.2','x_.4','y_.4'])
            for r in rows:w.writerow([r['t'],r['observation']['detected'],r['forecasting']['valid'],r['forecasting']['feature_diagnostic']['active'],r['primary'],*[v for h in ['0.1','0.2','0.4'] for v in (r['primary_prediction'][h] if r['primary_prediction'] else [None,None])]])
        cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'));count=0;snap={int(len(rows)*f) for f in [.25,.55,.85]};selected=[]
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(720,1280)
            if count in snap:
                small=cv2.resize(frame,(640,360));canvas=np.full((395,640,3),245,np.uint8);canvas[35:]=small
                cv2.putText(canvas,f'{case} / {rows[count]["t"]:.2f}s',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(50,65,85),1);selected.append(canvas)
            count+=1
        cap.release();assert count==len(rows) and len(selected)==3;images.append(np.hstack(selected));video_checks.append(dict(case=case,decoded_frames=count,shape=[720,1280]))
    cv2.imencode('.png',np.vstack(images))[1].tofile(str(OUT/'prediction_contact.png'));save(OUT/'video_checks.json',video_checks)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三十阶段：RAW姿态转速自适应预测入口</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;margin:0;background:#f3f6fa;color:#243342}main{max-width:1300px;margin:auto;background:white;padding:32px}h1{font-size:30px}h2{margin-top:35px}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}.warn{background:#fff4dd;border-color:#b88726}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dae4ee}th{background:#eef3f7}video,img{max-width:100%;height:auto}pre{background:#eef3f7;padding:18px;overflow:auto}a{color:#176c99}</style><main><h1>第三十阶段：只输入RAW，真正接入姿态、逐桨频率和在线预测</h1><p>2026-10-18 · 新独立研究入口 · 清晰2秒+侧视5秒 · 700时刻真实重解码</p><div class="note"><b>完整路径已能运行并精确回归。</b>RAW原生ON/OFF→桨盘/整机检测→逐ID频率→固定符号法向→相对频率平方推力前向积分→到期误差修正/模型混合→0.1/0.2/0.4s图像位置。700个当前观测、频率及全部15模型预测，与阶段12/29对应缓存逐项一致。</div><p>旧入口阶段24/26频率只作诊断；这个新入口的物理专家实际使用法向与频率，并保留恒速/历史模型对照和自动回退。整合成功不是新的准确率证据：开发集对照见<a href="../stage29_delayed_model_weights_20261018/第二十九阶段延迟在线模型选择报告.html">阶段29报告</a>，侧视只用历史混合仍更好，不能宣称物理稳定增益。</p><h2>1. 直接看预报</h2><p>绿色为当前整机框，蓝线为CV，橙线为默认在线混合，紫圈为单独姿态频率专家的未来端点；均为当时预报，没有未来真值轨迹。画面标明physics是否实际生效；没有当前测量时短时续推，失测160ms后停止新预报。视频100fps只是原始时间轴。</p>'''
    for s in summary:
        case=s['case'];text+=f'<h3>{case}</h3><video controls preload="metadata" src="{case}/forecasts.mp4"></video><p><a href="{case}/observations_forecasts.jsonl">完整15模型JSON</a> · <a href="{case}/primary_forecasts.csv">主预测CSV</a> · <a href="{case}/regression.json">逐项回归核验</a></p>'
    text+='<h2>2. 性能与覆盖</h2><table><tr><th>片段</th><th>当前发布/时刻</th><th>可预报</th><th>物理生效</th><th>处理中位/P95 ms</th><th>含读写秒数</th></tr>'
    for s in summary:text+=f'<tr><td>{s["case"]}</td><td>{s["detected"]}/{s["frames"]}</td><td>{s["valid_forecasts"]}</td><td>{s["physics_active_ticks"]}</td><td>{s["processing_median_ms"]:.2f}/{s["processing_p95_ms"]:.2f}</td><td>{s["wall_including_decode_preview_s"]:.2f}</td></tr>'
    text+='</table><div class="note warn">尚不实时。清晰2秒实际34.27秒、侧视5秒67.04秒；处理中位约50–57ms，P95约169–278ms，都超过10ms预算。包括15模型和视频的研究流水线计算成本高。发布/物理生效次数不是精度或真实转速召回。</div><img src="prediction_contact.png"><h2>3. 自己运行</h2><pre>&amp; "D:\\drone forecast\\.venv-event311\\Scripts\\python.exe" "D:\\drone forecast\\event_drone_forecast\\run_adaptive_physics.py" --raw "E:\\dataset\\train\\6\\media\\event.raw" --start 22 --duration 2 --output "D:\\drone forecast\\output\\my_adaptive_physics" --preview</pre><p>输出目录须不存在；start/duration为10ms整数倍，duration=0表示运行至文件末尾；原生冷启动，无人工框。可用--primary cv、adaptive_history_mix、adaptive_axis_freq等选择主显示模型，所有15个模型始终同时保存。只要RAW，不读取标签、缓存或manifest。原run.py及研究24/26保持。</p><h2>4. 可以和不可以声称的结果</h2><p>可以声称：只输入RAW的图像预测研究系统可运行；逐盘频率、轴、有效像素推力模型、因果在线加权已实际连通；700时刻重解码回归和全部视频完整解码通过。缓存clear历史frequency.py指纹不一致的溯源限制仍保留，但本轮已独立重算其中200时刻与缓存完全一致，不据此推广为整段全部复现。</p><p>不能声称：可靠识别未知真实质量/推力系数/风速、真实RPM误差、任意相机姿态下的真实三维法向、稳定实时、超过原论文。图像向上/非倒飞先验和弱透视仍存在，侧视物理可用时间少，检测错了也会污染在线反馈。下一步用其他自采录像冻结运行检查泛化，并继续保留CV和纯历史混合。</p><p><a href="protocol.json">研究冻结协议</a>、<a href="analysis.json">运行汇总</a>、<a href="video_checks.json">视频核验</a>、<a href="阶段结论.md">结论</a>。无G，未访问执行yolov10-main。</p></main></html>'
    (OUT/'第三十阶段RAW自适应物理预测报告.html').write_text(text,encoding='utf-8')
    (OUT/'阶段结论.md').write_text('# 阶段30：完整RAW路径回归\n\n新根run_adaptive_physics.py，真实执行10ms邻域/阶段9恢复、原RotorSensing、28 canonical_static法向/投影推力、29延迟模型加权，保存15预测器和视频。只有RAW输入，无标签或缓存推理。清晰22–24s/200时刻、侧视108–113s/500时刻的当前观测/频率/全部预测与12/29缓存精确一致。所有视频700帧完整解码。\n\n清晰178检测/192可预报/149物理生效，wall34.27s，中位57.44/P95277.61ms；side327/356/150，wall67.04s，中位50.38/P95168.67ms。尚不实时，视频100fps是源时间轴。整合是回归证据，不是独立新精度。物理专家对侧视未胜纯历史，真实质量/RPM/姿态/风速均无真值。旧入口/default保持。\n',encoding='utf-8');print('Stage30 report and all video frames verified',flush=True)


if __name__=='__main__':main()
