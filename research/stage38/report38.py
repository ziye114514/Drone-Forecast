import json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common38 import ROOT,OUT,OLD,CASES,PARAMS,load,save,sha,verify


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');details={r['id']:r for r in a['details'] if r['method']=='fused_all'}
    fused=list(map(json.loads,(OUT/'fused_predictions.jsonl').read_text(encoding='utf-8').splitlines()));assert sha(OUT/'fused_predictions.jsonl')==load(OUT/'fusion_completion.json')['fused_sha256']
    assert all(r['boxes']==details[r['id']]['boxes'] for r in fused);save(OUT/'fusion_regression.json',dict(frames=len(fused),independent_truth_blocked_fusion_matches_scorer_exactly=True))
    targets={r['id']:r for r in load(OLD/'targets.json')['targets']};inputs=load(OLD/'inputs.json')['images'];selected=[r for r in a['details'] if r['method']=='fused_all' and r['detected'] and r['state']=='present']
    selected += [details[ident] for ident in [r['id'] for r in targets.values() if r['case']=='train3' and r['state']=='present'][-2:]]
    fig,axs=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for ax,d in zip(axs.flat,selected):
        t=targets[d['id']];image=next(r for r in inputs if r['id']==d['id'] and r['format']=='sdk_default' and r['window_us']==50000)
        frame=cv2.imdecode(np.fromfile(OLD/image['image'],np.uint8),cv2.IMREAD_COLOR);ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));b=t['bbox']
        boxes=[b]
        ax.add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,color='#bd3f9b',lw=2))
        if d['boxes']:
            top=max(d['boxes'],key=lambda b:b['confidence']);q=top['bbox'];boxes.append(q);ax.add_patch(plt.Rectangle((q[0],q[1]),q[2]-q[0],q[3]-q[1],fill=False,color='#e9a32a',lw=2))
        bb=np.array(boxes);lo=np.maximum(bb[:,:2].min(0)-35,0);hi=np.minimum(bb[:,2:].max(0)+35,[1280,720]);ax.set(xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]),title=d['id']+(' / model' if d['detected'] else ' / NO DET'));ax.set_xticks([]);ax.set_yticks([])
    for ax in list(axs.flat)[len(selected):]:ax.set_visible(False)
    fig.suptitle('Purple = archived whole-UAV box; orange = top confidence detection mapped back to native image. Background50ms, candidate10/50ms.',fontsize=11)
    fig.savefig(OUT/'mapped_detections.png',dpi=160);plt.close(fig)
    counts=load(OUT/'preparation.json')['counts'];retention=[]
    for case in CASES:
        for window in PARAMS['windows_us']:
            ids=[r['id'] for r in targets.values() if r['case']==case];ratios=[]
            for ident in ids:
                before=next(r['events'] for r in counts if r['id']==ident and r['window_us']==window and not r['noise']);after=next(r['events'] for r in counts if r['id']==ident and r['window_us']==window and r['noise'])
                if before:ratios.append(after/before)
            retention.append(dict(case=case,window_us=window,mean_global_event_retention=float(np.mean(ratios)) if ratios else None))
    save(OUT/'event_retention.json',retention)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三十八阶段：事件噪声与尺度迁移</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1350px;margin:auto;background:white;padding:32px}h1{font-size:30px}img{max-width:100%}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}.warn{background:#fff4dd;border-color:#b88726}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;white-space:nowrap;text-align:left;border-bottom:1px solid #dae4ee}th{background:#eef3f7}a{color:#176c99}</style><main><h1>第三十八阶段：降噪和尺度确有影响，但“出框”仍不等于找对整机</h1><p>2026-10-18 · 权重/阈值固定 · SDK现成过滤×空间重分箱 · 同40时刻/480图</p><div class="note"><b>从全零变成少量检出，尚不够接管当前定位。</b>固定多视图融合：清晰3/3出框，侧视3/14，另一自采0/12；11个无机时刻零出框。29个有机时刻最高置信框只有1个IoU≥0.5，部分框只覆盖局部，不能把6个有框时刻称整机检测已解决。</div><h2>1. 只改变事件表示</h2><p>使用安装SDK的<a href="https://docs.prophesee.ai/stable/api/python/cv/bindings.html#metavision_sdk_cv.ActivityNoiseFilterAlgorithm">ActivityNoiseFilterAlgorithm</a>，10ms邻域活动时间窗；先读过去60ms，让50ms输出窗前有10ms历史。另一因素是原始事件坐标整数重分箱(d=1/4/8)并居中到同一1280×720画布，极性/时间保持，SDK同一调色板。三个尺度×过滤开关×10/50ms共12表示，权重和conf0.25/IoU0.45均不变。</p><p>框必须中心在有效视口内，再按(p−offset)·d还原并裁到原图。融合只按置信度/NMS，真值不可访问的独立进程先保存40个融合结果，之后评分；与评分器重算逐项一致。d1/未过滤80张图像与阶段37 SDK输入SHA完全一致，检测结果也完全相同。</p><h2>2. 预设主方案：全部固定视图融合</h2>'''
    text+=table(['片段','有机出框/目标','中心联合通过','最高框IoU≥.5','任意框IoU≥.5(诊断)','条件中心误差px','无机发布'],[[r['case'],f"{r['detected']}/{r['present']}",r['top_joint_center'],r['top_iou50'],r['any_iou50'],fmt(r['conditional_mean_px']),f"{r['absent_published']}/{r['absent']}"] for r in a['table'] if r['method']=='fused_all'])
    text+='<img src="mapped_detections.png"><p>紫框为原粗框参考，<b>橙框才是检测结果</b>。背景显示相同截止的原生50ms事件图，候选来自10/50ms，都是过去数据。清晰段最高框中心均值39.37px，比旧桨盘几何约10px差；侧视3个检出中心较近，但框范围仍没覆盖完整目标。另一录像的长时失锁依旧未补回。</p><h2>3. 因素分解</h2>'
    rows=[]
    for noise in [False,True]:
        for d in [1,4,8]:
            for window in [10000,50000]:
                name=f'n{int(noise)}_d{d}_w{window}';per={r['case']:r for r in a['table'] if r['method']==name}
                rows.append([noise,d,window/1000,*[f"{per[c]['detected']}/{per[c]['present']}" for c in ['clear6','side6','train3']],sum(per[c]['absent_published'] for c in per)])
    text+=table(['活动过滤','重分箱d','过去窗ms','清晰检出','侧视检出','train3检出','无机出框/11'],rows)
    text+='<p>仅过滤/原尺度仍0；仅d4在清晰只有1/3；过滤+d4清晰3/3、侧视10ms窗2/14。支持噪声与外观尺度共同影响这次迁移，但不是对所有事件模型/传感器的定论。12张相关视图不能当12倍独立飞行情形。</p><h2>4. 验证与代价</h2><p>SDK滤波合成前缀因果一致，坐标投影不改时间/极性，反投影/量化界/填充区域拒绝通过。所有480输出先保存，GT只供评分；80原尺度图像/结果严格回归，40独立融合结果逐项回归。评测仍是开发自采的近似整机框，无真实姿态/RPM/质心真值。</p><p>GPU480批图约5.19秒，中位每4图39.74ms；每个原始时刻需要12个视图，尚未计RAW解码/SDK表示/数据搬运，不能写作完整实时100Hz。推理前已把评分真值文件的校验从运行时移到评分专用字段，保留原冻结，访问拦截未绕开。</p><h2>5. 下一步取舍</h2><p>保持旧默认。可把现成SDK过滤独立放到原桨盘前端验证，避免把神经迁移与几何变化混成一个收益；整机检测若要恢复train3长期漏测，需要针对自采事件外观的训练或更合适的预训练模型，而不是仅凭更多缩放视图。全部方案必须继续报无机误报、覆盖和整机框范围，不能只报中心条件均值。</p><p><a href="protocol.json">v2协议</a>、<a href="analysis.json">全部单视图/融合评分</a>、<a href="fusion_regression.json">融合无真值回归</a>、<a href="event_retention.json">全图事件保留率(非信号召回)</a>、<a href="module_checks.json">检查</a>、<a href="阶段结论.md">结论</a>。无G/yolov10-main访问。</p></main></html>'
    (OUT/'第三十八阶段事件噪声与尺度迁移报告.html').write_text(text,encoding='utf-8')
    (OUT/'阶段结论.md').write_text('# 阶段38\n\n固定权重/conf0.25，SDK活动过滤10ms与事件整数重分箱d1/4/8，过去10/50ms同40时刻480图。融合clear3/3/39.37px，side3/14/18.72px，train3 0/12，11无机0。29有机仅1最高框IoU>=.5，部分只框局部，当前定位未解决，不升级默认。\n\n仅滤波原尺度0，未滤d4清晰1/3，滤波+d4清晰3/3、side10ms2/14，有交互影响但未形成足够覆盖/框精度。原尺度80PNG与输出严格回归37，独立无真值融合40结果回归评分器；GPU批图非全RAW实时。\n\n下一步单独试SDK过滤的桨盘前端收益；神经整机模型仍需跨自采域适应/训练。保持所有失败和指标范围，不能将有框数写成整机精度。\n',encoding='utf-8');print('Stage38 report rendered',flush=True)


if __name__=='__main__':main()
