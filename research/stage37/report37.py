import json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common37 import ROOT,OUT,load,save,verify
from score37 import iou


def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');meta=load(OUT/'model_metadata.json');probe=load(OUT/'nonzero_parity_probe.json');control=load(OUT/'control_predictions.json')['results'];truth={r['id']:r for r in load(OUT/'control_targets.json')['targets']};controls=[]
    for r in control:
        top=max(r['boxes'],key=lambda b:b[4]) if r['boxes'] else None;controls.append(dict(id=r['id'],boxes=len(r['boxes']),top_confidence=top[4] if top else None,top_iou=iou(top[:4],truth[r['id']]['bbox']) if top else None))
    save(OUT/'control_analysis.json',dict(rows=controls,detected=sum(r['boxes']>0 for r in controls),top_iou50=sum(r['top_iou'] is not None and r['top_iou']>=.5 for r in controls),
        scope='Selected FRED functionality controls, training overlap acknowledged; NOT an unbiased benchmark.'))
    inputs=load(OUT/'inputs.json')['images'];targets=load(OUT/'targets.json')['targets'];chosen=[]
    for case,last in [('clear6',False),('train3',False),('side6',True),('train3',True)]:
        tt=[r for r in targets if r['case']==case and r['state']=='present'];t=tt[-1] if last else tt[0]
        image=next(r for r in inputs if r['id']==t['id'] and r['format']=='sdk_default' and r['window_us']==50000);chosen.append((image['image'],t['bbox'],None,f'{case} / {t["target_t"]:.2f}s / NO DET'))
    ci={r['id']:r for r in load(OUT/'control_inputs.json')['images']};cp={r['id']:r for r in control}
    pictures=[]
    for ident in ['fred0_q0.50','fred4_q0.50']:
        pred=max(cp[ident]['boxes'],key=lambda b:b[4]) if cp[ident]['boxes'] else None;pictures.append((ci[ident]['image'],truth[ident]['bbox'],pred[:4] if pred else None,ident+' / model output'))
    pictures=[pictures[0],chosen[0],chosen[1],pictures[1],chosen[2],chosen[3]]
    fig,axs=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    for ax,(path,b,pred,title) in zip(axs.flat,pictures):
        frame=cv2.imdecode(np.fromfile(OUT/path,np.uint8),cv2.IMREAD_COLOR);ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        for box,color in [(b,'#c43f9a'),(pred,'#e6a329')]:
            if box is not None:ax.add_patch(plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=color,lw=1.5))
        ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Purple = archived box reference; orange = detector. Native event images, no reconstructed G.',fontsize=12)
    fig.savefig(OUT/'domain_comparison.png',dpi=170);plt.close(fig)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三十七阶段：公开整机检测器迁移</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1350px;margin:auto;background:white;padding:32px}h1{font-size:30px}img{max-width:100%}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}.warn{background:#fff4dd;border-color:#b88726}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;white-space:nowrap;text-align:left;border-bottom:1px solid #dae4ee}th{background:#eef3f7}a{color:#176c99}</style><main><h1>第三十七阶段：公开整机检测器在自采RAW上没有直接迁移成功</h1><p>2026-10-18 · 固定公开权重 · 40时刻×3原生表示×2过去时间窗 · 240图</p><div class="note warn"><b>conf=0.25下所有自采输入均零检出，暂不接默认预测。</b>29个有机时刻、11个无机时刻，每种方案都相同。该结果不能写成检测成功或未来预测改善。FRED原始事件帧功能对照9张中7张出框，说明网络有基本检测功能，但不是未知自采域泛化保证。</div><h2>1. 复用了什么</h2><p><a href="https://github.com/takara-ai/DroneStalker">DroneStalker</a>公开YOLO检测权重，固定提交93e846eef8abc974da1e0c5cf71331729648f143，16,105,601字节，Git blob SHA1/每文件SHA256核验。作者代码复制FRED Event/Frames训练，默认640/conf0.25/IoU0.45；不是通用RGB模型，也不是本轮新训练。没有执行其Web/游戏/外部服务。</p><p>原检查点只含EMA网络而model字段为None，按EMA读取；显式受控类安全加载后导出纯张量状态，官方已装网络严格重建。2,568,243参数、单drone类。零输入、随机非零及3种真实输入的原EMA/张量重建原始输出逐项完全一致，所检最大绝对差0；未发现此次转换差异。这个检查在当前软件版本进行，不等同完整复现作者旧训练环境。</p><h2>2. 固定试验</h2><p>全相机原生ON/OFF着色(原作者纯get_frame函数)、原生事件计数白底黑图、SDK默认调色板；各取过去10/50ms，阈值不变。SDK默认调色板与FRED控制帧颜色一致。GT只用于准备采样/事后计分；模型输入没有状态/框，检测进程安装文件访问拦截，拒绝读取评分真值。所有240结果保存后再评分。</p>'''
    text+=table(['片段','表示','过去窗ms','有机检出/目标','中心联合通过','最高置信框IoU≥.5','无机发布/目标'],[[r['case'],r['format'],r['window_us']/1000,f"{r['present_detected']}/{r['present']}",r['top_joint_center'],r['top_iou50'],f"{r['absent_published']}/{r['absent']}"] for r in a['table']])
    text+='<p>同一时刻的六种图像相关，不能当240个独立飞行情形；无机0输出也不能弥补有机全部漏检。原桨盘几何在相同当前时刻仍有发布，说明不是所有自采数据都不可见。</p>'
    text+=table(['片段','旧当前几何覆盖','条件中心误差px'],[[r['case'],f"{r['detected']}/{r['present']}",'—' if r['conditional_center_mean_px'] is None else f"{r['conditional_center_mean_px']:.2f}"] for r in a['baseline_current_geometry']])
    text+='<h2>3. 阳性功能对照与域差异</h2><img src="domain_comparison.png"><p>紫框是原有标框，<b>不是检测结果</b>；橙框才是网络结果。FRED目标通常更小，事件外观/密度与自采近距桨盘不同，尺度和噪声是下一步应隔离的因素，尚未证明其中哪项是唯一原因。</p>'
    text+=table(['控制帧','框数','最高置信度','最高框IoU'],[[r['id'],r['boxes'],'—' if r['top_confidence'] is None else f"{r['top_confidence']:.3f}",'—' if r['top_iou'] is None else f"{r['top_iou']:.3f}"] for r in controls])
    text+='<p>这9张来自用户已有FRED压缩包的原始Event/Frames，按三个序列的目标面积分位选取，用于检查功能。权重本来训练过FRED，不能拿7/9称无污染准确率或超过FRED论文。未使用RGB/G重建。自采首个清晰示例原始最大类分数约0.0013/0.0175/0.0058，均低于既定0.25，当前轮未按结果降低阈值。</p><h2>4. 可复核性与工程修正</h2><p>检测运行在Python3.13/PyTorch2.13.0+cu130/Ultralytics8.4.110，SDK输入导出在3.11，以无损PNG桥接；首批重复结果一致。240图GPU批次总约2.90秒，中位每4图47.92ms，这是预导出图像批处理，不是完整RAW在线实时速度。</p><p>首次冻结误把GT框/状态放进采样协议，尚未导出或推理就分离并保留v1；v2协议只有时间/ID，检测文件访问拦截开启。首次导出又遇SDK不能向后seek，保留3张部分图，新增冻结I/O补充：只读最大50ms再派生10ms。33个10ms窗口事件数与原观测逐项相同，全部窗无未来事件。没有为出分修改检测参数，失败与修正均留档。</p><h2>5. 下一步</h2><p>先在固定权重/阈值下做原生事件噪声过滤和空间重分箱对照，测试与FRED训练外观的差异；坐标必须严格反投影回原图并保留无机负例。如果仍无效，再考虑弱标框微调或更针对桨盘的EVPropNet，不能直接把公开权重接进主预测后声称有效。</p><p><a href="protocol.json">v2协议</a>、<a href="analysis.json">自采评分</a>、<a href="inference.json">推理/访问拦截</a>、<a href="nonzero_parity_probe.json">非零一致性</a>、<a href="control_analysis.json">功能对照</a>、<a href="preparation_bridge.json">I/O补充</a>、<a href="input_window_checks.json">时间窗核验</a>、<a href="阶段结论.md">结论</a>。旧默认不改，无G/yolov10-main访问。</p></main></html>'
    (OUT/'第三十七阶段公开整机检测器迁移报告.html').write_text(text,encoding='utf-8')
    (OUT/'阶段结论.md').write_text('# 阶段37\n\n固定DroneStalker提交93e846e公开YOLO16MB权重，受控安全读取EMA并严格张量重建；零/随机非零/真实3表示原logit完全相同。40自采/无机时刻×3表示×10/50ms=240图，conf0.25全部零检出，暂不接预测。FRED原生事件9功能对照7出框，训练重合，非无污染准确率。\n\n完整结果先保存后评分；模型输入/元数据无框状态，访问真值文件会拦截。早期冻结v1含框元数据在任何输入/预测前已拆分，旧源/协议保留。SDK不能倒退seek的I/O错误以最大窗一次读/子窗筛选修正，部分3图保留，33个10ms窗口数与旧当前观测一致。240GPU批图约2.9s非全RAW实时。\n\n下一步隔离事件背景噪声/空间尺度差异，继续保留固定阈值及负例、准确坐标还原。纯迁移失败不证明所有事件网络无效，也不调低阈值凑分。\n',encoding='utf-8');print('Stage37 report rendered',flush=True)


if __name__=='__main__':main()
