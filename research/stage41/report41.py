import html,json
import cv2,numpy as np
from common41 import *
def table(rows,cols):return '<table><tr>'+''.join('<th>'+c+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c)))+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'
def main():
    verify();a=load(OUT/'analysis.json');curr=load(OUT/'current_localization.json');pair=[]
    for case in CASES:
        for h in [.1,.2,.4]:
            for model in ['cv','trusted_acc']:
                for other in ['body_only','geometry_first','body_first']:
                    dd={method:{r['target_t']:r for r in a['details'] if r['case']==case and r['frontend']==method and r['model']==model and r['nominal_h']==h and r['target_state']=='present'} for method in ['geometry',other]};shared=[t for t in dd['geometry'] if dd['geometry'][t]['forecast_valid'] and dd[other][t]['forecast_valid']]
                    pair.append(dict(case=case,h=h,model=model,other=other,n=len(shared),geometry_mean=float(np.mean([dd['geometry'][t]['error_px'] for t in shared])) if shared else None,other_mean=float(np.mean([dd[other][t]['error_px'] for t in shared])) if shared else None))
    save(OUT/'pairwise_common.json',pair)
    examples=[];video_checks=[]
    for case in ['clear6','side6','train3']:
        obs=list(map(json.loads,(OUT/case/'observations.jsonl').read_text(encoding='utf-8').splitlines()));fore=list(map(json.loads,(OUT/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines()));inputs=[r for r in load(OUT/'inputs.json')['images'] if r['case']==case];path=OUT/case/'comparison.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),50.,(1280,720));assert writer.isOpened()
        for i,(o,q,source) in enumerate(zip(obs,fore,inputs)):
            im=cv2.imdecode(np.fromfile(OUT/source['image'],np.uint8),1)
            for method,color in [('geometry',(240,160,60)),('body_first',(0,185,255))]:
                item=o['methods'][method]['observation'];forecast=q['frontends'][method]
                if item['detected']:
                    b=np.rint(item['bbox']).astype(int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,2)
                if forecast['valid']:
                    pts=[forecast['coefficients']['cv']['position']]+[forecast['predictions']['cv'][str(h)] for h in [.1,.2,.4]]
                    for x,y in zip(pts,pts[1:]):cv2.line(im,tuple(np.rint(x).astype(int)),tuple(np.rint(y).astype(int)),color,2)
            cv2.putText(im,f'{case} {o["t"]:.2f}s blue=rotors-CV orange=body-first-CV',(18,696),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);writer.write(im)
            if i in [len(obs)//3,2*len(obs)//3]:examples.append(cv2.resize(im,(640,360)))
        writer.release();cap=cv2.VideoCapture(str(path));n=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(720,1280,3);n+=1
        cap.release();assert n==len(obs);video_checks.append(dict(case=case,decoded_frames=n))
    canvas=np.concatenate([np.concatenate(examples[i:i+2],axis=1) for i in range(0,len(examples),2)],axis=0);cv2.imencode('.png',canvas)[1].tofile(str(OUT/'forecast_comparison.png'));save(OUT/'video_checks.json',video_checks)
    selected=[r for r in a['table'] if r['frontend'] in ['geometry','body_only','body_first'] and r['model'] in ['cv','trusted_acc'] and r['nominal_h']==.4 and r['case']!='negative5']
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第四十一阶段连续整机预测</title><style>body{font:16px/1.8 Arial;margin:30px auto;max-width:1300px;color:#233d4b}img,video{max-width:100%;width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px;border:1px solid #ccd9e1}th{background:#edf3f7}</style><h1>第四十一阶段：整机检测补上长期失锁，但未来机动仍难预测</h1><p><b>31秒、1550个连续20ms时刻，五条定位分支、相同六个历史预测器。</b>3号片段当前有机目标从3/12变12/12，0.4s预报覆盖从3/12变11/12。检测改进已经传到真实未来评测，但绝对未来误差仍大，不能称持续精准。</p><p>40微调只用1/2/4梯度、6选权重；3对权重留出但已被早期开发查看。41在40训练期间冻结，GPU预测先保存，再按时间顺序回放历史滤波，最后读稀疏框。GPU独立eval图像批处理不是读取未来轨迹。相同20ms原桨盘观测与六系数全部1550时刻精确重现21/23/24；38个与40重合输入PNG SHA完全一致。</p><h2>当前定位</h2>'''+table([r for r in curr['table'] if r['method'] in ['geometry','body_only','body_first']],['case','method','present','detected','joint_success','mean_error_px','absent','false_positive_frames','known_empty_published'])
    text+='<h2>0.4秒未来端点：条件误差必须连同覆盖读</h2>'+table(selected,['case','frontend','model','present_targets','valid_forecasts','mean_fde_px','max_fde_px','joint_success'])
    text+='<h2>共同目标，避免覆盖改变混淆</h2>'+table([r for r in pair if r['h']==.4 and r['other'] in ['body_only','body_first'] and r['case']!='negative5'],['case','model','other','n','geometry_mean','other_mean'])
    text+='''<p>所有五路共同目标包含几乎从不出框的pretrained，故旧评分器全路共同样本为空；这里另给逐对共同目标，不隐瞒定义。3号原有3目标与新增8目标分别保留，不能直接把整体211→137px都归成同目标精度改善。不同模型、短时距失败见<a href="analysis.json">全部评分</a>和<a href="pairwise_common.json">逐对共同评分</a>。侧视收缩平均71.32→69.96px几乎不变，联合成功8→6；定位更准并不保证下一步加速度估计更准。</p><h2>对照视频</h2><p>蓝=原桨盘中心/CV；橙=整机优先/CV，点线为0.1/0.2/0.4秒。没有画未来真值轨迹；视频50fps是源时间轴。</p><video controls src="train3/comparison.mp4"></video><video controls src="side6/comparison.mp4"></video><video controls src="clear6/comparison.mp4"></video><img src="forecast_comparison.png"><p>三视频1150帧全部解码通过。其余400负例当前及预报所有分支0；只覆盖这一空场，不能代替复杂背景误报评测。40整段3号无机时刻另有2输出，未被此短片覆盖。</p><h2>仍然卡在哪里</h2><p>整机网络恢复了对象定位，但没有提供可靠法向或机械转速；姿态、推力的物理支路仍需单独验证。当前框中心小抖动及侧视遮挡会影响速度/加速度，而无人机未来转向在单目过去画面里不可预知。3号0.4s收缩误差约110px、成功仅2/12，离可靠预测还有距离。</p><p>离线4图批次中位约40ms，约10ms/图吞吐，排除解码、SDK预处理和原几何；不是在线10ms延迟或实时证明。已有独立RAW桥42按单张实际处理与IPC测延迟，不能把批次吞吐冒充实时。准备首次漏了原start时刻一次几何初始化，补充冻结startup_supplement并保留失败；最终全部1550精确回归。只解决序列化的40桥也被明确记录，无结果驱动改阈值。</p></html>'''
    (OUT/'第四十一阶段连续整机预测报告.html').write_text(text,encoding='utf-8');(OUT/'阶段结论.md').write_text('当前定位train3从3/12到12/12，0.4s有效3/12到11/12。未来误差仍大，侧视加速度联合成功有退步；必须看pairwise_common共同目标。1550基线精确、38PNG精确、三视频1150帧解码通过。未替换旧默认。',encoding='utf-8');print('Stage41 report and videos complete',flush=True)
if __name__=='__main__':main()
