import collections,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study35 import ROOT,OUT,CASES,METHODS,load,save,verify,study34


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');current=[];reasons=[];drops=[];runstats=[];fig,axs=plt.subplots(2,1,figsize=(13,7),constrained_layout=True)
    for case,cfg in CASES.items():
        rr=list(map(json.loads,(OUT/case/'observations.jsonl').read_text(encoding='utf-8').splitlines()));tt=np.array([r['t'] for r in rr]);ff=list(map(json.loads,(OUT/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines()));runstats.append(dict(case=case,**load(OUT/case/'completion.json')))
        if case=='clear_dropout':
            for m in METHODS:
                indices=[i for i,t in enumerate(tt) if 23<=t<23.5];body=sum(rr[i]['methods'][m]['observation']['detected'] for i in indices);late=sum(ff[i]['frontends'][m]['valid'] for i in indices if tt[i]>=23.17)
                assert body==0 and late==0
                drops.append(dict(method=m,current_detections_during_dropout=body,valid_new_forecasts_after_170ms=late))
            continue
        labels=[l for l in load(study34.ROOT.parent/'output/raw_only_project/evaluation_data'/('train3.json' if case=='train3' else 'train6.json'))['labels'] if cfg['start']<l['t']<=cfg['start']+cfg['duration'] and l['state']=='present']
        for m in METHODS:
            errors=[]
            for label in labels:
                i=int(np.searchsorted(tt,label['t'],side='right')-1)
                if i<0 or label['t']-tt[i]>.025:continue
                o=rr[i]['methods'][m]['observation']
                if o['detected']:
                    b=np.array(label['bbox']);errors.append(float(np.linalg.norm(np.asarray(o['center'])-(b[:2]+b[2:])/2)))
            current.append(dict(case=case,method=m,covered=len(errors),targets=len(labels),conditional_mean_px=float(np.mean(errors)) if errors else None))
            reasons.append(dict(case=case,method=m,reasons=dict(collections.Counter(r['methods'][m]['audit'].get('reason',r['methods'][m]['audit']['source']) for r in rr))))
        if case in ['side6','train3']:
            ax=axs[0 if case=='side6' else 1]
            for m in METHODS:ax.step(tt,[int(r['methods'][m]['observation']['detected'])+METHODS.index(m)*1.2 for r in rr],where='post',label=m)
            ax.set(title=case,ylabel='current measurement (+row offset)');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2)
    axs[-1].set_xlabel('RAW time (s)');fig.savefig(OUT/'body_measurement_coverage.png',dpi=150);plt.close(fig)
    save(OUT/'diagnostics.json',dict(current=current,reasons=reasons,dropout=drops,runstats=runstats))
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三十五阶段：整机事件特征光流补测</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1300px;margin:auto;background:white;padding:32px}h1{font-size:30px}img{max-width:100%}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}.warn{background:#fff4dd;border-color:#b88726}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;white-space:nowrap;text-align:left;border-bottom:1px solid #dae4ee}th{background:#eef3f7}details{padding:15px;border:1px solid #dae4ee;margin:18px 0}a{color:#176c99}</style><main><h1>第三十五阶段：用整机特征补测，没有解决困难侧视</h1><p>2026-10-18 · 原生ON/OFF计数图 · OpenCV现成光流 · 1500时刻及断流控制</p><div class="note warn"><b>保留失败，不升级默认。</b>清晰补20个当前时刻；侧视没有补测成功；train3相似仅补1个10ms时刻、平移补2个。短300ms/长2s没有实质区别，因为特征先丢了；继续加有效期不能解决。</div><h2>1. 实际实现</h2><p>只有原生ON+OFF的20ms事件计数图(每10ms截止)，不是重建G。检测成功时在整机框中用OpenCV GFTT取最多80个强事件特征；之后用PyrLK正反跟踪、当前事件支持、≥6点、RANSAC/中位平移、尺度/旋转/累计跨度门槛产生整机当前中心。光流期间不补新特征，所有body输出明确rotors=[]，不伪造转速/姿态。年龄自上一次独立几何观测起算。</p><p>比较原当前几何、300ms/2s相似、2s平移。当前几何来自原冻结缓存，每个匹配RAW截止顺序消费；全流程没有读取未来标签或带标注视频作光流输入。频率/物理不在此消融里重算，预测只用六历史模型，普通case原baseline有效性/系数与先前完全一致。</p><h2>2. 发布与当前框</h2>'''
    text+=table(['片段','方法','当前发布','其中body补测','总时刻'],[[r['case'],m,r['methods'][m]['detected'],r['methods'][m]['body_measurements'],r['frames']] for r in runstats for m in METHODS])
    text+=table(['片段','方法','当前框覆盖','条件中心误差px'],[[r['case'],r['method'],f"{r['covered']}/{r['targets']}",fmt(r['conditional_mean_px'])] for r in current])
    text+='<img src="body_measurement_coverage.png"><h2>3. 为什么均值会让人误读</h2><p>train3名义0.4s，原CV208.52px(3/12)，平移177.79px(4/12)。但共同三个目标的CV误差两者都208.52px，完全没改善；两次补测仅让短时续推多维持20ms，多覆盖一个误差较小目标。不能把均值下降30px写成共同轨迹预测更准。后9个当前标签依旧缺测。</p>'
    for case in ['clear6','side6','train3']:
        text+=f'<details'+(' open' if case=='train3' else '')+f'><summary>{case}未来位置</summary>'+table(['前端','模型','名义s','有效/目标','均值FDEpx','最大px','联合通过'],[[r['frontend'],r['model'],r['nominal_h'],f"{r['valid_forecasts']}/{r['present_targets']}",fmt(r['mean_fde_px']),fmt(r['max_fde_px']),r['joint_success']] for r in a['table'] if r['case']==case and r['model'] in ['cv','trusted_acc']])+'</details>'
    text+='<h2>4. 卡点定位</h2><p>side第一次缺几何时只剩5个前向特征，低于6点；一旦拒绝就不再用光流自己重建身份。train3在26.65s相似变换被拒，26.66s平移剩4个一致点而终止。事件亮度图的桨叶周期变化、姿态变形、部分遮挡与特征消失，违背普通亮度块稳定跟踪条件。部分几何更新本身也只有少量强特征，不能只把错误归成TTL太短。</p><p>这里侧视只测108–113s，并未覆盖116–120s所有极端片段；不扩大成完整侧飞已测试。后续如再验证应冻结协议扩大到完整12s，并比较更少点的平移/可靠当前ROI补点，同时监测漂移，不能直接放宽门槛宣布改善。</p><h2>5. 验证与来源</h2><p>合成已知平移误差&lt;1.5px、原几何不变、不造桨盘、无当前事件不发、光流不自播种通过。完全中断500ms时四路当前零发布，170ms后新预测均失效；普通1300 baseline时刻六模型精确回归。真实短前缀核验见链接，包含实际body时刻。</p><p>执行<a href="https://github.com/opencv/opencv/blob/5.x/doc/tutorials/others/optical_flow.markdown">OpenCV作者教程对应API</a>；查阅<a href="https://github.com/ialzugaray/haste">HASTE事件特征跟踪</a>和<a href="https://github.com/uzh-rpg/rpg_eklt">EKLT作者实现</a>，未编译运行其C++/ROS。EKLT是帧与事件互补，这里没有把纯计数图LK冒称EKLT。当前环境未找到cl/clang++/g++/cmake/ninja，原作者系统仍需独立构建准备。</p><p><a href="protocol.json">协议</a>、<a href="analysis.json">完整FDE/共同样本</a>、<a href="diagnostics.json">当前覆盖/拒绝/断流</a>、<a href="module_checks.json">合成检查</a>、<a href="real_prefix_check.json">真实前缀检查</a>、<a href="阶段结论.md">结论</a>。无G/yolov10-main访问，旧默认保留。</p></main></html>'
    (OUT/'第三十五阶段整机事件特征补测报告.html').write_text(text,encoding='utf-8')
    (OUT/'阶段结论.md').write_text('# 阶段35结论\n\n1500时刻纯ON/OFF计数图GFTT/LK/FB/RANSAC；clear补20，side0，train3相似1/平移2。当前稀疏框覆盖没增加。原几何/普通baseline六模型精确回归；断流500ms当前零发布，170ms后无新预报。\n\ntrain3 0.4s CV208.52(3/12)→平移177.79(4/12)，但共同3点均208.52，变化是新目标覆盖，非共同轨迹更准。side首缺测只有5前向点，train3随后仅4一致点，先丢特征而非TTL到期，长2s无收益。不推广默认。\n\nside只108–113s，后续应扩大到全108–120，隔离3点支持/ROI补点与漂移；仍需更合适事件特征或整机再检测。未执行HASTE/EKLT，不冒称其复现。无G、独立姿态/RPM真值或新标注。\n',encoding='utf-8');print('Stage35 report rendered',flush=True)


if __name__=='__main__':main()
