import collections,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study36 import OUT,ROOT,CASES,METHODS,load,save,verify


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');current=[];runtime=[];diag=[];drop=[];train_rows=None;fig,axs=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
    for case,cfg in CASES.items():
        rr=list(map(json.loads,(OUT/case/'observations.jsonl').read_text(encoding='utf-8').splitlines()));tt=np.array([r['t'] for r in rr]);runtime.append(dict(case=case,**load(OUT/case/'completion.json')))
        if case=='clear_dropout':
            ff=list(map(json.loads,(OUT/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines()))
            for m in METHODS:
                assert not any(r['methods'][m]['observation']['detected'] for r in rr if 23<=r['t']<23.5)
                assert not any(r['frontends'][m]['valid'] for r in ff if 23.17<=r['t']<23.5)
            drop.append(dict(case=case,all6_current_zero_during500ms=True,all6_new_forecasts_expired_after170ms=True));continue
        path=ROOT.parent/'output/raw_only_project/evaluation_data'/('train3.json' if case=='train3' else 'train6.json')
        labels=[l for l in load(path)['labels'] if cfg['start']<l['t']<=cfg['start']+cfg['duration'] and l['state']=='present']
        for m in METHODS:
            errors=[];individual=[]
            for l in labels:
                i=int(np.searchsorted(tt,l['t'],side='right')-1)
                if i<0 or l['t']-tt[i]>.025:continue
                o=rr[i]['methods'][m]['observation']
                if o['detected']:
                    b=np.asarray(l['bbox']);error=float(np.linalg.norm(np.asarray(o['center'])-(b[:2]+b[2:])/2));errors.append(error);individual.append(dict(t=l['t'],error_px=error,source=o.get('source','geometry')))
            current.append(dict(case=case,method=m,covered=len(errors),targets=len(labels),mean_px=float(np.mean(errors)) if errors else None,details=individual))
            diag.append(dict(case=case,method=m,reseed_ticks=sum(r['methods'][m]['audit'].get('feature_reseeded',False) for r in rr),
                body_times=[r['t'] for r in rr if r['methods'][m]['observation'].get('source')=='body_flow'],
                reasons=dict(collections.Counter(r['methods'][m]['audit'].get('reason',r['methods'][m]['audit']['source']) for r in rr))))
        if case in ['side6','train3']:
            ax=axs[0 if case=='side6' else 1]
            for j,m in enumerate(METHODS):ax.step(tt,[int(r['methods'][m]['observation']['detected'])+j*1.15 for r in rr],where='post',label=m)
            ax.set(title=case,ylabel='current measurement (+row offset)');ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
        if case=='train3':train_rows=rr
    axs[-1].set_xlabel('RAW time (s)');fig.savefig(OUT/'current_coverage.png',dpi=150);plt.close(fig)
    save(OUT/'diagnostics.json',dict(current=current,runtime=runtime,methods=diag,dropout=drop))
    # Existing count-video is post-hoc visual background only, never tracker input.
    cap=cv2.VideoCapture(str(ROOT.parent/'output/stage31_cross_recording_physics_20261018/train3/forecasts.mp4'));fig,axs=plt.subplots(1,4,figsize=(16,4),constrained_layout=True)
    tt=np.array([r['t'] for r in train_rows]);label=next(r for r in load(ROOT.parent/'output/raw_only_project/evaluation_data/train3.json')['labels'] if 26.69<r['t']<26.70)
    for ax,t in zip(axs,[26.64,26.69,27.05,27.09]):
        i=int(np.argmin(abs(tt-t)));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read();assert ok
        ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));o=train_rows[i]['methods']['replenish3_similarity']['observation']
        if o['detected']:
            b=o['bbox'];ax.add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#1b8ec2',lw=2));ax.scatter(*o['center'],color='#1b8ec2',s=25)
        if abs(t-26.69)<.001:
            b=label['bbox'];ax.add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#b93e94',lw=2))
        ax.set(title=f'{tt[i]:.2f}s / body={o.get("source")=="body_flow"}',xlim=(0,420),ylim=(570,200));ax.grid(alpha=.1)
    cap.release();fig.savefig(OUT/'body_continuation_views.png',dpi=150);plt.close(fig)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三十六阶段：完整侧视整机补测</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1350px;margin:auto;background:white;padding:32px}h1{font-size:30px}img{max-width:100%}.note{padding:18px;background:#edf6fb;border-left:5px solid #3787a7}.warn{background:#fff4dd;border-color:#b88726}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;white-space:nowrap;text-align:left;border-bottom:1px solid #dae4ee}th{background:#eef3f7}details{padding:15px;border:1px solid #dae4ee;margin:18px 0}a{color:#176c99}</style><main><h1>第三十六阶段：少点支持和补点能撑多久</h1><p>2026-10-18 · 侧视扩大到完整108–120秒 · 原6点严格回归+3点支持/补点</p><div class="note"><b>补点延长了短时接续，仍未解决失锁后的重新发现。</b>train3补点获得45个10ms body时刻，延续至27.08s，约0.45秒后再次失效；后面8个粗框标签仍缺当前定位。侧视新增18/1199时刻，0.4s CV127.62→126.38px，收益很小。</div><h2>1. 原实现不改，只隔离必要因素</h2><p>执行冻结阶段35的独立模块副本：原6点长相似/平移，两种3点版本，以及3点相似的当前ROI补点。其余图像、双向校验、残差/尺度/旋转/2s年龄不改。补点只在当前运动已经被接受且点数少于40时进行，不能在失效后凭自己重新发现；原独立几何年龄、累计尺度参考和有方向四角保留，防止补点延寿或反复轴对齐框膨胀。3个图像点不是三维姿态或语义身份保证。</p><p>side全12秒、clear2秒、train3 6秒、clear断流控制2秒，总2199时刻。阶段35四case共同1500前缀的原baseline/长相似/长平移完整观测和六预测全部相同，旧源未编辑。</p><h2>2. 当前发布与当前框</h2>'''
    text+=table(['片段','方法','当前发布','body补测','总时刻'],[[r['case'],m,r['methods'][m]['detected'],r['methods'][m]['body_measurements'],r['frames']] for r in runtime for m in METHODS])
    text+=table(['片段','方法','当前框覆盖','条件中心误差px'],[[r['case'],r['method'],f"{r['covered']}/{r['targets']}",fmt(r['mean_px'])] for r in current])
    text+='<img src="current_coverage.png"><h2>3. 跟得更久，不等于中心就准确</h2><img src="body_continuation_views.png"><p>蓝框/蓝点是补点模型，紫框是26.697s近似可见整机框；背景为既有原生10ms计数视频(含旧预报线)，仅事后展示，不是光流输入。26.69s补点中心约(162.65,351.48)，可见框中心约(89.18,362.10)，相差约74px。贴近视野边缘时，继承的整机中心、局部可见部分和近似框中心会分离；这不是已恢复可靠质心的证据，也不能用现有框直接断言真实质心误差。</p><p>27.09s仅剩2个一致点而停止，补点不能凭空恢复已消失的目标特征。后续长期缺失需要重新检测或更合适事件跟踪。短时接续与重新发现应分开评分。</p><h2>4. 未来位置：全部覆盖和共同样本一起看</h2>'
    for case in ['clear6','side6','train3']:
        text+=f'<details'+(' open' if case!='clear6' else '')+f'><summary>{case}</summary>'+table(['方法','模型','名义s','有效/目标','均值FDEpx','最大px','联合通过'],[[r['frontend'],r['model'],r['nominal_h'],f"{r['valid_forecasts']}/{r['present_targets']}",fmt(r['mean_fde_px']),fmt(r['max_fde_px']),r['joint_success']] for r in a['table'] if r['case']==case and r['model'] in ['cv','trusted_acc']])+'</details>'
    text+='<div class="note warn">train3名义0.4s：baseline208.52px(3/12)、3点平移174.01(4/12)、补点相似184.83(4/12)。共同三个目标的CV仍完全相同，新增覆盖和新增目标难度影响均值；补45帧的方案也未胜只补4帧的平移。不能只按补测数选最终系统。side相同12/14目标CV127.62→126.38，只是小幅开发收益。</div><h2>5. 验证与下一步</h2><p>合成平移、补点不刷新独立年龄、累计有方向框保留与到期停止通过；全断流500ms各路当前零发布，170ms后无新预报。原baseline普通1999时刻六历史模型精确回归，旧35共同前缀完整记录回归。没有独立电机/姿态真值或新标注。</p><p>不推广默认。下一步优先复用现成整机检测器作重新发现的小迁移测试，不能继续只放宽光流门槛；桨盘几何/频率保留在ROI内用于可观测性足够的时刻。此前核查的DroneStalker有公开整机权重，但混合训练/域差异需要检查；测试自采数据不能据此宣称FRED无污染基线。无G/yolov10-main访问。</p><p><a href="protocol.json">协议</a>、<a href="analysis.json">全部与共同FDE</a>、<a href="diagnostics.json">当前框/补点/断流</a>、<a href="strict_prefix_regression.json">严格回归</a>、<a href="module_checks.json">合成检查</a>、<a href="阶段结论.md">结论</a>。</p></main></html>'
    (OUT/'第三十六阶段完整侧视整机补测报告.html').write_text(text,encoding='utf-8')
    (OUT/'阶段结论.md').write_text('# 阶段36结论\n\n全side1199时刻加clear/train3/dropout合2199。原35共同1500前缀三严格分支观测/六预测精确相同，普通1999 baseline系数回归；断流500ms当前零发布，170ms后新预报均停。\n\nside baseline796，原长路各+13，3点相似+17/平移+16，补点+18；0.4s同12/14 CV127.62→补点126.38px，小收益。train3 baseline163，原长相似+1/平移+2，3点各+4，补点+45至27.08s后失败；当前新增标签26.697s可见框中心偏差约74px，不是真实质心误差。后8标签仍漏。\n\ntrain3 0.4s CV208.52(3/12)，3点平移174.01(4/12)，补点184.83(4/12)；共同3目标CV完全不变。更多补测不等于更准，不推广默认。下一步用现成整机检测器重发现，继续区分当前观测、纯外推和频率/姿态可观测性。\n',encoding='utf-8');print('Stage36 report rendered',flush=True)


if __name__=='__main__':main()
