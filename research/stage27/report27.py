"""Separate feature availability, adaptation and actual endpoint accuracy."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study27 import OUT,ROOT,CASES,MODELS,PHYSICS,load,save,verify,input_path


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def diagnostics():
    result=[];fig,axs=plt.subplots(4,2,figsize=(14,12),constrained_layout=True)
    for col,case in enumerate(CASES):
        raw=list(map(json.loads,input_path(case).read_text(encoding='utf-8').splitlines()))
        rr=[r['frontends']['per_id_frequency'] for r in map(json.loads,(OUT/case/'forecasts.jsonl').read_text(encoding='utf-8').splitlines())]
        tt=np.array([r['t'] for r in rr]);active=[r['feature_diagnostic']['active'] for r in rr]
        diag=[r['feature_diagnostic'] for r in rr];model=[r['models']['adaptive_axis_freq'] for r in diag]
        norm=np.array([m['normal'] if m['normal'] is not None else [np.nan]*3 for m in model]);ratio=np.array([m['ratio'] for m in model])
        counters=[m['updates'] for m in model];increments=[max(0,v-(counters[i-1] if i else 0)) for i,v in enumerate(counters)]
        allowed=np.array([m['update_allowed'] for m in model]);blocks=[];begin=None
        for i,yes in enumerate(allowed):
            if yes and begin is None:begin=i
            if begin is not None and (not yes or i==len(allowed)-1):
                end=i if yes else i-1;blocks.append(float(tt[end]-tt[begin]+.01));begin=None
        active_targets={}
        analysis=load(OUT/'analysis.json')
        for h in [.1,.2,.4]:
            dd=[d for d in analysis['details'] if d['case']==case and d['model']=='adaptive_axis_freq' and d['nominal_h']==h and d['target_state']=='present' and d['forecast_valid']]
            aa=[d for d in dd if rr[int(np.searchsorted(tt,d['anchor_t']))]['feature_diagnostic']['active']]
            active_targets[str(h)]=dict(valid_targets=len(dd),feature_active_targets=len(aa),actual_h_min=min([d['actual_h'] for d in dd],default=None),actual_h_max=max([d['actual_h'] for d in dd],default=None))
        result.append(dict(case=case,frames=len(rr),valid_forecast_ticks=sum(r['valid'] for r in rr),feature_active_ticks=sum(active),
            four_frequency_ticks=sum(r['sensing']['four_frequencies'] for r in raw),four_visible_ticks=sum(r['sensing']['four_visible'] for r in raw),
            update_allowed_ticks=int(allowed.sum()),longest_contiguous_allowed_s=max(blocks,default=0),actual_update_events=sum(increments),
            axis_points_down_when_active=sum(bool(active[i] and norm[i,1]>0) for i in range(len(rr))),
            frequency_ratio_min=float(ratio.min()),frequency_ratio_max=float(ratio.max()),active_targets=active_targets,
            final_effective_parameters=model[-1]['effective_parameters'],final_residual=model[-1]['disturbance_px_s2']))
        axs[0,col].plot(tt,ratio);axs[0,col].set(title=case,ylabel='relative frequency squared')
        axs[1,col].plot(tt,norm[:,1]);axs[1,col].axhline(0,color='black',ls='--',lw=1);axs[1,col].set_ylabel('filtered normal y (+down)')
        axs[2,col].step(tt,np.asarray(active,int),label='physics active');axs[2,col].step(tt,allowed.astype(float)+1.2,label='update allowed (+1.2)');axs[2,col].legend(fontsize=8);axs[2,col].set_ylabel('gates')
        axs[3,col].plot(tt,counters);axs[3,col].set(ylabel='updates in current segment',xlabel='RAW time (s)')
        for ax in axs[:,col]:ax.grid(alpha=.2)
    fig.savefig(OUT/'feature_diagnostics.png',dpi=140);plt.close(fig);save(OUT/'feature_diagnostics.json',result);return result


def endpoint_figure(a):
    fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for row,case in enumerate(CASES):
        for col,model in enumerate(['cv','trusted_acc','adaptive_axis_freq']):
            ax=axs[row,col];dd=[d for d in a['details'] if d['case']==case and d['model']==model and d['nominal_h']==.4 and d['target_state']=='present' and d['forecast_valid']]
            for d in dd:
                target=np.asarray(d['target_center']);pred=np.asarray(d['predicted'])
                ax.plot([target[0],pred[0]],[target[1],pred[1]],color='#b5bfca',lw=1)
                ax.scatter(*target,marker='x',color='#25885e',s=30);ax.scatter(*pred,color='#d97e28',s=14)
            ax.set(title=f'{case} / {model} / n={len(dd)}',xlim=(0,1280),ylim=(720,0),xlabel='x (px)',ylabel='y (px)');ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle('Nominal 0.4s endpoints: green x = archived box center, orange = forecast; lines are errors')
    fig.savefig(OUT/'future_endpoints.png',dpi=150);plt.close(fig)


def main():
    verify();a=load(OUT/'analysis.json');d=diagnostics();endpoint_figure(a)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十七阶段：姿态与转速的预测贡献</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1350px;margin:auto;background:white;padding:32px}h1{font-size:30px}h2{margin-top:35px}.note{background:#edf6fb;border-left:5px solid #3787a7;padding:18px}.warn{background:#fff4dd;border-color:#bd8c28}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{white-space:nowrap;text-align:left;padding:8px;border-bottom:1px solid #dae4ee}th{background:#eef3f7}img{max-width:100%}details{padding:15px;border:1px solid #dae4ee;margin:15px 0}a{color:#176c99}</style><main><h1>第二十七阶段：桨盘姿态、转速与误差修正到底有用多少</h1><p>2026-10-18 · 阶段12真实RAW观测缓存 · 清晰5秒+侧视12秒 · 同定位/速度/覆盖对照</p><div class="note"><b>核心物理路径已参加真正未来框端点评测。</b>清晰0.4s，CV21.89px，姿态+转速+自适应20.71px，7个有效目标，属于很小的开发集收益。侧视127.62→129.13px未改善；侧视自适应没有发生任何更新，不能据此判定“在线学习无效”。</div><h2>1. 复用了什么</h2><p>原论文<a href="https://arxiv.org/abs/2604.18289">Relative State Estimation using Event-Based Propeller Sensing</a>使用逐桨频率作为推力输入、桨盘椭圆反投影获取倾斜轴。本项目复用既有paper_physics.py的法向滤波、推力前向积分、延迟误差更新，并把所有预测器固定到相同阶段23 CV位置/速度状态；不把另一套位置滤波的改变算作动力学收益。不是作者代码，也不是作者成绩复现。</p><p>本轮重用阶段12已保存的原生10ms位置和逐ID频率，共1699时刻，不重新猜频率或读取未来事件。六历史模型完整保持；四物理分支分别为名义姿态+频率、自适应姿态(相对频率固定1)、自适应姿态+频率、自适应固定初始轴+频率。所有物理分支共享频率/几何新鲜度和质量准入，缺失时精确回退CV；轴滤波差分频率输入设0，防止“只用姿态”暗含转速差输入。</p><p>有效像素模型 a = k·s·n_xy − g·image_up − drag·v + residual，s来自相对频率平方和。未来s保持，轴角速衰减。在线更新仅在新到的当前定位满足条件时，比较过去预测与当前CV状态；未使用评分框。k、g、drag和残差都是图像有效量，不能解读成真实质量、推力系数或风速。</p><h2>2. 真正用上物理信息的时刻</h2>'''
    text+=table(['片段','总时刻','预测有效','物理生效','四频内部有效','当帧四桨','允许学习','实际更新','生效时法向向下'],[[r['case'],r['frames'],r['valid_forecast_ticks'],r['feature_active_ticks'],r['four_frequency_ticks'],r['four_visible_ticks'],r['update_allowed_ticks'],r['actual_update_events'],r['axis_points_down_when_active']] for r in d])
    text+='<img src="feature_diagnostics.png"><p>“四频内部有效”含跟踪时效，并不等同当帧四桨都可见。允许更新要求当前检测、四桨、联合质量&gt;0.45，另有0.6秒启动和0.2秒延迟。侧视允许区间太碎，更新队列无法形成到期样本。不能简单认为再飞久一点就会自动学好。</p><div class="note warn">另一个实际缺陷：原轴滤波对近线段/角度跳变继续传播旋转，有时会把图像推力轴带到向下半平面；这与原图像向上先验冲突。轴的几何线方向与有正负号的推力方向需要分开处理。没有相机姿态/三维真值，这里只能指出先验自洽问题，不能给出真实法向误差。</div><h2>3. 全部未来目标评分</h2><p>相同模型覆盖完全一致；差异来自动力学而非删掉难例。选择过去锚点，在目标框确切时间插值已保存前向曲线，实际跨度比名义值最多多25ms(本10ms记录通常更近)。不使用检测器未来输出作参考，不报密集ADE。</p>'
    for case in CASES:
        text+=f'<details'+(' open' if case=='clear6' else '')+f'><summary>{case}</summary>'
        text+=table(['模型','名义s','有效/有机','均值FDEpx','中位px','最大px','联合通过'],[[r['model'],r['nominal_h'],f"{r['valid_forecasts']}/{r['present_targets']}",fmt(r['mean_fde_px']),fmt(r['median_fde_px']),fmt(r['max_fde_px']),r['joint_success']] for r in a['table'] if r['case']==case])+'</details>'
    text+='<img src="future_endpoints.png"><p>绿色叉为存档整机框中心，橙点为当时预报；连线是误差，不是无人机轨迹。本轮10ms源与阶段23的20ms源不同，不能跨阶段均值相减来估计物理收益。</p><h2>4. 当前结论与下一步</h2><p>清晰7点的约1.18px改善，小于现有粗框与中心定义不确定性，不能宣称稳定胜出。姿态+频率20.71px比只姿态21.65px略好；固定初始轴也有21.09px，说明尚未证明精细动态姿态是主要贡献。侧视多数目标回退CV，频率质量与四桨更新门槛先限制了学习。</p><p>下一步先隔离轴向符号/传播退化，给出相同输入的轴滤波对照；再研究不依赖持续四桨的有界学习与在线模型选择。参数估计仍需足够激励；任何放宽准入都应检查错误定位污染，不能把“更新次数多”当作准确。</p><h2>5. 可复核边界</h2><p>解析合成检查：相同CV状态、无特征精确回退、延迟更新、前缀一致、完全中断超时、保存积分曲线范围通过；两段各80时刻前缀重复完全一致。所有预测先写完，再读标签评分。协议冻结后没有按结果改增益。</p><p>缓存溯源发现：清晰阶段12的frequency.py存档指纹与目前脚本不同；其他运行依赖由阶段26链校验。本轮不运行该历史包装脚本，固定输入文件SHA及其原协议，明确记录差异，不能称已重新逐位复现阶段12解码。旧src及结果没有为本轮修改。四频无独立RPM真值，图像轴无独立姿态真值；所有样本已用于开发，不是盲测。</p><p><a href="protocol.json">冻结协议/来源</a>、<a href="analysis.json">逐目标评分</a>、<a href="feature_diagnostics.json">生效/更新诊断</a>、<a href="module_checks.json">检查</a>、<a href="阶段结论.md">结论</a>。无G，未访问执行yolov10-main。</p></main></html>'
    text=text.replace('侧视自适应没有发生任何更新，不能据此判定“在线学习无效”','侧视整段只有7次更新，断轨重置后末段为0，不能据此判定“在线学习无效”')
    text=text.replace('侧视允许区间太碎，更新队列无法形成到期样本','侧视允许区间最长仅80ms，队列只能形成少量到期样本，重置还会丢失既有学习')
    (OUT/'第二十七阶段姿态转速预测贡献报告.html').write_text(text,encoding='utf-8')
    conclusion='# 阶段27：同状态物理贡献\n\n清晰0.4s同7/8目标：CV21.887px，名义姿态+频率23.686，自适应只姿态21.649，姿态+频率20.709，固定初始轴+频率21.089。收益很小且粗框/开发样本不足，未证实稳定优势。侧视同12/14：CV127.617，姿态+频率129.125，无改善。\n\n'+json.dumps(d,ensure_ascii=False,indent=2)+'\n\n侧视学习未发生且原轴滤波有向下传播的先验自洽问题，不能仅用侧视总分裁定物理路径或自适应方向无效。下一步隔离轴滤波，再验证部分观测下更新污染/模型选择。参数是像素有效量；原论文结构独立实现，不是作者复现表格。缓存历史frequency.py指纹差异在协议保留，本轮不重新运行此旧脚本。\n'
    conclusion=conclusion.replace('侧视学习未发生且原轴滤波','侧视整段仅7次更新、末段重置为0，且原轴滤波')
    (OUT/'阶段结论.md').write_text(conclusion,encoding='utf-8');print('Stage27 report rendered',flush=True)


if __name__=='__main__':main()
