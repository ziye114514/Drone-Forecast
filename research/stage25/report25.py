"""Controlled erasure diagnostics, always labelled surrogate reference."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study25 import OUT,CONDITIONS,PARAMS,load,save,verify


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');rows=list(map(json.loads,(OUT/'observations.jsonl').read_text().splitlines()));t=np.array([r['t'] for r in rows])
    assert all(r['reference']==r['conditions'][name]['methods']['neighbor']['observation'] for r in rows if r['t']<PARAMS['erase_start'] for name in CONDITIONS)
    links=[]
    for condition in ['adjacent_0.60','opposite_0.60','dropout_.60']:
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
        for method,color in [('neighbor','#187ba2'),('bounded_layout','#de8024')]:
            errors=[];valid=[]
            for r in rows:
                q=r['conditions'][condition]['methods'][method];o=q['observation']
                errors.append(np.linalg.norm(np.array(o['center'])-r['reference']['center']) if o['detected'] and r['reference']['detected'] else np.nan);valid.append(q['forecast_valid'])
            axes[0].plot(t,errors,label=method,color=color);axes[1].step(t,np.array(valid).astype(float),where='post',label=method,color=color)
        for ax in axes:
            ax.axvspan(PARAMS['erase_start'],PARAMS['erase_start']+.60,color='#b0bbc5',alpha=.2);ax.grid(alpha=.2);ax.legend()
        axes[0].set_ylabel('deviation to unmasked detector [px]');axes[0].set_title(condition+' / artificial erasure, not GT accuracy')
        axes[1].set_ylabel('forecast publication valid');axes[1].set_xlabel('RAW time [s]');axes[1].set_ylim(-.1,1.1)
        name=f'{condition}_timeline.png';fig.savefig(OUT/name,dpi=140);plt.close(fig);links.append(name)
    r=min(rows,key=lambda r:abs(r['t']-23.48))
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,name in zip(axes,['adjacent_0.60','opposite_0.60','dropout_.60']):
        count=np.load(OUT/f'snapshot_{name}.npz')['total'];image=np.uint8(255-230*np.clip(np.log1p(count)/np.log(8),0,1));ax.imshow(image,cmap='gray',vmin=0,vmax=255)
        o=r['reference'];box=o.get('bbox',[0,0,1280,720])
        for method,color in [('neighbor','#187ba2'),('bounded_layout','#de8024')]:
            q=r['conditions'][name]['methods'][method]['observation']
            if q['detected']:
                b=q['bbox'];ax.add_patch(plt.Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,color=color,lw=2));ax.scatter(*q['center'],color=color,marker='x',s=70,label=method)
        ax.set_xlim(max(0,box[0]-60),min(1280,box[2]+60));ax.set_ylim(min(720,box[3]+60),max(0,box[1]-60));ax.set_title(name+' /23.48s');ax.set_xlabel('native x [px]');ax.set_ylabel('native y [px]')
        if name!='dropout_.60':ax.legend(fontsize=9)
    fig.savefig(OUT/'erased_views.png',dpi=135);plt.close(fig)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十五阶段：清晰建模后的受控部分观测</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;margin:0;background:#f3f6fa;color:#243342}main{max-width:1200px;margin:auto;background:white;padding:32px}h1{font-size:30px;line-height:1.3}h2{margin-top:36px;border-bottom:2px solid #dce5eb}.note{padding:18px;background:#fff3de;border-left:5px solid #c68c27}a{color:#176d99}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;white-space:nowrap;padding:9px;border-bottom:1px solid #dce4ec}th{background:#edf2f7}img{max-width:100%;height:auto}</style><main><h1>第二十五阶段：清晰建模后只见部分桨叶</h1><p>2026-10-18 · train6 23–24.8s · 固定相邻/对角事件删减0.12/0.30/0.60s</p><div class="note"><b>这是机制试验，绝不是自然遮挡精度。</b>未删减检测器只负责生成空间事件删除区域和事后参考；各被测分支只收到删减后的事件。参考是未删减检测器的中心，不是独立真值。</div><h2>1. 先清晰获取，再人工移除事件</h2><p>原邻域前端与已有阶段19有期限布局匹配，均先看到相同清晰事件。23.40s起的截止时刻删去指定两桨椭圆附近事件；相邻/对角分别测试三种时长。遮挡区域由未删减检测器的四槽位置跟随生成，缺四点时沿用上一组位置，属于oracle控制损坏，不是实际遮挡生成器。</p><p>另将0.60s所有事件清空，检查模型是否会无观测仍发布。布局300ms完整观测有效期保持，未延长，也没新增配准算法。删减前各条件的原邻域完整观测与未删减参考逐项相同。</p>'''
    text+=table(['条件','方法','删减时刻','当前位置发布','条件参考偏差px','发布且距参考≤20px','布局匹配','预测仍有效'],[[r['condition'],r['method'],r['erasure_ticks'],r['published'],fmt(r['conditional_reference_error_px']),r['within20px_reference'],r['layout_matches'],r['forecast_valid_ticks']] for r in a['table']])
    text+='<h2>2. 观察与边界</h2><p>相邻删减0.12s参考偏差18.69→3.55px，0.30s 17.16→4.18px；0.60s平均12.57→6.08px，但布局本身只匹配前14个时刻，后续是既有状态/分组行为，不能称完整模板一直有效。对角删减两方法约7px，无额外收益；对称可见点本来就更有利于定位整体中心。</p><p>完全中断30个时刻，检测两分支均零发布；预测只在最初7个时刻仍有效，随后超时停止。这支持“一次清晰模型可帮助短时部分观测”，不支持“一次成功后一直成功”。</p><h2>3. 看实际删减画面</h2><img src="erased_views.png"><p>纯原生事件计数；蓝色原邻域、橙色布局。删除可能同时影响机体局部，不声称只删掉真实物理桨盘。只统计这段开发录像，不外推到任意侧视/实际遮挡。</p><h2>4. 完整时间变化</h2>'
    for name in links:text+=f'<img src="{name}">'
    text+='<p>下一步把布局但无周期增强单独放回真实片段，对比是否保留原基底覆盖并改善中心及未来端点。继续保留独立框FDE与缺失，不将本轮参考偏差冒充真实精度。</p><p><a href="protocol.json">协议</a>、<a href="analysis.json">完整表</a>、<a href="module_checks.json">掩膜不变量</a>、<a href="阶段结论.md">阶段结论</a>。旧源码保持，无G/无yolov10-main访问，无物理RPM/姿态真值声明。</p></main></html>'
    (OUT/'第二十五阶段受控部分观测报告.html').write_text(text,encoding='utf-8');print('Stage25 report rendered')


if __name__=='__main__':main()
