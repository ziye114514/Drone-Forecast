"""Window-quality report and rate-normalized native/warped count visualizations."""
import html,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study21 import ROOT,OUT,CASES,METHODS,load,save,verify
from windows21 import select,warp
from event_drone_forecast.raw_io import open_raw,raw_chunks

VISUALS={'clear6':[23.],'side6':[110.,117.28,118.],'review4':[75.,82.96]}


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');links=[]
    for case,stamps in VISUALS.items():
        rr=list(map(json.loads,(OUT/case/'observations.jsonl').read_text().splitlines()))
        for t in stamps:
            r=min(rr,key=lambda r:abs(r['t']-t));assert abs(r['t']-t)<1e-8;cached=OUT/case/f'visual_rate_{t:.2f}.npz'
            if not cached.exists():
                reader=open_raw(CASES[case]['raw']);h,w=reader.get_size();chunks=list(raw_chunks(reader,t-.02,.02,20000))
                assert len(chunks)==1 and abs(chunks[0][0]-t)<1e-8;e=chunks[0][1];arrays={}
                for m,(window,compensate) in METHODS.items():
                    selected=select(e,t,window)
                    if compensate:selected=warp(selected,t,np.array(r['methods'][m]['warp_velocity']),w,h)
                    p=selected['y'].astype(np.int32)*w+selected['x'];arrays[m]=np.bincount(p,minlength=w*h).reshape(h,w).astype(np.float32)/(window/10000)
                np.savez_compressed(cached,**arrays)
            arrays=np.load(cached);h,w=arrays['native20'].shape;points=[]
            for m in METHODS:
                o=r['methods'][m]['observation']
                if o['detected']:points.extend(np.array(o['bbox']).reshape(2,2).tolist())
            if points:
                p=np.array(points);lo=np.maximum(0,np.floor(p.min(0)-[60,50])).astype(int);hi=np.minimum([w,h],np.ceil(p.max(0)+[60,50])).astype(int)
                x0,y0=lo;x1,y1=hi
            else:x0,y0,x1,y1=0,0,w,h
            fig,axes=plt.subplots(1,4,figsize=(18,4.5),layout='constrained')
            for ax,m,color in zip(axes,METHODS,['#1c819b','#d97b1e','#6e56a3','#21996f']):
                grey=np.uint8(255-230*np.clip(np.log1p(arrays[m])/np.log(8),0,1));ax.imshow(grey,cmap='gray',vmin=0,vmax=255)
                o=r['methods'][m]['observation']
                if o['detected']:
                    b=o['bbox'];ax.add_patch(plt.Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,color=color,lw=2));ax.scatter(*o['center'],marker='x',color=color,s=60)
                    for rotor in o['rotors']:ax.scatter(rotor['cx'],rotor['cy'],s=16,facecolors='none',edgecolors=color)
                ax.set_title(f"{m}: published={o['detected']}",fontsize=11);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_xlabel('native x [px]');ax.set_ylabel('native y [px]')
            fig.suptitle(f'{case} {t:.2f}s / event rate per10ms / no intensity reconstruction')
            name=f'{case}_{t:.2f}.png';fig.savefig(OUT/name,dpi=125);plt.close(fig);links.append((case,t,name))
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第二十一阶段：原生时间窗与运动补偿</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6fa;color:#243342;margin:0}main{max-width:1250px;background:white;margin:auto;padding:32px}h1{font-size:31px;line-height:1.3}h2{margin-top:35px;border-bottom:2px solid #dce5eb}.note{padding:18px;background:#fff4df;border-left:5px solid #ca8d27}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;white-space:nowrap;padding:9px;border-bottom:1px solid #dce4ec}th{background:#edf2f6}a{color:#176e99}img{max-width:100%;height:auto}figure{margin:25px 0}figcaption{font-size:14px;color:#637386}</style><main><h1>第二十一阶段：原生5/10/20ms积分与过去速度补偿</h1><p>2026-10-18 · 同35秒自采RAW · 50Hz截止 · 四分支 · 参数先冻结</p><div class="note"><b>没有单一窗口全面胜出。</b>10ms在train4覆盖5/5，比20ms的3/5好；清晰段却8/8降至7/8。5ms在train4出现一次明显错位，warp20也损失覆盖。因此暂不统一替换默认。</div><h2>1. 检测与当前定位</h2><p>四分支使用完全相同的原neighbor/阶段9状态保护，只改变当前事件积分长度；按每10ms事件率归一。warp20以本分支过去速度把事件平移到当前截止，仅最近160ms有实测才启用。没有未来速度、标签或全片统计参与推理。</p>'''
    text+=table(['片段','方法','发布/时刻','有机覆盖','条件中心误差px','联合通过','明显错位','无机标签发布'],[[r['case'],r['method'],f"{r['detected']}/{r['frames']}",f"{r['present_detected']}/{r['present_labels']}",fmt(r['conditional_error_px']),f"{r['joint_success']}/{r['present_labels']}",r['wrong_position'],f"{r['absent_detections']}/{r['absent_labels']}"] for r in a['table']])
    text+='<p>联合通过=有发布且误差≤框对角线25%；明显错位=&gt;50%。train5整段无机，四分支各0/400发布。条件误差不可脱离覆盖解释，例如warp20侧视10.53px对应11/14，并不自动胜过20ms的13/14。</p>'
    text+=table(['四者共同样本','数量',*METHODS],[[r['case'],r['n'],*[fmt(r['errors_px'][m]) for m in METHODS]] for r in a['common']])
    text+='<h2>2. 原生事件画面比较</h2><p>灰度只表示原生ON+OFF事件计数，按每10ms归一以比较不同窗长，不是G灰度重建。每幅显示该方法实际输入的计数/补偿图及其当前位置结果；窗口包含全部方法发布框。</p>'
    for case,t,name in links:text+=f'<figure><a href="{name}"><img src="{name}" loading="lazy"></a><figcaption>{case} {t:.2f}s，来自协议预选时刻。空心点为当前实测桨中心；不表示未来轨迹。</figcaption></figure>'
    text+='<h2>3. 可以和不可以得出的结论</h2><p>短窗在部分重叠/快速运动片段有帮助，但较少的事件也会破坏完整桨盘轮廓；5ms并非越短越好。仅用过去整机速度的平移补偿不能校正旋转、姿态改变、错误位置关联或背景运动；本轮结果不等于否定完整运动补偿方法。</p><p><a href="https://arxiv.org/abs/1804.01306">Gallego等CVPR2018原始框架</a>通过扭曲事件对齐处理运动；本轮只复用项目现有的过去速度平移公式，没有执行对比度参数优化或声称完整复现。</p><p>接下来用同一批真实稀疏框评价未来位置端点：检测覆盖与中心误差的取舍是否传递到0.1/0.2/0.4s预测，不能靠当前定位均值推断。稀疏框不足以算完整轨迹ADE，但可在已标注目标时刻选择较早锚点评估端点；这一口径比此前“FDE也不能算”的说法准确。</p><h2>4. 完成性与局限</h2><p>native20的1750/1750完整观测与阶段19neighbor20缓存相同，真实RAW前缀各方法一致；四段全部完成。输入边界、解析补偿位移、零速、越界/未来点检查通过。原始评分及共同样本见<a href="analysis.json">analysis.json</a>，配置和指纹<a href="protocol.json">protocol.json</a>，<a href="阶段结论.md">阶段结论</a>。</p><p>这些片段已用于开发，train4也非盲测；标签是近似整机框中心而非质心/姿态/RPM真值。旧src和默认保持，未使用G、未访问执行yolov10-main，无未来预测精度提升声明。</p></main></html>'
    (OUT/'第二十一阶段原生时间窗报告.html').write_text(text,encoding='utf-8');print('Stage21 report rendered',flush=True)


if __name__=='__main__':main()
