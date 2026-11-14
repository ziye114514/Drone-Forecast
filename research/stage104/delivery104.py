"""Plots, native-event video overlays and a self-contained research report."""
from study104 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS=dict(clear6='清晰段',side6='侧视段',train3='3号录像')


def plots():
    sim=load(OUT/'synthetic104.json');fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,case,title in zip(axes.flat,['stationary_wobble','translating_wobble','abrupt_turn_wobble','edge_on_missing'],
            ['Fixed center, tilting rotor plane','Translation + tilt','Direction change + tilt','Edge-on / missing rotors']):
        group=[r for r in sim['records'] if r['case']==case and 'body_displacement_error_px' in r]
        ax.plot([r['t'] for r in group],[r['body_displacement_error_px'] for r in group],label='Appearance-box center')
        ax.plot([r['t'] for r in group],[r['corrected_displacement_error_px'] for r in group],label='Planar-center correction')
        ax.set(title=title,xlabel='Source time (s)',ylabel='Displacement error (px)');ax.grid(alpha=.25)
    axes[0,0].legend();fig.suptitle('Known geometry, 0.3 px contour noise; initial offset aligned once')
    fig.savefig(OUT/'synthetic_controls104.png',dpi=160);plt.close(fig)
    matched=load(OUT/'matched_controls104.json');scores=load(OUT/'scores104.json');fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,case in zip(axes,CASES):
        def val(name):return next(r['mean_px'] for r in matched['table'] if r['case']==case and r['h']==.4 and r['method']==name)
        old=next(r['mean_px'] for r in scores['table'] if r['case']==case and r['h']==.4 and r['method']=='complement_fallback')
        v=[val('body_only_kf'),val('corrected_kf'),old]
        ax.bar(['Matched\nbody KF','Planar\ncenter KF','Original\nfusion'],v,color=['#a0a8b0','#179b86','#7758ad'])
        ax.set_ylim(0,max(v)*1.14)
        ax.set(title=case,ylabel='0.4 s endpoint error (px)');ax.grid(axis='y',alpha=.2)
        for i,y in enumerate(v):ax.text(i,y+.8,f'{y:.2f}',ha='center')
    fig.suptitle('Sparse box-center targets: matched filter and original fusion comparison')
    fig.savefig(OUT/'forecast_comparison104.png',dpi=160);plt.close(fig)
    data=load(OUT/'cached/clear6/forecasts.json');fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
    tt=np.array([q['t'] for q in data]);diag=[q['rigid_center_diagnostic'] for q in data]
    axes[0].plot(tt,[d['weight'] for d in diag],label='Center correction weight');axes[0].plot(tt,[d['current'] for d in diag],alpha=.4,label='Current four-conic center');axes[0].legend();axes[0].set_ylabel('Admission')
    cc=np.array([d['correction_px'] for d in diag]);axes[1].plot(tt,cc[:,0],label='x correction');axes[1].plot(tt,cc[:,1],label='y correction');axes[1].legend();axes[1].set_ylabel('Offset (px)')
    for kind in ['event','constant']:
        axes[2].plot(tt,[d['gains'].get(kind,{}).get('gain_px_s2',0.) for d in diag],label=kind)
    axes[2].legend();axes[2].set(xlabel='Source time (s)',ylabel='Effective gain (px/s2)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('clear6: actual admitted layout updates and separately learned effective gains')
    fig.savefig(OUT/'model_diagnostics104.png',dpi=160);plt.close(fig)


def video(case):
    p=OUT/'raw'/case/'forecasts.mp4'
    if not p.exists():return None
    data=rows(OUT/'raw'/case/'observations_forecasts.jsonl');cap=cv2.VideoCapture(str(p))
    target=OUT/'raw'/case/'planar_center_forecasts.mp4';writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),50.,(1280,816));assert writer.isOpened()
    count=0
    while True:
        ok,image=cap.read()
        if not ok:break
        r=data[count];q=r['frontends']['delayed_feedback'];d=q['rigid_center_diagnostic']
        if d['virtual_hubs'] is not None:
            hubs=np.asarray(d['virtual_hubs']);center=d.get('model_geometric_center')
            if center is None:
                order=cv2.convexHull(hubs.astype(np.float32),returnPoints=False).reshape(-1)
                center,_=geometry.intersection(hubs[order])
            else:center=np.asarray(center)
            if np.isfinite(hubs).all() and abs(hubs).max()<10000:
                for point in hubs:
                    point=tuple(np.int32(point));cv2.circle(image,point,9,(0,210,255),2)
                    cv2.line(image,tuple(np.int32(center)),point,(0,210,255),1)
                cv2.circle(image,tuple(np.int32(center)),7,(0,210,255),2)
        if d.get('center') is not None:cv2.drawMarker(image,tuple(np.int32(d['center'])),(255,255,0),cv2.MARKER_CROSS,15,2)
        for name,color in [('complement_fallback',(190,80,210)),('rigid_center_kf',(255,255,0))]:
            if name in q['coefficients']:
                points=[q['coefficients'][name]['position']]+[q['predictions'][name][str(h)] for h in [.1,.2,.4]]
                points=np.asarray(points)
                if np.isfinite(points).all() and abs(points).max()<10000:
                    cv2.polylines(image,[np.int32(points)],False,color,2);cv2.circle(image,tuple(np.int32(points[-1])),5,color,-1)
        canvas=np.full((816,1280,3),(20,28,38),np.uint8);canvas[:720]=image;image=canvas
        cv2.putText(image,f"104 {case} t={r['t']:.2f} | layout={d['mode']} weight={d['weight']:.2f}",(12,747),0,.56,(240,240,240),1)
        cv2.putText(image,'cyan=corrected-center KF | purple=original fusion | gold=layout (hidden hubs are predictions)',(12,777),0,.52,(240,240,240),1)
        cv2.putText(image,'Native 1280x720 ON/OFF image above is uncropped. Source-time playback; not real-time inference.',(12,802),0,.5,(200,210,220),1)
        if count in [len(data)//4,len(data)//2,3*len(data)//4]:cv2.imwrite(str(OUT/f'{case}_example_{count}.png'),image)
        writer.write(image);count+=1
    cap.release();writer.release();assert count==len(data)
    check=cv2.VideoCapture(str(target));decoded=0
    while True:
        ok,image=check.read()
        if not ok:break
        assert image.shape==(816,1280,3);decoded+=1
    check.release();assert decoded==count
    return dict(case=case,frames=count,all_frames_decoded=True,video=target.relative_to(OUT).as_posix())


def table(headers,values):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in values)+'</tbody></table>'


def main():
    verify();plots();checks=[video(c) for c in ['clear6','side6','negative5']];checks=[c for c in checks if c]
    sim=load(OUT/'synthetic104.json');cached=load(OUT/'cached_checks104.json');scores=load(OUT/'scores104.json');matched=load(OUT/'matched_controls104.json')
    rawchecks=[load(p) for p in sorted((OUT/'raw').glob('*/checks104.json'))]
    if len(rawchecks)!=4:raise RuntimeError('All four planned actual RAW checks required before delivery')
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    def m(case,name):return next(r for r in matched['table'] if r['case']==case and r['h']==.4 and r['method']==name)
    def s(case,name):return next(r for r in scores['table'] if r['case']==case and r['h']==.4 and r['method']==name)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>104：机体中心与四桨姿态分离</title><style>body{max-width:1220px;margin:30px auto;padding:0 18px;font:17px/1.8 "Microsoft YaHei",Arial;color:#24394b;background:#f9fbfc}h1,h2{line-height:1.4}section{padding:18px 22px;background:white;border:1px solid #dae3e8;border-radius:12px;margin:22px 0}img,video{width:100%;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d7e1e8;padding:7px;text-align:left}th{background:#eef4f7}.lead{border-left:5px solid #159b86;padding:12px 20px;background:#eaf7f3}code{background:#eef3f5;padding:2px 5px}a{color:#195aa1}.warn{background:#fff6e5}</style><h1>104：把机体中心平移与四桨投影变化分开</h1><div class="lead"><b>已经接成可运行的 RAW 实验分支，仿真通过，实拍清晰段能建立模型；尚未超过原融合预测，也没有解决长时间侧视。</b><p>原93的55路完整保留，新加5路，默认54不替换。输入仍为RAW和原项目权重，不使用G通道；E盘和yolov10-main均未修改。</p></div>'''
    page+='<section><h2>本次真正接入了什么</h2><p>复用58的四共面圆锥曲线联合拟合，分别估计机架几何中心与四桨投影布局。连续5个合格当前观测初始化，半径/机架面积尺度比固定为初始化值，并检查后续一致性。中心第一次对齐当时整机框，此后初始图像偏置固定；它是相对位置对照，不是重心标定，也不是已知尺寸的3D建模。</p><p>清晰时用当前四桨联合中心，按四点扁平程度平滑降权；侧视/缺桨时保留最后中心校正160ms，300ms归零，500ms无合格四桨则重新初始化。当前整机位置仍更新滤波器，不把无人机强行设为静止。虚拟桨心只画图，不作为位置或频率测量。</p><p>另加独立有效力分支：椭圆法向与已连续确认的整机线法向按视角条件混合，利用原周期特征，在校正中心上另学增益。恒定强度、零修正控制同在；不是把事件密度直接当成物理推力或真实RPM。</p></section>'
    page+='<section><h2>固定中心、真实平移、突然转向：受控验证</h2><p>人为针孔相机、对称矩形机架与等半径共面圆；轮廓加入0.3px噪声。误差是相对首次锁定后的中心位移，不能解释为实拍绝对重心精度。静止不是算法输入。</p>'+table(['场景','初始化','原框中心位移RMSE px','校正后位移RMSE px','真实首末位移 px'],[[r['case'],r['initialized'],fmt(r['body_displacement_rmse_px']),fmt(r['corrected_displacement_rmse_px']),fmt(r['end_displacement_px'])] for r in sim['summary']])+'<img src="synthetic_controls104.png"><p>侧视缺桨控制在几何过期后回到整机中心，误差再次增大；这项失败保留，不能称“一次建模后永远成功”。非共圆干扰控制未初始化，但单个合成负例不等于实际误报率保证。</p></section>'
    page+='<section><h2>你的自采数据：模型实际启用程度</h2>'+table(['片段','20ms时刻','四桨中心更新','缺桨/侧视短时保持','完全回退','最大校正 px','有效力启用'],[[LABELS[r['case']],r['ticks'],r['current'],r['modes'].get('side_body_hold',0)+r['modes'].get('partial_body_hold',0),r['modes'].get('body_fallback',0),fmt(r['max_correction_px']),r['force_current']] for r in cached])+'<p>清晰段真实RAW重新运行250时刻、侧视段400时刻；3号为已有RAW当前特征的因果缓存复算450时刻。侧视段与3号没有成功初始化四桨模型；它们的有效力输出不能归功于骨架中心。三段不是彼此独立的新测试集。</p><img src="model_diagnostics104.png"></section>'
    page+='<section><h2>0.4秒预测：相同输入控制与原完整融合</h2><p>沿用既有稀疏人工整机框中心作为未来参考。先生成预测，再读参考评分。所有方法只比较相同可用锚点；没有提供真实机架中心、姿态或转速真值。</p>'+table(['片段','有效/全部目标','相同body-only KF','校正中心KF','相同body-only历史','校正中心历史','新有效力','原完整融合'],[[LABELS[c],f"{m(c,'body_only_kf')['n']}/{m(c,'body_only_kf')['targets']}",*[fmt(v) for v in [m(c,'body_only_kf')['mean_px'],m(c,'corrected_kf')['mean_px'],m(c,'body_only_history')['mean_px'],m(c,'corrected_history')['mean_px'],s(c,'rigid_center_view_force')['mean_px'],s(c,'complement_fallback')['mean_px']]]] for c in CASES])+'<img src="forecast_comparison104.png"><p class="warn">新中心KF的清晰段变化需与相同body-only KF比较；原kinematic_train6400含几何补位，不能把位置来源改变算作中心模型收益。侧视/3号的校正KF与body-only控制逐项完全相同，因为根本没建模。新分支仍未超过原完整融合，不升默认。</p>'+table(['片段','新频率有效力','恒定强度控制','零修正控制'],[[LABELS[c],*[fmt(s(c,n)['mean_px']) for n in ['rigid_center_view_force','rigid_center_constant_force','rigid_center_zero_force']]] for c in CASES])+'<p>频率分支与恒定/零修正差异很小或相等，不能称已经验证转速带来的独立预测收益。</p></section>'
    page+='<section><h2>实际RAW视频：模型与预测同时看</h2><p>青色为校正中心KF预测，紫色为原融合预测；金色是机架图形。缺桨时金色点是短时模型推测，绝不是观测真值。50fps仅为源视频时间轴。</p>'
    for item in checks:page+=f'<h3>{html.escape(item["case"])}</h3><video controls preload="metadata" src="{item["video"]}"></video>'
    page+='</section><section><h2>当前真正卡在哪里</h2><ol><li><b>观察点定义不一致。</b>几何中心、机体重心和可见框中心不等同。目前只能验证框中心预测，无法独立证明真实机体中心更准。</li><li><b>模型输入仍会错。</b>实拍桨盘受拖尾、局部弧段、遮挡、镜头畸变影响，等半径共面圆模型只是近似。低拟合残差不是正确转子身份的证明。</li><li><b>侧视信息不足。</b>两条对角线近乎重合，四桨中心解噪声放大。当前侧视片段从退化状态起步，没有一个可靠四桨初始化，不能凭旧片段隔几十秒强行续锁。</li><li><b>更换位置观测会改变速度/加速度。</b>清晰段校正中心历史外推反而变差，说明还需研究中心不确定度和模式交接，不能把仿真几何改善直接当预测改善。</li><li><b>还不实时。</b>本轮保留原55路并加拟合/新预测。完整RAW单步仍远高于20ms；并发运行也影响耗时，不能把前后两次非配对耗时当严格加速比。</li></ol><p>下一步应在同一连续近景录像中验证“正视建模→侧视短时缺桨→恢复”，优先改观测不确定度和状态交接，再考虑接管原预测。先不继续堆叠推力参数或宣布创新。</p></section>'
    page+='<section><h2>复用依据与完整性</h2>'+''.join(f'<p><a href="{r["url"]}">{html.escape(r["title"])}</a>：{html.escape(r["use"])}</p>' for r in REFERENCES)
    page+='<p>初始缓存前缀校验曾误比拟合耗时；预测数值已相同。仅去掉诊断中的processing_ms/elapsed_ms后重查，原源码及补充说明保留。首次RAW主循环成功，其外层旧版哈希校验因该校验器修订失败，随后对同份RAW结果重新校验通过，未重算或挑选结果。</p><p><a href="protocol104.json">冻结协议</a> · <a href="validation_amendment104.json">校验修订</a> · <a href="synthetic104.json">全部受控结果</a> · <a href="cached_checks104.json">缓存/前缀检查</a> · <a href="matched_controls104.json">相同输入控制</a> · <a href="scores104.json">全部未来评分</a> · <a href="delivery_checks104.json">实际RAW及视频验证</a></p></section></html>'
    report=OUT/'第一百零四阶段机体中心与四桨姿态分离报告.html';report.write_text(page,encoding='utf-8')
    save(OUT/'delivery_checks104.json',dict(raw=rawchecks,videos=checks,synthetic_checks_passed=True,
        cached_prefixes_exact=True,original55_preserved=True,default_promoted=False,
        report=str(report),delivery_source_sha256=sha(__file__),scope='Actual RAW650 full ticks +90 truncation/negative ticks; cached third recording450. No full end-to-end improvement or real-time claim.'))
    print(str(report),flush=True)


if __name__=='__main__':main()
