"""Deliver106-108 tests, all failures, optional human review and handoff."""
from study107 import *
import html,collections,zipfile
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DIAG108=ROOT.parent/'output/stage108_current20_geometry_20261022'


def plots():
    scores=load(OUT/'scores107.json');fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,case in zip(axes,['clear6','side6']):
        methods=['center_body_kf107','center_smooth_kf107','center_bias_fixed107','center_bias_observable107','complement_fallback']
        values=[next(r['mean_px'] for r in scores['table'] if r['case']==case and r['h']==.4 and r['method']==m) for m in methods]
        ax.bar(['Body\nKF','Simple\nsmoothing','Bias\nfixed','Bias\nview-aware','Original\nfusion'],values,color=['#9aa5b0','#83a8bd','#169c85','#d58b42','#7957a2'])
        ax.set(title=case,ylabel='0.4 s endpoint error (px)',ylim=(0,max(values)*1.15));ax.grid(axis='y',alpha=.2)
        for j,v in enumerate(values):ax.text(j,v+max(values)*.025,f'{v:.2f}',ha='center',fontsize=9)
    fig.suptitle('Same sparse BOX-center targets: new geometry/bias gains do not beat original fusion')
    fig.savefig(OUT/'forecast_comparison107.png',dpi=160);plt.close(fig)
    sim=load(OUT/'synthetic107.json');fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,case in zip(axes,['stationary_wobble','long_side_gap']):
        rr=next(g['rows'] for g in sim['records'] if g['case']==case)
        for method in NAMES:
            ax.plot([r['t'] for r in rr],[np.linalg.norm(r['output']['coefficients'][method]['velocity']) for r in rr],label=method.replace('center_','').replace('107',''))
        ax.set(title=case,xlabel='Time (s)',ylabel='Estimated speed (px/s)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Measurement-layer synthetic controls, stationary true center; not hardware event results')
    fig.savefig(OUT/'synthetic_speeds107.png',dpi=160);plt.close(fig)
    original=rows(INPUTS['clear6']);fixed=load(OUT/'cached/clear6/forecasts.json');fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
    tt=np.array([r['t'] for r in original]);body=np.array([r['body_observation'].get('center') or [np.nan,np.nan] for r in original])
    for j,label in [(0,'x'),(1,'y')]:
        axes[j].plot(tt,body[:,j],label='Current box center',alpha=.6)
        for method in ['center_bias_fixed107','center_bias_observable107']:
            values=[r['coefficients'].get(method,{}).get('position',[np.nan,np.nan])[j] for r in fixed]
            axes[j].plot(tt,values,label=method.replace('center_','').replace('107',''))
        axes[j].set_ylabel(f'{label} (px)');axes[j].legend(fontsize=8)
    for j,label in [(0,'x bias'),(1,'y bias')]:
        axes[2].plot(tt,[r['diagnostics']['fixed'].get('appearance_bias_px',[0,0])[j] for r in fixed],label=label)
    axes[2].set(xlabel='Source time (s)',ylabel='Appearance bias (px)');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('clear6: unified dynamic state and separately estimated appearance bias')
    fig.savefig(OUT/'center_and_bias107.png',dpi=160);plt.close(fig)


def video(case):
    path=OUT/'raw'/case/'forecasts.mp4'
    if not path.exists():return None
    data=rows(OUT/'raw'/case/'observations_forecasts.jsonl');cap=cv2.VideoCapture(str(path));target=path.with_name('center_bias_forecasts.mp4')
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),50.,(1280,816));assert writer.isOpened();n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        row=data[n];q=row['frontends']['delayed_feedback'];d=q['center_bias_diagnostic']['observable']
        for name,color in [('center_bias_fixed107',(255,255,0)),('center_bias_observable107',(0,130,255)),('complement_fallback',(190,80,210))]:
            c=q['coefficients'].get(name)
            if c is None:continue
            from hybrid74 import prior
            points=np.asarray([prior.evaluate(c,h) for h in [0.,.1,.2,.4]])
            if np.isfinite(points).all() and abs(points).max()<10000:
                cv2.polylines(im,[np.int32(points)],False,color,2);cv2.circle(im,tuple(np.int32(points[-1])),5,color,-1)
        canvas=np.full((816,1280,3),(20,28,38),np.uint8);canvas[:720]=im
        cv2.putText(canvas,f"107 {case} t={row['t']:.2f} | geometry reference={d.get('has_geometry_reference',False)} current={d.get('current_geometry',False)}",(12,748),0,.54,(235,235,235),1)
        cv2.putText(canvas,'cyan=fixed bias model | orange=view-aware (failed side control) | purple=original fusion',(12,775),0,.51,(235,235,235),1)
        cv2.putText(canvas,'Original native image above is uncropped. Source-time playback; inference is not real-time.',(12,803),0,.49,(200,210,220),1)
        writer.write(canvas)
        if n==len(data)//2:cv2.imwrite(str(OUT/f'{case}_midpoint107.png'),canvas)
        n+=1
    cap.release();writer.release();assert n==len(data)
    cap=cv2.VideoCapture(str(target));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(816,1280,3);decoded+=1
    cap.release();assert decoded==n
    return dict(case=case,frames=n,decoded=True,path=target.relative_to(OUT).as_posix())


def table(headers,values):
    return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in values)+'</table>'


def main():
    verify();raw=[load(p) for p in sorted((OUT/'raw').glob('*/checks107.json'))];assert len(raw)==4
    plots();videos=[video(c) for c in ['clear6','continuous6','negative5']];videos=[r for r in videos if r]
    screen=load(SCREEN/'screen106.json');geo=load(DIAG108/'streak_diagnostic108.json');human=load(OUT/'visibility_review/human_visibility107.json')
    scores=load(OUT/'scores107.json');sim=load(OUT/'synthetic107.json');cache=load(OUT/'cached_checks107.json')
    def score(case,method):return next(r for r in scores['table'] if r['case']==case and r['h']==.4 and r['method']==method)
    names=['center_body_kf107','center_smooth_kf107','center_bias_fixed107','center_bias_observable107','rigid_center_kf','complement_fallback']
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>106—108：连续筛查与中心偏移滤波</title><style>body{max-width:1220px;margin:30px auto;padding:0 18px;font:17px/1.8 "Microsoft YaHei",Arial;color:#24394b;background:#f8fafc}section{padding:18px 22px;background:white;border:1px solid #dce4eb;border-radius:12px;margin:22px 0}img,video{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border:1px solid #d7e1e8;text-align:left}th{background:#eef4f8}.lead{border-left:5px solid #169c85;padding:15px;background:#eaf8f3}.warn{padding:12px;background:#fff1dd}a{color:#175da6}</style><h1>106—108：连续RAW筛查与中心/外观偏移分离</h1><div class="lead"><b>已实现并真实运行：固定噪声的中心偏移模型在清晰段小幅改善；侧视降权候选反而退步。默认方案保留。</b><p>最新实验入口run_center_bias_forecast.py，原104全部60路保留，新增4条候选。实际RAW、真实截断与负例均完成；没有靠放宽初始化门槛凑成功。</p></div>'''
    page+='<section><h2>现在实现的滤波逻辑</h2><p>复用固定提交的FilterPy作者predict/update原函数，状态为[cx,cy,vx,vy,bx,by]。框中心观测c+b，当前合格四桨的几何中心约束c；b表示外观偏移。首次锁定对齐一次图像坐标，随后即使104机架过期/重建也保留这个坐标偏置，不因视角改变直接重置中心或速度。</p><p>对照包括body-only KF、100ms因果校正平滑、固定噪声中心偏移KF、按几何条件和侧视方向调整噪声的同状态KF。位置/运动噪声沿用既有参数；新增噪声与相关系数是冻结的假设，协方差未校准。框与桨盘来自相同事件，联合更新保留假定相关项，不将二者称独立真值。</p><p>侧视候选仅借线方向放大沿线框中心的测量噪声，没有从线端点伪造桨心。它的失败恰好提醒：桨盘姿态退化，不意味着当前整机框位置也同等不可靠。</p></section>'
    page+='<section><h2>清晰段0.4秒预测与失败对照</h2><p>下表沿用相同稀疏人工整机框中心端点，每段7/8个有效目标；不是独立机体重心/姿态真值，不是新盲测。</p>'+table(['方法','清晰 px','侧视 px'],[[n,f"{score('clear6',n)['mean_px']:.3f}",f"{score('side6',n)['mean_px']:.3f}"] for n in names])+'<img src="forecast_comparison107.png"><p>固定偏移模型清晰18.537优于简单平滑22.236及104中心KF19.106，但仍差于原完整融合13.708。侧视没有四桨初始化，固定偏移退回body-only的81.723；按侧视方向降权变为94.086，反而更差。不能把这项失败包装为侧视恢复成功。</p><img src="center_and_bias107.png"></section>'
    page+='<section><h2>受控测量层实验</h2><p>已知中心/外观偏移与噪声，不是事件相机光电仿真，也没有跑真实检测器。含静止摆动、平移、突然转向、长侧视缺失、无几何、错误几何。所有方法共享一次对齐的图像偏置，报告相对图像状态。稳定时的改善不能保证缺失期间正确。</p><img src="synthetic_speeds107.png">'+table(['场景','body KF位置RMSE','固定偏移模型位置RMSE','视角噪声模型位置RMSE'],[[c,*[f"{next(r['position_rmse_px'] for r in sim['summary'] if r['case']==c and r['method']==m):.3f}" for m in [NAMES[0],NAMES[2],NAMES[3]]]] for c in ['stationary_wobble','translation_wobble','abrupt_turn','long_side_gap','no_geometry','accepted_bad_geometry']])+'<p>无几何时固定偏移模型退化为普通KF。一个被错误接纳的几何测量仍能产生速度尖峰，长侧视期间位置偏差仍大；不能靠一个模型消除缺失信息。原4D与扩展6D等价控制有约4.55e-13浮点差异，校验仅对这项改为1e-8容差；同维度因果前缀仍严格相等。原源码、协议和校验适配记录保留，推理/参数未修改。</p></section>'
    page+='<section><h2>连续110—119.5秒：问题实际出现在哪里</h2><p>事先按105选定的固定窗口，原104完整RAW运行475父/950原生时刻。当前body有效370/475、线方向有效353/475，但四桨模型初始化0次，缺少“成功建模→侧视→恢复”完整过程。后续107在同一RAW重新运行，原60路逐项精确；这两遍是同一数据，不能加成独立样本。</p><img src="../stage106_continuous_screen_20261022/fixed_contact106.png"><p>108只改为使用已到达的当前20ms桨盘候选，原104拟合与初始化门槛全部不变。四候选83次（10ms为42），合格拟合14次（10ms为2），最长连续3帧，仍低于5帧，因此0初始化。它是当前特征的缓存几何诊断，不是新的RAW部署或预测提升。</p>'+table(['输入','四候选时刻','合格拟合','最长连续合格','初始化'],[['10ms当前跟踪',42,2,'未重新统计',0],['20ms当前观测',83,14,geo['maximum_consecutive_qualified'],0]])+'<p class="warn">先修复可见桨盘的漏检/合并，再考察初始化证据积累；没有因为看到最长3帧就把5帧门槛改成3。</p></section>'
    page+='<section><h2>人工核验已经提供的线索</h2><p>用户直接反馈：A三个桨盘，C四个桨盘，B/D/F基本成线，E部分重叠。这只核验可见性，不是像素桨心、真实COM、姿态或RPM真值，未进入预测器。</p><a href="visibility_review/六帧桨盘可见性核验.html"><img src="visibility_review/A_F_visibility_review.png"></a><p>C（112.02秒）人能分出四盘，但当前10ms和20ms分支都只有三个候选，说明这帧至少值得重点检查漏检或轮廓合并；A只有三个可见桨盘，不能拿来强行初始化四桨。下一步应逐层保存C附近的原生极性支持、阈值连通域、轮廓、弧段和椭圆拒绝原因，追踪第四盘在哪一步丢失。用这次核验选择的片段是开发数据，不再称盲测。</p></section>'
    page+='<section><h2>实际RAW预测视频</h2><p>青色固定偏移模型、橙色视角噪声候选、紫色原融合。信息栏在原生画面外，50fps仅源时间播放，仍不代表实时推理。</p>'
    for v in videos:page+=f'<h3>{v["case"]}</h3><video controls preload="metadata" src="{v["path"]}"></video>'
    page+='</section><section><h2>下一步与当前边界</h2><ol><li>优先查C附近第四盘的提取链路；区分遮挡、连通域合并、阈值和椭圆拒绝，不用虚拟点补成测量。</li><li>分别估计姿态、几何中心与整机框定位的可靠性；当前侧视降权失败，不替换默认。</li><li>只有连续建模/退化/恢复真正出现后，再测新偏移状态是否能保持连续并改善未来位置。当前固定模型仍未胜过原完整融合。</li><li>本轮不调整推力参数，也没有新机械RPM收益；未知质量/风等仍不可分别辨识。实际整链路远高于20ms，尚不实时。</li></ol><p>本轮用户已完成所需可选核验，暂不需要补采飞控日志或更多FRED。E盘保持可读即可；后续若需要少量标注，会先提供明确关键帧和用途。</p></section>'
    page+='<section><h2>复用、验证与交接</h2><p><a href="https://github.com/rlabbe/filterpy">FilterPy作者代码</a>固定提交3b51149，MIT许可与完整源码保留，实际执行原predict/update/reshape_z函数；无新依赖安装。<a href="https://github.com/Zibin6/LOPET">LOPET</a>只作事件线匹配文献参考，本轮没有运行其MATLAB系统。</p><p>107真实RAW四项共815父/1630原生（清晰250、连续475、独立截断60、无机30），原60路与新4路缓存严格对照，独立真实前缀通过，负例无预测；三个新叠加视频755帧全解码。106另475是同一连续数据重跑；108只复算当前缓存。默认54和API93不变。</p><p><a href="protocol107.json">107冻结协议</a> · <a href="scores107.json">完整评分</a> · <a href="synthetic107.json">全部受控结果</a> · <a href="synthetic_validation_adapter107.json">校验适配说明</a> · <a href="cached_checks107.json">缓存前缀</a> · <a href="visibility_review/human_visibility107.json">用户核验记录</a> · <a href="delivery_checks107.json">RAW/视频检查</a> · <a href="../stage106_continuous_screen_20261022/screen106.json">连续筛查</a> · <a href="../stage108_current20_geometry_20261022/geometry108.json">20ms全部诊断</a></p></section></html>'
    report=OUT/'第一百零六至一百零八阶段连续观测与中心滤波报告.html';report.write_text(page,encoding='utf-8')
    save(OUT/'delivery_checks107.json',dict(raw=raw,videos=videos,actual_RAW_parent=sum(r['parent'] for r in raw),
        actual_RAW_native10=sum(r['native10'] for r in raw),video_frames_decoded=sum(r['frames'] for r in videos),
        original60_preserved=True,new4_exact_cached=True,true_prefix_verified=True,source_protocol_verified=True,
        default_promoted=False,report=str(report),delivery_source_sha256=sha(__file__),human_review_received=True,
        entire_objective_complete=False,stable_realtime=False))
    print(str(report),flush=True)


if __name__=='__main__':main()
