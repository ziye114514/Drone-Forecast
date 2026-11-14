"""Extend93 evidence report with versioned near-target native experiments."""
from pathlib import Path
import html,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import deliver_stages88_93 as base
ROOT=base.ROOT;WORK=ROOT.parent;OUT=base.OUT
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage88'))
from inventory88 import save,load,sha


def main():
    base.main()
    folders={s:WORK/f'output/{name}_20261022' for s,name in [(94,'stage94_official_large_target'),(95,'stage95_official_visible_interval'),(96,'stage96_native_frequency_input'),(97,'stage97_projected_axis_magnitude')]}
    scores={s:load(folders[s]/f'scores{s}.json') for s in [95,96,97]}
    metrics={s:base.rowmap(scores[s],'fred194') for s in scores}
    methods=['cv','trusted_acc','event','constant','zero','lag200','kf_tuned','fallback_router','full_router']
    labels=['CV','History','Event','Constant','Zero','Lag200','TrainKF','Fallback','Full mix']
    fig,ax=plt.subplots(figsize=(12,4.5));x=np.arange(len(methods));width=.25
    for j,(s,label) in enumerate([(95,'FFT + unit projected direction'),(96,'SDK relative frequency'),(97,'Projected-magnitude ablation')]):
        ax.bar(x+(j-1)*width,[metrics[s][m]['mean_fde'] for m in methods],width,label=label)
    ax.set_xticks(x,labels,rotation=22);ax.set_ylabel('0.4s endpoint error (px)');ax.set_title('Official194 visible interval:23 overlapping anchors, past-box diagnostic');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'official194_fixed_controls.png',dpi=165);plt.close(fig)
    near_rows=[]
    for s in scores:
        complete=load(folders[s]/'fred/sequence194/completion.json')
        near_rows.append([s,f"{complete['current_physics_ticks']}/{complete['ticks']}",*[base.num(metrics[s][m]['mean_fde']) for m in ['event','constant','zero','kf_tuned','fallback_router']]])
    own=load(folders[97]/'self_scores97.json');old=load(base.B88/'scores88.json');ownrows=[]
    for c in base.CASES:
        a=base.rowmap(old,c);b=base.rowmap(own,c)
        ownrows.append([c,base.num(a['event']['mean_fde']),base.num(b['event']['mean_fde']),base.num(b['constant']['mean_fde']),base.num(b['zero']['mean_fde'])])
    d94=load(folders[94]/'download94.json');p95=load(folders[95]/'protocol_feature95.json')
    section=f'''<section><h2>94—97：当前官方近景194号与物理模型差异</h2>
    <p>已从固定官方版本仅提取194号原生HDF5事件，实际下载{d94['compressed_bytes']/1e6:.1f}MB，展开{d94['native_bytes']/1e6:.1f}MB，ZIP CRC及SHA已核验；没有下载RGB/事件PNG整包。数据位于 <code>{html.escape(d94['path'])}</code>，不需要重复下载194号事件。</p>
    <p>94先固定57—67秒中段：有事件但没有位置标注，按规则不发布预测，不能评价精度，也不能据此宣称检测零误报。保留此空段。95随后只按标注连续可见时长选最长区间，冻结{p95['start_s']:.2f}—{p95['start_s']+p95['duration_s']:.2f}秒，再提取事件。该区间属于可见性/尺寸条件下的小试验，非盲测、非完整官方成绩。</p>
    <p>95共167个20ms时刻、约1041万原生事件、100次过去位置更新；原FFT合格93次、SDK strict7合格107次，几何支持122次。各方法0.4秒均有23个有效锚点，锚点高度重叠，不能当23次独立飞行。对照仍不能证明真实周期比常量、零修正或打乱更有用。</p>
    {base.table(['实验','当前物理支持','真实周期FDE','恒定FDE','零修正FDE','训练KF','新回退融合'],near_rows)}
    <img src="official194_fixed_controls.png">
    <p>96只将相对频率来源换成已通过人工控制的SDK strict7，保持RLS、方向和所有对照；其候选权重适配器的中性1不代表测到相位。SDK提高启用覆盖，仍未胜过匹配常量/零修正，不接入默认推力。</p>
    <p>97按原论文推力公式补查投影幅值：3D单位法线的二维投影通常不为单位长。旧69只保留二维方向，97保留已有弱透视法线的xy长度，侧视机体线仍假设幅值1。所有分支共享原单位方向的发布时钟，避免覆盖变化伪装收益。弱透视近似仍不等于标定姿态；该改动是论文一致性诊断，不是原创方法。</p>
    {base.table(['自采片段','原单位方向','保留投影幅值','匹配恒定','匹配零修正'],ownrows)}
    <p>97未直接升级RAW93默认或生产模型。所有额外失败/恒定/打乱结果见各scores文件，不按片段选择不同模型拼成一个成绩。</p>
    <h3>官方194号可见区间（原生事件＋过去提供框）</h3>
    <video controls preload="metadata" src="../stage95_official_visible_interval_20261022/fred/sequence194/native_predictions.mp4"></video>
    <p><a href="../stage94_official_large_target_20261022/download94.json">事件下载指纹</a> · <a href="../stage94_official_large_target_20261022/fred/sequence194/completion.json">94空段</a> · <a href="../stage95_official_visible_interval_20261022/scores95.json">95完整对照</a> · <a href="../stage96_native_frequency_input_20261022/scores96.json">96 SDK输入对照</a> · <a href="../stage97_projected_axis_magnitude_20261022/scores97.json">97近景对照</a> · <a href="../stage97_projected_axis_magnitude_20261022/self_scores97.json">97自采对照</a></p></section>'''
    original=OUT/'第八十八至九十三阶段因果对照与互补预测报告.html';text=original.read_text(encoding='utf-8')
    text=text.replace('第88—93阶段','第88—97阶段').replace('第88—93','第88—97')
    text=text.replace('<section><h2>仍需优先解决</h2>',section+'<section><h2>仍需优先解决</h2>')
    target=OUT/'第八十八至九十七阶段实验总览.html';target.write_text(text,encoding='utf-8')
    save(OUT/'delivery88_97.json',dict(report=str(target),report_sha256=sha(target),latest_RAW_stage=93,latest_native_module_stage=97,
        independent_RPM_gain=False,default_promoted=False,existing_data_enough_for_pilot=True,
        additional_data_recommendation='Official195 first for more large-target visibility,111 as smaller supplemental. Keep current official version separate from legacy E:/FRED.',
        data194_already_downloaded=True))
    print(str(target),flush=True)


if __name__=='__main__':main()
