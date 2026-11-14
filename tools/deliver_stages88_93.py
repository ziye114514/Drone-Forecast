"""Evidence-first HTML/figures for causal controls and complementary prediction."""
from pathlib import Path
import datetime,html,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path.insert(0,str(ROOT/'research/stage88'))
from inventory88 import load,save,sha,parse,unique_rows
OUT=WORK/'output/stage93_RAW_complementary_20261022'
B88=WORK/'output/stage88_causal_event_controls_20261022'
B89=WORK/'output/stage89_small_target_periods_20261022'
B90=WORK/'output/stage90_train_only_kinematic_20261022'
B91=WORK/'output/stage91_FRED_download_audit_20261022'
B91B=WORK/'output/stage91b_FRED_all_test_screen_20261022'
B92=WORK/'output/stage92_complementary_router_20261022'
CASES=['clear6','side6','train3','extra1','extra2','extra4']


def rowmap(score,case):return {r['method']:r for r in score['table'] if r['case']==case and r['horizon']==.4 and r['group']=='all'}
def num(v):return '—' if v is None else f'{v:.2f}'
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    OUT.mkdir(exist_ok=True)
    s88=load(B88/'scores88.json');s90=load(B90/'scores90.json');s92=load(B92/'scores92.json');inv=load(B88/'inventory88.json')
    q93=load(OUT/'checks93.json') if (OUT/'checks93.json').exists() else None
    actual=[c for c in ['clear6','side6','train3'] if (OUT/c/'regression93.json').exists()]
    # Source-time endpoint metrics remain separate from actual publish latency.
    fig,axes=plt.subplots(1,2,figsize=(13,4.4));x=np.arange(6);width=.17
    for j,(method,label) in enumerate([('event','Event'),('constant','Constant'),('zero','Zero correction'),('lag200','Lag200ms')]):
        values=[rowmap(s88,c)[method]['mean_fde'] for c in CASES];axes[0].bar(x+(j-1.5)*width,values,width,label=label)
    axes[0].set_xticks(x,CASES,rotation=20);axes[0].set_ylabel('0.4s endpoint error (px)');axes[0].set_title('Own automatic observations: matched controls');axes[0].legend(fontsize=8)
    names=['cv','trusted_acc','kf_tuned','raw_q'];values=[s90['test_04_mean_fde'][m] for m in names]
    names+=['fallback_router'];values+=[s92['test_04_mean_fde']['fallback_router']]
    axes[1].bar(range(len(values)),values,color=['#7696bc','#b39bc8','#5aa786','#d3ad65','#68a7b7']);axes[1].set_xticks(range(len(values)),names,rotation=25)
    axes[1].set_ylabel('0.4s endpoint error (px)');axes[1].set_title('Legacy FRED8/9: past-provided-box diagnostic')
    fig.tight_layout();fig.savefig(OUT/'causal_controls_and_FRED.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.4))
    for j,(method,label) in enumerate([('primary','Original primary'),('kf_tuned','Train-only KF'),('fallback_router','Fallback router'),('full_router','Full router')]):
        axes[0].bar(x+(j-1.5)*width,[rowmap(s92,c)[method]['mean_fde'] for c in CASES],width,label=label)
    axes[0].set_xticks(x,CASES,rotation=20);axes[0].set_ylabel('0.4s endpoint error (px)');axes[0].set_title('Fixed candidates: no per-recording selection');axes[0].legend(fontsize=8)
    coverage=[]
    for seq in [0,4,6,10,8,9]:coverage.append(load(B89/f'sequence{seq}/completion.json'))
    labels=[str(r['sequence']) for r in coverage];xx=np.arange(6)
    for j,(m,label) in enumerate([('original_spectrum','Original spectrum'),('guarded_hist','Guarded histogram'),('strict7','SDK strict7'),('paper_argmax','Raw argmax (unreliable)')]):
        axes[1].bar(xx+(j-1.5)*width,[100*r['qualified_ticks'][m]/r['ticks'] for r in coverage],width,label=label)
    axes[1].set_xticks(xx,labels);axes[1].set_xlabel('FRED recording ID');axes[1].set_ylabel('Candidate coverage (%)');axes[1].set_title('Coverage is NOT RPM accuracy');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'router_and_period_coverage.png',dpi=165);plt.close(fig)
    rows1=[];rows2=[]
    for c in CASES:
        r=rowmap(s88,c);rows1.append([c,f"{r['event']['valid']}/{r['event']['targets']}",*[num(r[m]['mean_fde']) for m in ['event','constant','zero','lag200']],num(float(np.median([r[f'past_shuffle{s}']['mean_fde'] for s in [101,202,303]])))])
        r=rowmap(s92,c);rows2.append([c,*[num(r[m]['mean_fde']) for m in ['primary','kf_tuned','history_router','fallback_router','full_router']]])
    dataset=load(B91B/'inventory91b.json') if (B91B/'inventory91b.json').exists() else load(B91/'inventory91.json')
    ranked=sorted([r for r in dataset['records'] if 'error' not in r and r['sequence'] not in [8,9]],key=lambda r:(r['fraction_side96'],r['median_min_box_side_px']),reverse=True)
    downloads=[]
    for r in ranked[:8]:downloads.append([f"<a href='{html.escape(r['download_url'],quote=True)}'>{r['sequence']}.zip</a>",f"{r['archive_bytes']/1e6:.0f} MB",num(r['median_min_box_side_px']),f"{100*r['fraction_side96']:.2f}%"])
    provenance=[]
    for seq in [8,9]:
        a,_=parse((B88/'annotations'/f'{seq}.txt').read_text(encoding='utf-8-sig'));b,_=parse((B91/'annotations'/f'{seq}.txt').read_text(encoding='utf-8-sig'))
        aa={round(r['t'],6):r for r in unique_rows(a)};bb={round(r['t'],6):r for r in unique_rows(b)};common=aa.keys()&bb.keys()
        shifts=[np.linalg.norm((np.array(aa[t]['bbox'][:2])+aa[t]['bbox'][2:])/2-(np.array(bb[t]['bbox'][:2])+bb[t]['bbox'][2:])/2) for t in common]
        provenance.append(dict(sequence=seq,legacy_labels=len(a),current_labels=len(b),shared_timestamps=len(common),median_center_difference_px=float(np.median(shifts)),annotation_sha_equal=False))
    save(OUT/'annotation_version_audit.json',dict(records=provenance,note='Shared timestamps suggest related recordings, but altered boxes/time coverage and RAW vs HDF5 packaging prevent version-equivalence claims.'))
    clips=''.join(f"<div><h3>{c}：蓝CV／紫旧IMM／黄原主线／青新回退融合</h3><video controls preload='metadata' src='{c}/complement_forecasts.mp4'></video></div>" for c in actual)
    status='完整RAW验证已完成：'+str(q93['parent'])+'个20ms父窗口、'+str(q93['native10'])+'个原生10ms窗口，旧49与当前观测保持、新6与缓存92一致，两个真实截断前缀通过。' if q93 else '完整RAW验证仍在进行。以下92精度属于缓存预测层，已结束的真实RAW片段另列；不能把预备实现视为全部完成。'
    html_text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第88—93阶段：因果对照、FRED版本与互补预测</title><style>
    body{{font:16px/1.75 "Microsoft YaHei",sans-serif;max-width:1180px;margin:28px auto;padding:0 22px;color:#263648;background:#f7f9fc}}h1,h2{{line-height:1.35}}section{{background:white;padding:22px 26px;margin:20px 0;border-radius:12px;border:1px solid #dde4ed}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #dce2eb;padding:8px;text-align:left}}th{{background:#edf2f8}}img,video{{max-width:100%;width:100%}}.warn{{background:#fff1d9;border-left:5px solid #d89c28;padding:14px}}code{{overflow-wrap:anywhere}}a{{color:#166da1}}</style>
    <h1>第88—93阶段：真实事件信号是否有用，如何与运动预测互补</h1>
    <p>2026-10-22 · 保存全部负结果 · 原默认54与原49路不改 · 新入口 <code>run_complementary_forecast.py</code></p>
    <section><h2>结论</h2><p>新一轮没有证明机械转速或周期信号的独立预测收益。旧版FRED固定测试片段没有足够周期观测，物理分支全部回历史；训练录像选择的运动KF和互补路由改善了部分误差，但侧视等片段仍有退步，没有升级为稳定高精度实时系统。</p><p>{status}</p>
    <p>现有11包足够做小型诊断，含约{sum(r['unique_valid_rows'] for r in inv['sequences']):,}条可用独立位置框。当前对照只覆盖旧版官方测试编号8、9：425个有效0.4秒锚点、95个按未来标注方向定义的变向锚点、0个当前物理输入锚点。它们以前已被分析，不能称盲测。</p></section>
    <section><h2>88：真实周期与恒定、零修正、延迟、过去打乱</h2><p>六段自采使用原78实际自动检测与事件输入缓存，真实/恒定结果逐点精确复现。过去打乱仅取200ms—2s之前的合格值，三固定种子；所有对照共享真实输入的发布资格，保留原覆盖。</p>
    {table(['片段','有效/目标','真实周期','恒定','零修正','延迟200ms','3次过去打乱中位'],rows1)}
    <p>清晰段只有很小的周期增益；侧视恒定略好；额外4真实输入明显差于常量/打乱。不能把相对最初CV的改善都归因于转速。评分使用独立稀疏框中心，单位像素，不是真实重心或米制位置。</p><img src="causal_controls_and_FRED.png"></section>
    <section><h2>89：周期观测卡在哪里</h2><p>实际复用SDK FrequencyAlgorithm和已核验的原生数组适配。按论文补周期直方图，并保留裸argmax及带唯一像素/背景环拒绝的版本。42个人工过程：带拒绝的直方图在18个正过程末时刻接受17个，在24个随机/有色/突发/整体闪烁负过程接受0个；裸argmax误收10个负过程。此结果只验证这组人工信号，不能替代飞行RPM真值。</p>
    <p>六段各30秒真实FRED：旧频谱只有9/9000时刻得到合格周期；带拒绝直方图116/9000，全部来自训练0、6号；测试8、9仍为0。更高裸峰覆盖没有接入推力。ON-after-OFF间隔存在占空比、叶片数、谐波歧义。</p><img src="router_and_period_coverage.png"></section>
    <section><h2>90/92：先让运动基线扎实，再做互补</h2><p>仅在训练0、4、6、10选择KF过程噪声，网格50—6400，选中6400（3200与6400训练均值十分接近）；观测噪声6px不变，测试/自采未参与选参。FRED8/9的0.4秒误差：原CV75.68、历史60.43、训练选KF52.03、互补回退53.76px。裸直方图调Q53.51px，未胜过强KF；可靠周期为0时，带拒绝的Q分支与KF一致。</p>
    <p>复用29延迟专家权重（0.2s反馈、0.26s失效、35px损失尺度、0.08学习率、0.01共享）。新回退：当前物理输入可靠时保留原主线，否则融合CV/历史/训练选KF。另保留四路全融合和固定权重对照；不按录像挑最好结果。</p>
    {table(['自采片段','原主线','训练选KF','历史路由','物理缺失时新回退','四路全融合'],rows2)}
    <p>新回退在清晰/录像3/额外1/额外2改善，侧视和额外4退步；四路全融合也不普遍更好。尚不替换默认。位置反馈是后续检测位置，自采不是飞控真值。FRED输入为过去提供框，不能宣称只输入RAW的部署精度。</p></section>
    <section><h2>93：真实RAW预测展示</h2><p>新六路由实际当前事件、实际NN/桨叶worker结果和过去位置驱动；不读取缓存或标注。以下视频按源50fps播放，不代表运行50Hz。新青线仅显示预先指定的新回退候选；原黄线仍保留，失败/出画不删。</p>{clips}</section>
    <section><h2>FRED还要下载多少</h2><p>完整轨迹评测还需要更多固定且未参与开发的测试录像；桨盘/姿态研究优先自采近景。仅为增加相似小目标包，不能恢复看不见的桨叶信息。已通过HTTP范围读取{dataset.get('successes',len(dataset['records']))}个当前官方测试包的目录/标注，未下载完整视频或RAW；按框短边筛选下表，只是数据筛查，不保证四桨可见。</p>
    {table(['当前官方包','大小','无人机框短边中位px','短边≥96px占比'],downloads)}
    <div class="warn">现有旧包8、9与当前官方同编号包标注不同，时间覆盖也有变化；当前包事件为HDF5、现有为RAW。不能覆盖旧文件或混用两个坐标版本。后续建议另建 <code>E:\\FRED_official_20261022\\test</code>，保留完整包、官方划分与版本指纹。新版标注已经用于本次数据筛查，因此不能再称完全未查看的盲测材料。</div>
    {table(['编号','旧框行数','当前框行数','同时间戳数','中心差中位px'],[[r['sequence'],r['legacy_labels'],r['current_labels'],r['shared_timestamps'],num(r['median_center_difference_px'])] for r in provenance])}
    <p>当前官方完整test约40.5GB。本轮无需先全部下载：先按上表补几段并核验事件/框对齐，统一新版协议后再扩展。FRED提供位置框，不能解决真实电机RPM、推力和姿态真值缺失。</p></section>
    <section><h2>仍需优先解决</h2><ol><li>近景中验证周期变化是否先于独立位置变向；继续保持恒定、零修正、延迟、过去打乱对照。</li><li>提高可信方向/周期的联合覆盖，记录未启用原因；不能放宽门槛后把噪声峰叫RPM。</li><li>解决侧视新回退退步；在新的固定片段验证，不在这批稀疏标签上继续挑路由。</li><li>端到端仍需有界队列/异步源时间更新的实际实验；本轮没有解决原NN/观测耗时，缓存路由很快不等于RAW链路实时。</li></ol>
    <p>椭圆/周期推力、KF事件修正和多模型融合已有研究；本轮主要贡献是可审计的对照、数据版本核验和工程整合，未形成已证明的算法新颖性。</p></section>
    <section><h2>来源与复查文件</h2><p><a href="https://github.com/miccunifi/FRED">FRED官方</a> · <a href="https://huggingface.co/datasets/GabrieleMagrini/FRED/tree/main/test">官方测试包</a> · <a href="https://arxiv.org/html/2603.01997v1">RPM调KF论文</a> · <a href="https://arxiv.org/html/2609.00839v1">事件残差KF论文</a> · <a href="https://filterpy.readthedocs.io/en/latest/kalman/MMAEFilterBank.html">FilterPy</a></p>
    <p><a href="../stage88_causal_event_controls_20261022/protocol88.json">88冻结协议</a> · <a href="../stage88_causal_event_controls_20261022/scores88.json">88完整正负结果</a> · <a href="../stage89_small_target_periods_20261022/synthetic89.json">89人工正负控制</a> · <a href="../stage90_train_only_kinematic_20261022/selected_sigma.json">90训练选参</a> · <a href="../stage92_complementary_router_20261022/scores92.json">92所有路由</a> · <a href="protocol93.json">93RAW协议</a></p></section></html>'''
    path=OUT/'第八十八至九十三阶段因果对照与互补预测报告.html';path.write_text(html_text,encoding='utf-8')
    save(OUT/'delivery88_93.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),report=str(path),actual_RAW_cases=actual,
        raw93_complete=q93 is not None,report_sha256=sha(path),independent_event_gain_proven=False,default_promoted=False,
        latest_complete_cached_stage=92,latest_complete_RAW_stage=93 if q93 else 85))
    print(str(path),flush=True)


if __name__=='__main__':main()
