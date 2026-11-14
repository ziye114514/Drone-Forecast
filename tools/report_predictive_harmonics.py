"""Local result report and handoff for the new temporal/arc observation branch."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,ast,re
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'event_drone_forecast'
OUT=ROOT/'output/predictive_harmonics_20261027'


def load(p):return json.loads((OUT/p).read_text('utf-8'))


def main():
    controls=load('final_controls/controls.json');native=load('native_pilot/summary.json')
    aligned=load('aligned/summary.json');gpu=load('cuda_batched/summary.json');sparse=load('cuda_sparse/summary.json')
    split=load('final_arc_split/summary.json');side=load('final_arc_split/side.json')
    # Show fitted curves separately from the actual contour samples.
    canvas=cv2.cvtColor(cv2.imread(str(OUT/'aligned/side/count20_mask.png'),0),cv2.COLOR_GRAY2BGR)
    for parent in side['parents']:
        for child in parent['children']:
            if 'center' not in child:continue
            points=np.asarray(child['original_density_contour_points'],int)
            color=(40,210,50) if child['qualified'] else (30,130,240)
            for p in points:cv2.circle(canvas,tuple(p),1,color,-1)
            c=tuple(np.rint(child['center']).astype(int));axes=tuple(np.maximum(1,np.rint([child['major']/2,child['minor']/2])).astype(int))
            cv2.ellipse(canvas,c,axes,child['angle'],0,360,color,1)
            cv2.circle(canvas,c,3,color,-1)
    crop=canvas[465:565,690:1020];crop=cv2.resize(crop,(990,300),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT/'qualified_partial_arc.jpg'),crop)
    old=controls['same_input_old_reference_accepted'];scores=controls['scores']
    total=lambda kind:sum(v['accepted'] for v in scores[kind].values())
    table=''.join(f'<tr><td>{name}</td><td>{r["original_accepted_pixels"]}</td><td>{r["aligned_accepted_pixels"]}</td><td>{r["aligned_CPU_ms"]:.1f}</td><td>{g["steady_mean_ms"]:.1f}</td><td>{s["steady_mean_ms"]:.1f}</td></tr>'
        for name,r,g,s in zip(['clear','partial','side'],aligned['records'],gpu['records'],sparse['records']))
    plots=''.join(f'<section><h2>{r["name"]}：同事件补偿前后</h2><img src="aligned/{r["name"]}/comparison.jpg" alt="固定像素周期与过去速度补偿对比"></section>' for r in aligned['records'])
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>可预测谐波与原弧分组优化</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.8;max-width:1100px;margin:36px auto;padding:0 20px;background:#f2f5f8;color:#233145}section{background:white;padding:23px 27px;margin:18px 0;border-radius:10px}h1{font-size:29px}h2{font-size:21px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #dce3e8;text-align:left}code{background:#eef2f5}.note{background:#fff0d4;padding:12px}a{color:#1b6598}</style>
<h1>可预测谐波：筛选、运动补偿与原弧分组</h1><p>2026-10-27 · 已实现新的观测分支及CUDA版本 · 默认检测/预测与公开ZIP保留。</p>
<section><h2>这次实际优化了什么</h2><ol><li>前半窗找频率模板，后半窗检查相位/极性能否被预测；再要求跨时间分布，拒绝单次边缘突发。不是继续调高旧峰值阈值。</li><li>复用项目stage21原生事件平移：速度必须早于整个33.3ms窗口，保留每个事件原时间和极性。</li><li>频率区域仅给分组种子，拟合仍使用原计数掩码经过原形态学后的外轮廓点，避免把裁切口当桨盘边界。</li><li>既有拟合资格不放宽；完整双桨拆分失败时，仍单独保留可靠的那条当前弧，不要求所有兄弟弧同时成功。</li><li>CUDA批量计算及既有固定25%像素采样，保留所选像素的全部时间序列。</li></ol><p class="note">本轮没有新的未来轨迹精度成绩。新增的是周期证据与局部几何候选，未宣称测出真实电机RPM、完整四桨姿态或牛顿推力。</p></section>
<section><h2>同输入反例：旧问题确实被修正</h2><table><tr><th>固定合成输入，每类64像素</th><th>旧本地频谱参考接纳</th><th>新方法接纳</th></tr>'''+''.join(f'<tr><td>{label}</td><td>{old[kind]}</td><td>{total(kind)}</td></tr>' for kind,label in [('random','独立随机时刻/极性'),('single_edge','单次边缘突发'),('periodic','已知周期'),('common_flicker','共同周期照明')])+'''
</table><p>每类包含12/24/48/96四种事件量，每种16像素。真实已知周期在24事件时15/16、48和96时各16/16，12事件时0/16；拒绝弱证据会损失召回。共同照明仍能通过，因为它确实周期性，不能把周期候选直接称无人机检测。</p><p>后半窗界依赖独立平稳随机极性的理想假设，不是已校准的真实相机误检概率。源码与参数在读取实际窗口结果前固定，没有在真实标签上搜索阈值。</p></section>
<section><h2>实际RAW：侧视周期证据对齐后更完整</h2><table><tr><th>窗口</th><th>未对齐周期像素</th><th>对齐后</th><th>CPU组件ms</th><th>CUDA批量ms</th><th>CUDA固定25%像素ms</th></tr>'''+table+'''
</table><p>三个33.333ms窗口与上轮相同。侧视速度约(252,243)px/s，对应窗口内(8.4,8.1)px移动；对齐后131→917像素通过独立检查。清晰段1942→1656反而减少，不能称补偿处处改善。像素数不是标注召回率。</p><p>后续实验均复用缓存原生事件；没有反复读取RAW。GPU与CPU在三个窗口的掩码逐像素一致；25%采样与完整结果在同一固定像素集合逐项一致。GPU计时含CPU分组、传输和CUDA同步；每项仅两次稳态样本，冷启动另计，不是完整系统FPS或稳定延迟。</p><p class="note">当前组件仍有36–61ms的稀疏版本耗时，整条RAW/NN/几何/预测链路尚未达到30FPS。没有把少算像素或播放速度当成达标。</p></section>
'''+plots+'''
<section><h2>几何：保留一条可靠当前弧，其余不硬凑</h2><p>受控两个重叠椭圆：通过现有资格得到2个子中心。单个椭圆即使有两个频率种子，也不能变成两个电机；无种子不拆。</p><p>实际侧视两处粘连，完整双桨拆分均失败；但有1条分组后的原弧通过：中心sigma代理1.20px、边界支持0.833、条件数7.15。相同父轮廓使用同一残弧拟合器、不作频率分组时未合格，因此这是分组带来的局部候选进展。</p><img src="qualified_partial_arc.jpg" alt="绿色合格部分弧与橙色拒绝弧"><p>绿色为合格部分候选，橙色仍拒绝；细椭圆线是拟合模型，点才是原密度轮廓采样。不是四个真实桨心真值图。其他弧卡边界支持约0.46，或中心sigma代理3.63/5.84px；保持原门槛。</p><p>该弧尚无物理电机ID；未初始化完整模型时，只能给局部方向/部分观测，不能凭此初始化四桨或刷新隐藏姿态资格。</p></section>
<section><h2>保留了什么，下一步改什么</h2><p>保留旧默认、CPU参考、GPU与稀疏版本、所有失败候选和原位置核心。过去小频库虽把CPU降至253–616ms，却使侧视合格像素917→108：先前频率可能已经偏了，因此不能为了快永久锁死它。</p><p>下一步是低频宽带重发现与高频小频库验证分工，带失锁恢复；把合格单弧送到五圆可观测方向/部分信息更新。GPU目前仍有分组/传输/启动开销，需考虑固定预算与图捕获等结构优化。当前没有新的轨迹评分，旧52.718px等结果未被本轮更名为新收益。</p></section>
<section><h2>来源、代码与证据</h2><p><a href="https://arxiv.org/abs/2603.08386">DDHF</a>提供逐像素谐波思路；<a href="https://arxiv.org/abs/2610.31384">CoHelix</a>提供共享几何参考；<a href="https://github.com/JackPieCZ/EE3P">EE3P/EEPPR</a>为空间重复对照。没有复制作者实现。GPU分段归约参照<a href="https://docs.pytorch.org/docs/stable/generated/torch.segment_reduce.html">PyTorch官方文档</a>，先CPU检查长度再跳过重复校验，不把通用变换/滤波/补偿称为原创。</p><p><a href="../../event_drone_forecast/docs/PREDICTIVE_HARMONICS.md">方法、局限与运行入口</a> · <a href="final_controls/controls.json">同输入固定控制</a> · <a href="native_pilot/summary.json">实际窗口记录</a> · <a href="aligned/summary.json">过去速度补偿</a> · <a href="cuda_batched/summary.json">GPU批量结果</a> · <a href="cuda_sparse/summary.json">固定像素结果</a> · <a href="final_arc_split/side.json">全部侧视子弧与拒绝原因</a></p></section></html>'''
    (OUT/'index.html').write_text(report,'utf-8')
    local=[x for x in re.findall(r'(?:href|src)="([^"#]+)"',report) if '://' not in x]
    assert all((OUT/x).is_file() for x in local)
    # Preserve exact baseline and publication boundaries.
    archive=ROOT/'output/github_release_final_20261027/event-drone-forecast-v0.4.0-alpha.1.zip'
    zip_hash=hashlib.sha256(archive.read_bytes()).hexdigest();assert zip_hash=='ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'
    old=ROOT/'output/github_release_final_20261027/event-drone-forecast/src/event_drone_forecast/position'
    same=[p.name for p in old.glob('*.py') if p.read_bytes()==(CODE/'src/event_drone_forecast/position'/p.name).read_bytes()];assert len(same)==7
    paths=['predictive_harmonics.py','predictive_harmonics_cuda.py','native_event_alignment.py','harmonic_arc_split.py',
        'tools/check_predictive_harmonics.py','tools/probe_predictive_harmonics.py','tools/probe_aligned_harmonics.py',
        'tools/probe_past_band_harmonics.py','tools/bench_predictive_harmonics_cuda.py','tools/probe_harmonic_arc_split.py']
    snapshot=OUT/'final_source_snapshot';snapshot.mkdir(exist_ok=True);hashes={}
    for name in paths:
        data=(CODE/name).read_bytes();ast.parse(data,filename=name);p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    original=ast.parse((CODE/'research/stage21/windows21.py').read_text('utf-8'));current=ast.parse((CODE/'native_event_alignment.py').read_text('utf-8'))
    node=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='warp')
    assert ast.dump(node(original),include_attributes=False)==ast.dump(node(current),include_attributes=False)
    checks=dict(local_links_checked=len(local),original_position_core_files_unchanged=same,
        frozen_GitHub_ZIP_sha256=zip_hash,exact_stage21_warp_definition=True,source_hashes=hashes,
        default_forecast_unchanged=True,new_trajectory_accuracy_claim=False,stable30FPS_claim=False,
        current_jobs=[],all_RAW_unchanged=True,Zotero_untouched=True)
    (OUT/'artifact_checks.json').write_text(json.dumps(checks,indent=2),'utf-8')
    stamp=datetime.now(timezone.utc).isoformat();checkpoint=ROOT/'output/checkpoints/latest_handoff.json'
    handoff=json.loads(checkpoint.read_text('utf-8'))
    handoff.update(updated_utc=stamp,current_work_state='Completed predictive harmonic, past-warp, original-arc grouping and bounded CUDA optimization batch; no current jobs',current_jobs=[],current_running_jobs=[])
    handoff['predictive_harmonics_research']=dict(report=str(OUT/'index.html'),method=str(CODE/'docs/PREDICTIVE_HARMONICS.md'),
        random_old_new_same64=[64,0],single_edge_old_new_same64=[64,0],weak12_event_periodic_accepted=0,
        periodic48_or96_each16_accepted=16,common_flicker_not_resolved=True,
        side_period_pixels_unaligned_aligned=[131,917],current_qualified_side_partial_arcs=1,complete_pair_splits=0,
        whole_pipeline_30FPS=False,new_future_accuracy_scored=False,default_changed=False,
        GPU_sparse_component_means_ms=[r['steady_mean_ms'] for r in sparse['records']],
        next='Two-rate broad rediscovery and small-band validation with loss recovery; partial arc -> observable model directions only; reduce GPU grouping/launch costs without claiming pixel-count or partial-arc counts as accuracy')
    handoff['latest_research_report']=str(OUT/'index.html');checkpoint.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
    marker='## 2026-10-27 预测式谐波、过去速度对齐与原弧分组'
    worklog=ROOT/'WORKLOG.md'
    if marker not in worklog.read_text('utf-8'):
        with worklog.open('a',encoding='utf-8') as f:
            f.write('\n\n'+marker+'\n\n'
                '- 新增predictive_harmonics.py：前半窗选频率/四阶模板、后半窗验极性相位，跨时间覆盖与理想随机极性条件界；不是DDHF完整复现。固定同64随机/64突发，旧本地参考各64通过，新各0；已知周期24事件15/16、48/96各16/16、12事件0/16，弱信号有损失，共同闪烁未解决。\n'
                '- 首次读取同3窗口后缓存原生事件。复用stage21 warp定义，速度截止早于完整窗，零速度精确、未来速度拒绝；原时间极性不变。clear1942→1656、partial719→835、side131→917合格周期像素，不是检测召回提升。\n'
                '- harmonic_arc_split.py用频率种子分组原密度外轮廓，禁止把切口当边界、禁止种子质心当桨心。受控重叠2中心通过/单椭圆双种子不造2电机；实际side完整拆分0，但同父相同残弧拟合无分组不合格、分组后1条原弧合格，sigma1.20/support.833/condition7.15，作为部分候选保留，其余拒绝。\n'
                '- CPU参考约1.06–2.28s；CUDA批量清晰/partial/side86.5/40.8/58.3ms，固定25%像素61.0/36.3/45.6ms，所有对应掩码与CPU精确；只2稳态计时/项，包含分组传输同步，不是完整系统FPS。缩小到过去3–5候选虽快，side917→108，未采用固定旧频率。\n'
                '- 报告output/predictive_harmonics_20261027/index.html，docs/PREDICTIVE_HARMONICS.md与全部输入/输出/失败候选/最终源快照保留。7位置核心与已发布ZIP哈希不变，不改Zotero或yolov10目录。尚未接默认预测/新未来评分/达成30FPS。下一步双速率重发现/失锁恢复及合格单弧的可观测方向更新。无后台作业。\n')
    print(json.dumps(dict(report=str(OUT/'index.html'),local_links=len(local),qualified_side_partial_arcs=1,
        default_unchanged=True,whole_pipeline_30FPS=False),ensure_ascii=False))


if __name__=='__main__':main()
