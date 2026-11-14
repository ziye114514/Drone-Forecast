"""Authoritative end-of-batch handoff; retain missing origins and negative trials."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent

if __name__=='__main__':
    c=json.loads((WORK/'output/projection436_fixed_20261110/results.json').read_text('utf8'))
    a=json.loads((WORK/'output/raw_comparison438_20261110/results.json').read_text('utf8'))
    raw=json.loads((WORK/'output/raw_projection437_20261110/results.json').read_text('utf8'))
    pilot=json.loads((WORK/'output/temporal_birth439_20261110/results.json').read_text('utf8'))
    old={r['t']:r for r in pilot['scores']['control']['records'] if r['valid']};new={r['t']:r for r in pilot['scores']['temporal']['records'] if r['valid']}
    common=sorted(old.keys()&new.keys());pilot_common={k:float(np.mean([v[t]['error_px'] for t in common])) for k,v in [('control',old),('temporal',new)]}
    out=WORK/'output/projection_accuracy_20261110';out.mkdir(exist_ok=True)
    primary=c['pooled_same_original_origins']['qualified'];actual=a['pooled_identical_RAW_original101']
    table='\n'.join('|'+n+'|'+str(c['cases'][n]['qualified']['same_original_origins']['n'])+'|'+f'{c["cases"][n]["qualified"]["same_original_origins"]["mean"]:.3f}'+'|' for n in ('13','15','16','11','3','4'))
    note=f'''# 当前权威交接：432–440（2026-11-10）

用户本轮授权继续到额度耗尽；不使用重置券/购买额度。先0.4s精度30px与冻结新录像验证，再FPS；P95仅记录。原102起点不删，发出后才发生且不可观测的新操纵不强求提前猜中。所有本批作业已退出。

## 当前可运行候选与真实成绩

入口`run_rotor_projection_candidate.py`，原RAW/纯ONOFF桨叶/五圆4-3-2-线/可测＋未知力，无机体NN/G。原模型仍`models/native_response_plane_v1.json`，186原生教师、12残差＋gamma；6评测源整段排除拟合，但已暴露开发，弱轮廓代理不是独立COM真值。旧429入口`run_rotor_observation_candidate.py`、432入口`run_rotor_endpoint_candidate.py`和公开NN包保留。

**原缓存、同102起点：42934.607486→43234.422014→43634.295022px。**436中位29.056、30内54/102；429中位27.856、30内56，因此不称全面更稳。逐段：

|录像|原起点|436均值px|
|---|---:|---:|
{table}

**实际RAW437六段全部完成，采样13/15/16/11比旧缓存晚100us；原102有101可输出，均33.395801，不得当同102成绩。**11@12.0少1起点，另外5个新增输出单独记录。3/4原始包一致、预测差≤2.84e-14。

438已用完全相同RAW观测公平重放：旧429**34.161991**→新436**33.395801px**，同101原起点，当前输入/有效性完全相同，新入口重放差≤2.84e-14。中位28.527→26.630、30内55→53；11/4回退。不能把网格变化收益算到算法头上。**30未达，新录像未验证，未做FPS优化/验收。**

## 修了什么，什么没采用

432补相邻图像噪声相关：delta=p_t-p_prev+e_t-e_prev，原4px²配对方差拆2+2，位置与端点误差一起克隆。独立稠密Gaussian均值/协方差差2.13e-13/1.59e-11；这不是新数学发明。原RAW434181包复现差0。

436椭圆法线长度sin(tilt)与线的单位方向切换时，不能当成真实隐藏推力改变。保留投影幅值并共同变换d/h协方差，真实azimuth变化照常生效。控制例静止状态假加速度-400→0，圆/线往返0，实际0.05rad变化仍约30px/s²。默认旧flag=False数值不变。

433相同186教师重拟合同12+1模型34.608317，未胜432；435圆长轴尺度先验系数0.00285、34.422049略差，均未推广，不扫参。431旧两列转动增量仍未并入入口。

439跨两窗当前/历史严格像素确认冷启动：11号11.766649用17个不同注册像素、当前9个、双向闭环0.00356通过，未改当前频率或造力。但同原控制可用的{len(common)}个起点均值{pilot_common['control']:.6f}→{pilot_common['temporal']:.6f}px，未获稳定收益，**不推广、不接主入口、不再为这一帧降阈值**。全可用18→19点28.243→31.836是不同分母，不能直接比。第一次试跑最后60us包误取33333us触发原事件数断言，已停止；≤15.6全部评分起点之前180包对照差0，因此这些因果评分有效。固定包区间读取已修在工具里，未重跑以免无效消耗；失败记录保留。

436首次重放遗漏16号two-point启动元数据导致控制断言失败；正确重试从完整429观测重放并保存此字段，控制全6差0。432自身推断与RAW434不受影响，不能用旧432输出直接重放而忽略启动字段。

## 下一次只攻的卡点

1. 更早确认桨叶存在，不代表得到稳定机体材料参考。439提前出生后把局部支撑锚定为初始参考，速度/加速度有回退；下一步先核验该处相对位移参考及启动p/v交叉协方差，不降低五次观测门槛刷覆盖，不用按录像ID补丁。可复用已有两点初始化，但须由真正同一材料参考的两次当前观测支持。
2. 当前位移/净加速度与未来未知输入延续仍限制精度；4号12.0即使当前有效仍大误差。先区分起点已有偏差和随后新操纵，旧预测不改、不删除难点。
3. 保留E:/dataset/recordings/rec_20261004_162506/event.raw10–16s，仍未解码。先原102均值30，再冻结同一算法验证，不能先看新录像调参；还需核历史重叠。
4. 不重做已失败的边界/IoU、背景beta、窗口扫参、扩大教师量、435尺度权重。保护D:/yolov10-main、E:/dataset/software/yolov10-main、原RAW/SDK/权重/NN发布包。

报告`output/projection_accuracy_20261110/index.html`；源码/分数/失败快照在432–439目录。无安装、上传、外部消息或新代理。旧formal goal usageLimited不是账户额度证据；实际额度以get_usage_limits为准，最近核验短窗口98%已用、周82%已用。

---

'''
    p=ROOT/'docs/HANDOFF_20261110_CONDITIONAL_RESPONSE.md';s=p.read_text('utf8');mark='# 当前权威交接：432–440（2026-11-10）'
    if s.startswith(mark):s=s.split('\n---\n\n',1)[1]
    p.write_text(note+s,'utf8');(out/'handoff.md').write_text(note,'utf8')
    rows=''.join(f'<tr><td>{n}</td><td>{c["cases"][n]["qualified"]["same_original_origins"]["n"]}</td><td>{c["cases"][n]["qualified"]["same_original_origins"]["mean"]:.3f}</td></tr>' for n in ('13','15','16','11','3','4'))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶共享帧误差与侧视切换精度修复</title><style>body{{font:17px/1.75 system-ui,"Microsoft YaHei";max-width:1000px;margin:35px auto;padding:0 24px;background:#f5f7fa;color:#243248}}section{{padding:20px 26px;background:white;border-radius:12px;margin:18px 0}}td,th{{padding:8px 25px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}h1{{font-size:29px}}</style><h1>共享帧误差与椭圆／侧视切换修复</h1><p>2026-11-10 · 精度优先 · 30px未达 · 新录像未验证 · FPS后置</p><section><h2>两种对照分别记录</h2><p>原缓存同102起点：34.607→34.422→<b>34.295px</b>。原102起点全部保留，部分录像回退；30内56→54，不声称全面更稳。</p><p>六段实际RAW：旧429与新436在完全相同观测上，同101原起点<b>34.162→33.396px</b>，改善0.766px。原102中11@12.0缺输出，另5个新起点独立，不换分母冒充成功。4个源网格与旧缓存差0.1ms，故不能把33.396与34.295直接归因算法。</p></section><section><h2>与五圆及二力主线直接相关的修复</h2><p>相邻两次匹配共用同一帧，误差不独立；新增图像端点误差状态，原配对噪声总方差不变。椭圆能测倾角，线只能测方向；切换时只换未知响应的坐标表示，不凭空改变推力。静止控制例的假加速度−400→0，真实方向变化仍有效。当前可测与隐藏力分账保持，未测出RPM/牛顿力；无机体NN/G。</p></section><section><h2>原102逐录像均值</h2><table><tr><th>录像</th><th>起点</th><th>436误差px</th></tr>{rows}</table></section><section><h2>没有掩盖的失败</h2><p>同数据重训、圆尺寸尺度修正均未带来进一步收益。跨帧严格信号能提前出生，但共同{len(common)}点误差{pilot_common['control']:.3f}→{pilot_common['temporal']:.3f}px，未推广。更早看见桨叶，不等于找到了稳定的机体运动参考；这是下一处需要实际解决的问题。</p><p>439末尾60us包读取不匹配被断言拦截；此前全部被评分起点已计算，因果对照有效，未删除评分起点。修好工具后未浪费一次重跑来美化报告。</p></section><section><h2>继续工作的位置</h2><p>当前入口<code>event_drone_forecast/run_rotor_projection_candidate.py</code>，旧429/432入口保留；共享原186教师模型不变。先攻冷启动材料参考与当前运动偏差，再30px与新录像验证，最后FPS。P95仅附带记录。</p><p><a href="handoff.md">完整交接与下一步</a> · <a href="../projection436_fixed_20261110/results.json">102逐点评分</a> · <a href="../raw_comparison438_20261110/results.json">实际RAW公平对照</a> · <a href="../temporal_birth439_20261110/results.json">出生小试验与失败边界</a></p></section></html>'''
    (out/'index.html').write_text(page,'utf8')
    p=ROOT/'README.md';s=p.read_text('utf8');start='<!-- projection440-latest:start -->';end='<!-- projection440-latest:end -->'
    if start in s:s=s[s.index(end)+len(end):].lstrip()
    p.write_text(f'''{start}
> **当前原生桨叶候选（2026-11-10）：** `run_rotor_projection_candidate.py`补共享帧噪声与椭圆/侧视法线表示切换，仍为五圆＋可测/未知力主线，无机体NN/G。同原102起点34.607→**34.295px**；实际六段RAW同101起点公平旧/新34.162→**33.396px**，原102少1输出，不能混淆分母。11/4有回退，30px尚未达，新录像未验证，FPS后置。跨帧提早出生未稳定改善精度，保留负例未推广。原429/432入口和公开NN包保留。见[新报告](../output/projection_accuracy_20261110/index.html)与[权威交接](docs/HANDOFF_20261110_CONDITIONAL_RESPONSE.md)。以下为历史。
{end}

'''+s,'utf8')
    p=WORK/'output/checkpoints/latest_handoff.json';j=json.loads(p.read_text('utf8'));j.update(in_progress=False,background_jobs=[],running_jobs=[],experiments_running=[],objective_completed=False,
        candidate_entry=str(ROOT/'run_rotor_projection_candidate.py'),latest_report=str(out/'index.html'),current_precision_mean_px=primary['mean'],aggregate_same102=primary,
        candidate_scores={n:c['cases'][n]['qualified']['same_original_origins'] for n in c['cases']},
        status='Batch432-440 complete: cached original102 mean34.295; same actualRAW101 old34.162 to new33.396. One original missing, target30/new-video unmet. FPS deferred.',
        current_user_stage='432-440 complete; all jobs exited; accurate original102 and actual101 metrics separated',
        temporal_birth439=dict(promoted=False,completed_packets_per_arm=180,all_scored_origins_before_failure=True,common_available=len(common),common_means=pilot_common,
            final_packet_reader_fixed_not_rerun=True,birth_t=11.766649,reason='Earlier identity did not provide a stable initial motion reference'),
        next_action=['Investigate same material-reference displacement and startup p/v covariance after tentative rotor birth; do not lower fixed gates for source11.',
            'Preserve all original102; unobserved future commands cannot be retroactively predicted.',
            'Reach30px on original102, freeze and verify reserved video, then optimize FPS.'])
    p.write_text(json.dumps(j,ensure_ascii=False,indent=2),'utf8')
    p=WORK/'WORKLOG.md';s=p.read_text('utf8');head='## 2026-11-10 438–440：实际同包收益、冷启动负例与完整收尾'
    if head not in s:p.write_text(s+'\n\n'+head+f'\n\n实际RAW相同101原起点公平对照旧42934.161991→新43633.395801，候选实际复现差≤2.84e-14；同缓存原10234.295022，不混分母。439确实提前确认11.766649的17个不同严格像素/当前9，但共同{len(common)}可用点{pilot_common["control"]:.6f}→{pilot_common["temporal"]:.6f}px，未推广；新出生可能把局部支撑当初始机体运动参考，下一步只攻此参考/启动p-v协方差，不按录像调阈值。最后60us包误取全窗触发断言，原起点<=15.6均在失败前、因果评分完整有效；读取已修，未重跑。新报告projection_accuracy_20261110，README/交接/checkpoint同步。全部作业退出，未达30，新视频未解码，FPS未优化，保护路径/NN发布包未动；最近实际额度短窗98%周82%已用，未使用券或购买。\n','utf8')
    print('READY',primary,actual,'pilot common',pilot_common)
