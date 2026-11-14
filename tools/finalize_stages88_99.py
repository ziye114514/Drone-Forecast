"""Verify completed experiments, update handoff and create a recoverable snapshot."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import datetime,hashlib,importlib,json,os,sys,zipfile
import numpy as np
import deliver_stages88_97 as deliver
ROOT=deliver.ROOT;WORK=ROOT.parent;OUT=deliver.OUT
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage88'))
from inventory88 import save,load,sha


class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for key,v in attrs:
            if key in ['href','src'] and v and not urlsplit(v).scheme:self.local.append(unquote(v.split('#')[0]))


def main():
    deliver.main()
    modules=[(88,'study88'),(89,'study89'),(90,'study90'),(92,'study92'),(93,'study93'),(94,'study94'),(95,'study95'),(96,'study96'),(97,'study97'),(98,'per_rotor98'),(99,'study99')]
    validations=[]
    for stage,name in modules:
        sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/f'research/stage{stage}'));module=importlib.import_module(name);module.verify();validations.append(stage)
    b98=WORK/'output/stage98_four_rotor_measurements_20261022';b99=WORK/'output/stage99_four_rotor_force_input_20261022'
    c98=load(b98/'full/completion98.json');assert load(b98/'prefix/completion98.json')['prefix_exact']
    s99=load(b99/'scores99.json');assert load(b99/'fred/prefix194/completion.json')['actual_event_prefix_exact']
    rows=[r for r in s99['table'] if r['horizon']==.4 and r['group']=='all'];evidence=s99['input_evidence']
    # Independent source/position observations must match across these ablations.
    def trace(stage,name):return [json.loads(s) for s in (WORK/f'output/stage{stage}_{name}_20261022/fred/sequence194/forecasts.jsonl').open(encoding='utf-8')]
    ref=trace(95,'official_visible_interval');cross=[]
    for stage,name in [(96,'native_frequency_input'),(97,'projected_axis_magnitude'),(99,'four_rotor_force_input')]:
        current=trace(stage,name);assert len(current)==len(ref)==167
        for a,b in zip(ref,current):
            aa=dict(a['input_audit']);bb=dict(b['input_audit']);aa.pop('processing_ms');bb.pop('processing_ms');assert aa==bb
            for m in ['cv','trusted_acc','kf_tuned']:assert a['coefficients'].get(m)==b['coefficients'].get(m)
        cross.append(dict(stage=stage,ticks=167,all_current_native_observations_equal95=True,kinematic_references_exact=True))
    save(OUT/'cross_native_inputs.json',dict(records=cross,all=True))
    source=OUT/'第八十八至九十七阶段实验总览.html';text=source.read_text(encoding='utf-8')
    extra=f'''<section><h2>98—99：四个独立桨盘与频率平方和</h2>
    <p>98原样调用项目既有PaperPerception/RotorSensing，包括稳定ID、象限FFT/相位及频率平滑；同167个时刻中，当前四桨可见{c98['counts']['four_visible']}次、四椭圆{c98['counts']['four_ellipses']}次、保有四路新鲜频率状态{c98['counts']['four_frequencies']}次。频率状态可短暂延续，不能把它们都说成当前刚测到的四路RPM。</p>
    <p>99把四路频率平方和接到既有相对推力链，要求四个转子当前可见且四路频率时间戳都等于当前时刻，避免拿延续状态当新观测。参数/方向/延迟学习/对照保持。实际只有{evidence['four_measured_current_ticks']}个四路当前测量时刻，连续性处理后{evidence['relative_frequency_current_ticks']}次有效更新；有效增益反馈最高{evidence['max_gain_updates']}次，未满足原有10次启用门槛，所以整段物理发布为0。</p>
    <p class="warn">四桨链已接通，但这3.34秒窗口还没有完成在线校准，31.48px只是历史回退成绩，不能拿它评价四桨推力模型。没有为了出结果把10次门槛降为9次。下一步应在更长连续可见片段验证，而非继续在这23个重叠锚点上改参数。</p>
    {deliver.base.table(['方法','有效0.4s锚点','FDE像素'],[[r['method'],r['valid'],deliver.base.num(r['mean_fde'])] for r in rows])}
    <p>元数据核对：194最长连续提供框仅约3.37秒；195约17.13秒（0—17.13），111约78.83秒。195更适合补验证四路输入的在线校准，框短边中位约81px；它仍需事件/框对齐与实际四桨可见性检查。194事件已在D盘，不用再重复下载。</p>
    <p><a href="../stage98_four_rotor_measurements_20261022/full/completion98.json">98逐桨模块统计</a> · <a href="../stage99_four_rotor_force_input_20261022/scores99.json">99平方和与全部控制</a> · <a href="cross_native_inputs.json">95/96/97/99当前事件与位置输入一致性</a></p></section>'''
    text=text.replace('第88—97','第88—99').replace('<section><h2>仍需优先解决</h2>',extra+'<section><h2>仍需优先解决</h2>')
    report=OUT/'第八十八至九十九阶段实验总览.html';report.write_text(text,encoding='utf-8')
    parser=Links();parser.feed(text)
    for value in parser.local:
        if value:assert (report.parent/value).resolve().is_file(),value
    save(OUT/'final_checks88_99.json',dict(verified_frozen_stages=validations,report_local_links=len(parser.local),
        cross_native_inputs=True,actual_RAW93=load(OUT/'checks93.json'),per_rotor98=c98,per_rotor99=evidence,
        all_required_runs_complete=True,no_RPM_gain_proven=True,default_promoted=False))
    # Correct the superseded progress header; retain all historical text below.
    status=WORK/'PROJECT_STATUS_20261022.md';s=status.read_text(encoding='utf-8')
    s=s.replace('**88—97已完成；98逐桨模块在运行。','**88—99已完成；无本项目实验后台。')
    s=s.replace('第88—97阶段总览','第88—99阶段总览').replace('第八十八至九十七阶段实验总览.html','第八十八至九十九阶段实验总览.html')
    s=s.replace('98正在原样复用既有RotorSensing，检查四个独立桨盘/频率能否恢复；完成以completion98为准。',
        '98原样逐桨链四桨可见96/167、四路新鲜频率状态100/167；99接频率平方和仅22个四路当前测量/20次连续频率更新，增益学习最多9次，未达10次启用门槛，仍回历史。下一步优先更长连续近景（官方195标注最长17.13s），不降低门槛或继续调这23个锚点。')
    status.write_text(s,encoding='utf-8')
    readme=ROOT/'README.md';s=readme.read_text(encoding='utf-8').replace('88—97总览/视频','88—99总览/视频').replace('第八十八至九十七阶段实验总览.html','第八十八至九十九阶段实验总览.html').replace('94—97为过去提供框','94—99为过去提供框')
    readme.write_text(s,encoding='utf-8')
    worklog=WORK/'WORKLOG.md';s=worklog.read_text(encoding='utf-8').replace('前缀150? 实际1s为50时刻，精确通过，以completion为准。','1秒50时刻事件前缀精确通过。')
    if '## 127. 98/99逐桨链和平方和固定诊断完成' not in s:
        s+='''\n## 127. 98/99逐桨链和平方和固定诊断完成\n\n- 98原样复用PaperPerception/RotorSensing，20ms调用/原默认频率更新步数、无阈值搜索；同194可见区间167时刻，四桨当前可见96、四椭圆95、四频率新鲜状态100、任意新鲜频率116。1秒实际事件前缀精确。频率状态允许短暂延续，不宣称100次当前四路测量或真实机械RPM。\n- 99接已有逐桨频率平方和，相当sqrt(mean(f_i²))经原连续性/参考归一化，要求四个当前可见且四路frequency_t==当前；只22个四路当前测量，连续性后20次，增益反馈最高9次，未达原10次门槛。保留不启用结果，没有为了成功把门槛降为9。23锚点所有物理/常量/零/打乱都等于历史31.47903，不能评价为四桨模型效果；KF30.53503、回退29.60795。\n- 元数据确认194最长连续标注3.37秒，195为17.13秒、111为78.83秒。下一步需更长连续近景和新固定测试；优先当前官方195，保留旧/当前版本隔离。\n- 95/96/97/99共167时刻的事件/几何/背景候选/过去ROI审计字段一致（排除耗时），CV/历史/训练KF系数严格相同。所有冻结源再验证、总览本地链接核验、原结果/失败保留。无稳定实时、独立RPM增益或论文原创性已证明的主张。\n'''
    worklog.write_text(s,encoding='utf-8')
    # Extend the previous verified snapshot without putting raw/native data,
    # model weights, full JSONL or videos into the small recovery archive.
    prior=load(WORK/'output/checkpoints/latest_handoff.json');assert sha(prior['archive'])==prior['archive_sha256']
    summary=dict(prior);summary.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='checkpoint_ready',
        research_objective_status='incomplete',last_completed_stage=99,completed=list(range(19,100)),in_progress=[],experiments_running=[],
        latest_full_forecast_stage=93,latest_cached_forecast_stage=92,latest_native_diagnostic_stage=99,
        latest_report=str(report),latest_full_forecast_report=str(report),experimental_RAW_entry=str(ROOT/'core/run_complementary_forecast.py'),
        experimental_RAW_stage=93,compatibility_API_still_targets=85,default_promoted=False,
        independent_RPM_gain_proven=False,latest_data194=str(WORK/'output/stage94_official_large_target_20261022/input/events.hdf5'),
        new_thrust_input_enabled=False,new_thrust_input_scope='Original78 relative-periodic paths in RAW93 retained; research96 SDK and99 per-rotor sum not promoted.',
        completion_scope='88 causal controls;89 modules;90 train-onlyKF;91/91b versioned metadata;92 routing;93 actualRAW55;94 native194 and empty control;95 visible interval;96 SDK source;97 projected magnitude;98 fourrotor module;99 squaredsum warmup audit.',
        latest_checks=str(OUT/'final_checks88_99.json'),
        next_action='Prefer a longer new fixed close-view interval (official195 longest provided-box run0..17.13s), verify version/alignment, keep thresholds and full controls.99 has9 gain updates<10; do not lower gate on194 scores. Validate per-rotor harmonics/phase and predictive lead. RAW93 fallback worsens side/extra4, no promotion. True bounded queue/source-time async RAW tests and independent pose/RPM truth remain unresolved.',
        prior_archive=prior['archive'],prior_archive_sha256=prior['archive_sha256'])
    files=set()
    with zipfile.ZipFile(prior['archive']) as z:
        old_manifest=json.loads(z.read('SNAPSHOT_MANIFEST.json'))
        for name,value in old_manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==value
            p=WORK/name
            if p.is_file():files.add(p)
    for stage in range(88,100):files.update(p for p in (ROOT/f'research/stage{stage}').rglob('*') if p.suffix in ['.py','.md'] and '__pycache__' not in p.parts)
    files.update([Path(__file__),ROOT/'tools/deliver_stages88_93.py',ROOT/'tools/deliver_stages88_97.py',ROOT/'core/run_complementary_forecast.py',status,readme,worklog])
    for folder in (WORK/'output').glob('stage*20261022'):
        if not any(folder.name.startswith('stage'+str(stage)+'_') for stage in range(88,100)) and not folder.name.startswith('stage91b_'):continue
        files.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.json','.html','.md','.txt','.log'] and p.stat().st_size<2_000_000)
    failure=WORK/'output/stage90_pre_freeze_control_failure_20261022.json'
    if failure.exists():files.add(failure)
    manifest={p.relative_to(WORK).as_posix():sha(p) for p in sorted(files)}
    archive=WORK/'output/checkpoints'/('research_through_stage99_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    inside=dict(summary);inside.pop('archive',None);inside.pop('archive_sha256',None)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(WORK).as_posix())
        z.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));z.writestr('HANDOFF_CHECKPOINT.json',json.dumps(inside,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,report_local_links_checked=len(parser.local))
    save(WORK/'output/checkpoints/latest_handoff.json',summary);save(WORK/'output/checkpoints/latest_progress_20261022.json',summary)
    save(OUT/'delivery88_99.json',dict(report=str(report),archive=str(archive),files=len(files),zip_verified=True,latest_RAW=93,latest_native_diagnostic=99))
    print(json.dumps(dict(last_stage=99,files=len(files),verified=True,report=str(report),archive=str(archive)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
