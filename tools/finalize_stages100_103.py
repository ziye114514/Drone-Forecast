"""Finalize longer fixed test, native-data recovery and parent-safe API."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import datetime,hashlib,importlib,json,sys,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_input_learning
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage88'))
from inventory88 import save,load,sha
OUT=WORK/'output/stage93_RAW_complementary_20261022'
DATA=WORK/'output/stage100_official195_longer_20261022'


def table(headers,rows):return '<table><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'
def num(v):return '—' if v is None else f'{v:.2f}'


class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v and not urlsplit(v).scheme:self.local.append(unquote(v.split('#')[0]))


def main():
    s=load(DATA/'scores100.json');completion=load(DATA/'fred/sequence195/completion.json')
    assert load(DATA/'fred/prefix195/completion.json')['actual_event_prefix_exact']
    native=load(DATA/'download94.json');assert native['sequence']==195 and native['crc_matches_zip']
    assert sha(native['path'])==native['native_sha256']
    api=load(WORK/'output/stage101_complementary_API_20261022/checks101.json');assert api['all55_exact93'] and api['parent_IO_safe']
    recovered=load(WORK/'output/stage102_resumable_native_download_20261022/completion102.json');assert recovered['native_sha256']==native['native_sha256']
    background=load(WORK/'output/stage103_background_motion_audit_20261022/checks103.json')
    for stage,name in [(93,'study93'),(99,'study99'),(100,'longer100')]:
        sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/f'research/stage{stage}'));importlib.import_module(name).verify()
    for path,h in load(WORK/'output/stage101_complementary_API_20261022/protocol101.json')['source_hashes'].items():assert sha(path)==h
    assert sha(ROOT/'research/stage103/background103.py')==load(WORK/'output/stage103_background_motion_audit_20261022/protocol103.json')['source_sha256']
    mm={r['method']:r for r in s['table'] if r['horizon']==.4 and r['group']=='all'}
    methods=['cv','trusted_acc','event','constant','zero','lag200','past_shuffle101','past_shuffle202','past_shuffle303','kf_tuned','fallback_router','full_router']
    summary_rows=[[m,mm[m]['valid'],mm[m]['targets'],num(mm[m]['mean_fde']),num(mm[m]['mean_ade'])] for m in methods]
    controls=['constant','zero','lag200','past_shuffle101','past_shuffle202','past_shuffle303']
    relative={m:1-mm['event']['mean_fde']/mm[m]['mean_fde'] for m in controls}
    event_supported=all(v>.05 for v in relative.values())
    verdict='这个固定片段出现了超过全部匹配控制5%的收益，仍需其他新录像确认，不能直接当机械RPM或普遍增益。' if event_supported else '这个固定长片段仍没有证明四桨转速信息相对全部匹配控制的独立收益。'
    plot_input_learning.plot(DATA/'fred/sequence195/forecasts.jsonl',OUT/'four_rotor_learning100.png','Official195: same fixed99 four-rotor input on a longer visible interval')
    fig,ax=plt.subplots(figsize=(12,4.2));ax.bar(np.arange(len(methods)),[mm[m]['mean_fde'] for m in methods]);ax.set_xticks(np.arange(len(methods)),methods,rotation=35,ha='right');ax.set_ylabel('0.4s endpoint error (px)');ax.set_title('New195 fixed-parameter transfer: original future labels, no retuning');fig.tight_layout();fig.savefig(OUT/'official195_fixed_controls.png',dpi=160);plt.close(fig)
    rows=[json.loads(x) for x in (DATA/'fred/sequence195/forecasts.jsonl').open(encoding='utf-8')]
    evidence=s['input_evidence'];turn={r['method']:r for r in s['table'] if r['horizon']==.4 and r['group']=='turn'}
    turns=turn.get('event',{}).get('valid',0)
    bg_rows=[]
    for name in ['visible194','empty_midpoint194']:
        q=load(WORK/f'output/stage103_background_motion_audit_20261022/{name}.json');v=np.array([r['shift_xy_px'] for r in q['records'] if r['accepted']]);consistency=float(np.linalg.norm(v.sum(0))/np.linalg.norm(v,axis=1).sum())
        bg_rows.append([name,len(v),num(q['summary']['median_accepted_step_px']),num(consistency)])
    extra=f'''<section><h2>100：固定参数迁移到更长的新195号</h2>
    <p>该录像依据官方框尺寸和连续标注时长选取；在读取195事件或预测误差前冻结0—17.12秒。保持99逐桨平方和、10次增益门槛、RLS、方向、模式、训练KF和全部对照，没有针对195调参。是可见性/尺寸条件下的新录像迁移，不是完整官方盲测。</p>
    <p>实际{completion['ticks']}个20ms时刻、{completion['events']:,}个原生事件、{completion['position_updates']}次过去位置更新。四路新鲜频率状态{evidence['four_frequencies_ticks']}次，四路当前测量{evidence['four_measured_current_ticks']}次，合格连续频率{evidence['relative_frequency_current_ticks']}次；增益反馈最高{evidence['max_gain_updates']}次，物理分支实际启用{evidence['active_physical_ticks']}次。1秒独立事件前缀通过。</p>
    <p>0.4秒共有{mm['event']['valid']}个有效锚点，其中{turns}个按提供框定义的变向锚点。{verdict} 表中所有方法的过去位置、事件、输出覆盖一致，未来原始框仅评分使用，未插值。</p>
    {table(['方法','有效锚点','候选目标','FDE像素','实际提供标签ADE像素'],summary_rows)}
    <img src="official195_fixed_controls.png"><img src="four_rotor_learning100.png">
    <p>增益图单位为图像域有效系数，不是质量、风速或牛顿推力。末段系数变平与当前输入/反馈更新中断有关，不能据此认定参数收敛。长片段能否完成校准与输入变化激励有关，不能只看样本时长或把共用历史收益算成转速收益。原论文也利用转速差驱动姿态；当前主线尚未完整复现四路差分力矩到未来倾斜的模型。</p>
    <video controls preload="metadata" src="../stage100_official195_longer_20261022/fred/sequence195/native_predictions.mp4"></video>
    <p><a href="../stage100_official195_longer_20261022/protocol100.json">100冻结协议</a> · <a href="../stage100_official195_longer_20261022/scores100.json">全部控制/分组</a></p></section>
    <section><h2>101：93互补预测已有Python调用接口</h2><p><code>from complementary_raw_api import forecast_raw</code> 可隔离子进程运行93。干净父进程连续两次真实RAW，共24父/48原生时刻，当前观测、55路及学习状态精确93；父I/O正常、标准库导入、非RAW拒绝、已有输出保护通过。原 <code>raw_forecast_api</code> 仍85。同步接口含冷启动，不代表实时/异步能力。</p><p><a href="../../event_drone_forecast/research/stage101/README.md">用法</a> · <a href="../stage101_complementary_API_20261022/checks101.json">实际检查</a></p></section>
    <section><h2>102：195下载恢复与版本完整性</h2><p>195首次长连接传输被截断，留下800079695字节未通过校验的原生残片，未用于预测，失败保留。改为32MB分段续传，每段核对206范围/长度并保存本地SHA，最终解压文件{native['native_bytes']:,}字节，官方ZIP成员CRC通过并记录整文件SHA。最终所需事件压缩成员约{native['compressed_bytes']/1e6:.1f}MB，实际网络流量另含首次失败重传，没有下载RGB/PNG整包。</p><p>194/195原生事件现均已有，暂不必重复下载这两份事件。路径：<code>{native['path']}</code>。当前官方HDF5与旧版E盘RAW/标注分开保留。</p><p><a href="../stage100_official195_longer_20261022/download_first_attempt_failure.json">首次失败</a> · <a href="../stage102_resumable_native_download_20261022/completion102.json">完整性记录</a></p></section>
    <section><h2>103：背景运动只作诊断</h2><p>复用OpenCV相位相关，使用原生ON/OFF计数图并排除目标ROI，没有G重建。194两个窗口的单步小位移方向一致性低；不能据此断言相机在平移，也没有直接从无人机轨迹中扣除背景。该方法没有独立光流真值，树叶、噪声、掩膜、视差仍会影响结果。</p>{table(['窗口','接受帧对','单步位移中位px','方向一致性'],bg_rows)}<p><a href="../stage103_background_motion_audit_20261022/checks103.json">诊断记录</a></p></section>'''
    previous=OUT/'第八十八至九十九阶段实验总览.html';text=previous.read_text(encoding='utf-8').replace('第88—99','第88—103')
    text=text.replace('<section><h2>仍需优先解决</h2>',extra+'<section><h2>仍需优先解决</h2>')
    report=OUT/'第八十八至一百零三阶段实验总览.html';report.write_text(text,encoding='utf-8');links=Links();links.feed(text)
    for v in links.local:
        if v:assert (report.parent/v).resolve().is_file(),v
    save(OUT/'final_checks100_103.json',dict(longer_test100=s['input_evidence'],new_recording_event_gain_vs_controls=relative,
        event_gain_criterion_met=event_supported,API101=api,data102=native['native_sha256'],background103=background,
        report_links_checked=len(links.local),default_promoted=False))
    # Current entry information precedes preserved historical progress.
    status=WORK/'PROJECT_STATUS_20261022.md';text=status.read_text(encoding='utf-8')
    text=text.replace('**88—99已完成；100在提取新195原生事件并固定参数验证17.12秒连续段。',
        '**88—103已完成；无本项目实验后台。')
    text=text.replace('第88—99阶段总览','第88—103阶段总览').replace('第八十八至九十九阶段实验总览.html','第八十八至一百零三阶段实验总览.html')
    intro=f'''\n100新增195固定17.12秒长片段复核完成：{evidence['four_measured_current_ticks']}个四路当前测量、{evidence['max_gain_updates']}次最大增益反馈、{evidence['active_physical_ticks']}个物理启用时刻；0.4s真实频率{mm['event']['mean_fde']:.3f}px、恒定{mm['constant']['mean_fde']:.3f}、零修正{mm['zero']['mean_fde']:.3f}、训练KF{mm['kf_tuned']['mean_fde']:.3f}、新回退{mm['fallback_router']['mean_fde']:.3f}。以完整表和组别为准，不宣称机械RPM真值。195事件经102恢复已校验，不必再下载。101新complementary_raw_api运行93、旧API仍85；两次实际调用通过。103背景检查未证明明显相机平移，未补偿。以下88—99记录保留历史语境。\n\n'''
    if '100新增195固定17.12秒长片段复核完成' not in text:text=text.replace('# 当前继续状态：2026-10-22\n','# 当前继续状态：2026-10-22\n'+intro,1)
    status.write_text(text,encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8').replace('88—99总览/视频','88—103总览/视频').replace('第八十八至九十九阶段实验总览.html','第八十八至一百零三阶段实验总览.html')
    api_text='''\n新[complementary_raw_api.forecast_raw](complementary_raw_api.py)可用标准库父进程隔离调用93，两次实际RAW/全部55与父I/O检查通过。[接口说明](research/stage101/README.md)。旧raw_forecast_api仍85。当前官方194/195原生事件均已在output内校验保存，暂不必重复下载；100固定195长段对照见总览，未修改默认或证明机械RPM。\n\n'''
    if '新[complementary_raw_api.forecast_raw]' not in text:text=text.replace('# Event Drone Forecast\n','# Event Drone Forecast\n'+api_text,1)
    readme.write_text(text,encoding='utf-8')
    work=WORK/'WORKLOG.md';text=work.read_text(encoding='utf-8')
    if '## 128. 100—103长片段、下载恢复和接口完成' not in text:
        text+=f'''\n## 128. 100—103长片段、下载恢复和接口完成\n\n- 100在195事件/预测误差前冻结最长连续提供框0—17.12s，保持99全部参数/对照。实际{completion['ticks']}时刻/{completion['events']}事件，四路当前{evidence['four_measured_current_ticks']}、频率连续性{evidence['relative_frequency_current_ticks']}，最大增益反馈{evidence['max_gain_updates']}、物理启用{evidence['active_physical_ticks']}。1秒事件前缀通过；0.4s有效{mm['event']['valid']}锚点，event{mm['event']['mean_fde']:.6f}/constant{mm['constant']['mean_fde']:.6f}/zero{mm['zero']['mean_fde']:.6f}/KF{mm['kf_tuned']['mean_fde']:.6f}/fallback{mm['fallback_router']['mean_fde']:.6f}。完整组别/全部过去打乱保留，不按录像挑专家，不宣称完整官方或机械RPM成绩。\n- 首次195长连接留下800079695字节不完整原生残片，大小/CRC未通过，预测尚未启动。102改32MB压缩成员分段、范围/大小及本地SHA保存可续传，含一次URLError重试；最终压缩963819470/native1135318234字节、ZIP CRC/整文件SHA通过才启动100。无RGB/PNG整包，实际流量含失败重传。194/195事件现均在本项目，不需用户重复下载这两份事件。\n- 101新complementary_raw_api调用93，复用87的错误/结果类型与同步子进程隔离。干净父进程两次真实RAW24父/48原生当前/55/学习全精确93，父I/O正常；stdlib导入、非RAW拒绝和不创建输出、已有输出保护通过。旧85 API不改，不是异步实时服务。\n- 103OpenCV相位相关对194原生计数背景排除ROI；可见/空中段单步位移中位0.477/0.527px，但方向一致性约0.082/0.013，很低，未据此断言相机平移或扣补偿；无独立光流真值。原预测均未修改。\n'''
    work.write_text(text,encoding='utf-8')
    prior=load(WORK/'output/checkpoints/latest_handoff.json');assert sha(prior['archive'])==prior['archive_sha256']
    summary=dict(prior);summary.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=103,completed=list(range(19,104)),
        execution_status='checkpoint_ready',research_objective_status='incomplete',in_progress=[],experiments_running=[],latest_report=str(report),
        latest_full_forecast_report=str(report),latest_full_forecast_stage=93,latest_native_diagnostic_stage=100,latest_API_stage=101,
        experimental_API_entry=str(ROOT/'core/complementary_raw_api.py'),compatibility_API_still_targets=85,
        data195=native['path'],new_recording_event_gain_criterion_met=event_supported,default_promoted=False,
        new_thrust_input_enabled=True,new_thrust_input_scope='Original78 relative-periodic paths retained in RAW93; per-rotor99/100 native-benchmark inputs not promoted to RAW93.',
        prior_archive=prior['archive'],prior_archive_sha256=prior['archive_sha256'],
        next_action='Use completed195 test and full controls to decide before further tuning.194/195 native events already available, versions isolated. Investigate per-rotor frequency differences/attitude lead (original paper), geometry uncertainty, persistent excitation and real source-time latency; do not keep fitting these fixed targets.93 fallback still worsens some own side/extra4, no default promotion or mechanical-RPM claims.')
    files=set()
    with zipfile.ZipFile(prior['archive']) as z:
        manifest=json.loads(z.read('SNAPSHOT_MANIFEST.json'))
        for n,h in manifest.items():
            assert hashlib.sha256(z.read(n)).hexdigest()==h
            p=WORK/n
            if p.is_file():files.add(p)
    for stage in range(100,104):files.update(p for p in (ROOT/f'research/stage{stage}').rglob('*') if p.suffix in ['.py','.md'] and '__pycache__' not in p.parts)
    files.update([Path(__file__),ROOT/'tools/plot_input_learning.py',ROOT/'core/complementary_raw_api.py',status,readme,work])
    for pattern in ['stage100_*','stage101_*','stage102_*','stage103_*','stage93_RAW_complementary_*']:
        for folder in (WORK/'output').glob(pattern):files.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.json','.html','.txt','.md','.log'] and p.stat().st_size<2_000_000)
    archive=WORK/'output/checkpoints'/('research_through_stage103_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    manifest={p.relative_to(WORK).as_posix():sha(p) for p in sorted(files)};inside=dict(summary);inside.pop('archive',None);inside.pop('archive_sha256',None)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(WORK).as_posix())
        z.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));z.writestr('HANDOFF_CHECKPOINT.json',json.dumps(inside,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True)
    save(WORK/'output/checkpoints/latest_handoff.json',summary);save(WORK/'output/checkpoints/latest_progress_20261022.json',summary)
    save(OUT/'delivery88_103.json',dict(report=str(report),archive=str(archive),files=len(files),verified=True))
    print(json.dumps(dict(last_stage=103,report=str(report),archive=str(archive),files=len(files),verified=True),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
