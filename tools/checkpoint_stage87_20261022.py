"""Deliver verified process-safe API and extend the86 recoverable checkpoint."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import datetime,hashlib,html,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage87'))
import study87
load=study87.load;save=study87.save;sha=study87.sha;OUT=study87.OUT
class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src'] and value and not urlsplit(value).scheme:self.local.append(unquote(value.split('#')[0]))
def run():
    study87.verify();checks=load(OUT/'checks87.json');assert checks['parent']==24 and checks['native10']==48
    assert checks['all49_exact85'] and checks['two_calls_same_parent'] and checks['parent_file_IO_safe'] and checks['stdlib_only_API_import']
    inventory=load(OUT/'process_inventory_20261022.json');assert inventory['matching_project_python_count']==0
    prior=load(BASE/'checkpoints/latest_handoff.json');assert prior['last_completed_stage']==86 and sha(prior['archive'])==prior['archive_sha256']
    full=Path(prior['latest_report']);report=OUT/'第八十七阶段Python接口与连续RAW调用验证.html'
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['case'],r['parent'],r['all_current49_exact85'],r['parent_IO_after_child_ok'],f"{r['wall_including_cold_start_s']:.2f}"])+'</tr>' for r in checks['table'])
    report.write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>87：Python接口连续RAW验证</title><style>body{{font:17px/1.75 "Microsoft YaHei",sans-serif;max-width:1000px;margin:35px auto;padding:0 20px;color:#253248}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4df;padding:10px}}.lead{{background:#e6f3ef;padding:18px}}a{{color:#235eaa}}</style>
<h1>87：可连续调用的Python接口</h1><p class="lead">同一个父进程连续两次调用真实RAW，每次退出后正常读取其他项目JSON；24父/48原生窗口的当前观测与全部49预测精确85。新接口解决进程/审计生命周期问题，没有改变预测算法，也不是异步实时服务。</p>
<p>此前83首版在run返回、动态命名空间清理后进行文件I/O触发审计钩子NameError。现在用独立子进程运行原85，SDK和推理审计仍在子进程；父调用方只需标准库。原CLI和所有冻结数值保持。</p>
<table><tr><th>片段</th><th>时刻</th><th>当前/49一致</th><th>返回后父文件I/O</th><th>含冷启动墙钟秒</th></tr>{table}</table>
<p>另验证干净进程导入API不加载SDK/NumPy/OpenCV/Torch、已有目录不覆盖、非RAW拒绝且不创建输出。两段仅各0.24秒的接口冒烟控制，不能视为新精度或长期稳定性评测。每次包含原冷启动成本；当前接口同步，无取消/后台功能。</p>
<p><b>预测效果与速度仍以85/86为准。</b>清晰片段完整单步中位约493→387ms，额外2号221→235ms变慢；原预测误差/失败保持，尚未实现稳定高精度实时。下一步需要真正有界队列、异步源时间戳更新和延迟/跳帧下的新因果RAW验证。</p>
<p><a href="../{full.parent.name}/{full.name}">完整计算优化报告/四段预测视频</a> · <a href="checks87.json">87接口检查</a> · <a href="protocol.json">冻结协议</a> · <a href="../../event_drone_forecast/research/stage87/README.md">API用法</a> · <a href="../../event_drone_forecast/raw_forecast_api.py">接口源码</a> · <a href="clear6_process.log">清晰子进程日志</a> · <a href="side6_process.log">侧视子进程日志</a> · <a href="../../WORKLOG.md">工作日志</a></p></html>''',encoding='utf-8')
    parser=Links();parser.feed(report.read_text('utf-8'))
    for value in parser.local:assert (report.parent/value).resolve().is_file(),value
    readme=ROOT/'README.md';text=readme.read_text('utf-8')
    insertion='''## Python调用接口（87）

使用[raw_forecast_api.forecast_raw](raw_forecast_api.py)可在同一调用方进程连续运行RAW；它隔离子进程内的SDK与审计钩子，返回路径/完成元数据，不污染父进程文件I/O。接口为同步运行，每次含原冷启动，不等于异步实时管线。[用法和边界](research/stage87/README.md)、[实际两次调用检查](../output/stage87_process_safe_api_20261022/checks87.json)。原85算法与全部49保持。

'''
    if '## Python调用接口（87）' not in text:text=text.replace('## 方法与本轮改动',insertion+'## 方法与本轮改动',1)
    text=text.replace('最新79–86完成：','最新79–87完成（数值/性能79–86，加接口87）：',1);readme.write_text(text,encoding='utf-8')
    status=Path(prior['current_status_document']);text=status.read_text('utf-8');heading=text.splitlines()[0]
    update=f'''\n**87接口验证已完成；完整预测/性能仍是85/86。** [接口报告](</{report.as_posix()}>)。新增`event_drone_forecast/raw_forecast_api.py`，同一父进程两次真实RAW返回后文件I/O正常，24父/48原生当前/全部49精确85，干净导入只需标准库，已有输出保护/非RAW拒绝通过。同步子进程边界不是异步实时服务，不提高原预测精度。原79–86报告、所有失败和性能限制保留。\n'''
    if '**87接口验证已完成' not in text:text=text.replace(heading,heading+'\n'+update,1)
    status.write_text(text,encoding='utf-8')
    worklog=PROJECT/'WORKLOG.md';text=worklog.read_text('utf-8')
    if '## 116. 87独立进程API真实连续调用完成' not in text:
        text+='''\n## 116. 87独立进程API真实连续调用完成\n\n- 在86已核验归档后继续处理常驻服务前的审计生命周期边界。新增raw_forecast_api.py，只用标准库以独立子进程调用冻结85；不更改85算法，不向子进程传入缓存/标签。成功返回路径/元数据，失败日志/部分结果保留，已有输出不覆盖。同步运行含原冷启动，不是实时服务。\n- 73583退出0。一个父进程连续clear22–22.24/side108–108.24两次真实RAW，各12父/24原生，当前/全49/状态精确85对应前缀；每次返回后父进程读取其他项目JSON正常。干净导入无SDK/NumPy/OpenCV/Torch，非RAW拒绝且不创建输出、已有输出保护通过。没有新准确率或长期稳定性主张。\n- 接口说明/报告/源码和checks87已保存。最后本项目进程0，新快照延伸86完整清单并含87源/控制/元数据；RAW/权重/完整JSONL/视频仍在原路径。下一步才是真正有界队列、异步时间戳和延迟/跳帧因果验证，目标尚未完成。\n'''
    worklog.write_text(text,encoding='utf-8')
    summary=dict(prior);summary.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=87,
        completed=list(range(19,88)),execution_status='checkpoint_ready',in_progress=[],experiments_running=[],
        completed_scope=prior['completed_scope']+'87 independent-process API and real two-call parent IO validation.',
        latest_API_stage=87,experimental_API_entry=str(ROOT/'core/raw_forecast_api.py'),latest_API_report=str(report),
        API_checks=checks,latest_full_forecast_stage=85,latest_cached_forecast_stage=84,latest_full_forecast_report=str(full),
        local_report_links_checked=prior['local_report_links_checked']+len(parser.local),inventory_file=str(OUT/'process_inventory_20261022.json'),
        prior_archive=prior['archive'],prior_archive_sha256=prior['archive_sha256'],
        next_action='Use isolated raw_forecast_api for caller integration. True bounded queues/async source timestamps '
            'and delayed/out-of-sequence updates remain next; run actual causal dropped/delayed RAW tests before realtime claims. '
            'Keep85 numerical/error baselines and constant/history controls; no event RPM benefit is established.')
    files=set()
    with zipfile.ZipFile(prior['archive']) as zipped:
        oldmanifest=json.loads(zipped.read('SNAPSHOT_MANIFEST.json'))
        for name,value in oldmanifest.items():
            assert hashlib.sha256(zipped.read(name)).hexdigest()==value
            p=PROJECT/name
            if p.is_file():files.add(p)
    files.update([ROOT/'core/raw_forecast_api.py',Path(__file__),readme,status,worklog])
    files.update(p for p in (ROOT/'research/stage87').rglob('*') if p.suffix in ['.py','.md'] and '__pycache__' not in p.parts)
    for pattern in ['*.json','*.html','*.log']:files.update(p for p in OUT.rglob(pattern) if p.stat().st_size<2_000_000)
    archive=BASE/'checkpoints'/('research_through_stage87_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True)
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints/latest_progress_20261022.json',summary)
    print(json.dumps(dict(last_stage=87,files=len(files),archive=str(archive),verified=True),ensure_ascii=False),flush=True)
if __name__=='__main__':run()
