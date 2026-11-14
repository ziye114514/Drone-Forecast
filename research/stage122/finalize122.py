"""Verify delivery, update handoff and create a checksummed recovery snapshot."""
from pathlib import Path
import datetime,hashlib,json,sys,zipfile
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage122')]
from delivery122 import OUT,load,save
import study122

def main():
    study122.verify();checks=load(OUT/'delivery_checks122.json');assert checks['actual_RAW122']==330 and checks['new_overlay_decoded']==200
    report=Path(checks['report'])
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.urls=[]
        def handle_starttag(self,tag,attrs):self.urls.extend(v for k,v in attrs if k in ['src','href'] and v)
    parser=Links();parser.feed(report.read_text('utf-8'));local=0
    for url in parser.urls:
        if urlparse(url).scheme or url.startswith('#'):continue
        assert (OUT/unquote(url.split('#')[0])).resolve().is_file(),url;local+=1
    for m in range(4):assert load(ROOT.parent/f'output/stage119_partial_conics_20261023/clear6_mask{m}_prefix/prefix119.json')['all_states_forecasts_exact']
    lead='''## 119—122：缺桨部分模型、因果自检与实际RAW交付

2026-10-23。[报告与实际RAW预测视频](<D:/drone forecast/output/stage122_RAW_validated_structure_20261023/第一百一十九至一百二十二阶段缺桨与模型自检报告.html>)；[122复现/研究说明](<D:/drone forecast/event_drone_forecast/research/stage122/README.md>)。最新实际RAW122，入口`run_validated_structure_forecast.py`，默认54/API93不改。

- 118上一轮视频放大、交接及归档已完成。118归档2629文件SHA/CRC核验通过，没有遗失先前112—118负结果。
- 119：原五帧完整锁定后，用当前三椭圆周长约束H8、已学半径/比例固定；24种一一关联，只保留真实三个输入，不插入第四个观测；比较原机体hold、相同三中心仿射、三椭圆、加局部条件协方差/五份形状散布。当前三个拟合椭圆完整周长是近似支持，不是真实完整弧段。
- 同一真实RAW清晰段四mask各200、缺桨90；四个独立100前缀、自然侧视100、负例30，共1330缓存时刻；合成五情形各150，共750。前缀/身份断开/三点禁止首次锁定/6种顺序不变性/隔离关闭partial时原117路径精确通过。20ms帧不等于实时。
- 真实四mask中心一致性RMSE：bodyhold11.520355、affine3/自检2.264079、conic3/guarded3为3.980205px；0.4秒相对完整模型未来估计误差18.938919/13.527203/16.858738px。完整四桨参考不是真COM，360中心/340未来比较均来自同一段重叠数据，不能报作真实精度或四个独立飞行。
- 条件协方差和初始形状散布在真实四mask全放行，未识别模型偏差；不得当成已校准可靠性保证。三中心仿射在真实数据胜过三椭圆，理想投影相反。该负结果是121动机，未按目标挑参数。
- 121：四桨当前有效帧轮流隐藏一个观测，比较三个中心/椭圆估计中心与当前完整模型中心，四次平均平方误差按.2s EMA更新，至少5次有效检查后按严格过去分数选择；三输入时刻不更新验证、没有隐藏四输入读取。选择观测进入单一CenterBank，不在切换时重置速度。
- 121真实360缺桨时刻全选仿射，与始终仿射相同；理想静止/平移/变向各55缺桨时刻全选三椭圆，位置相对RMSE.576214/.705518/.839012px，突然变向.4s仍18.655862px。扁平突变与缺两个桨仍回退。真实同段动态切换收益尚未证明。121四100/五75独立缓存前缀严格。
- 120与122各实际重读RAW330父时刻（clear200/prefix100/negative30）。122执行输入仅RAW/旧权重、隔离worker禁止output缓存/标注读取；旧68系数/预测精确120，新增选择器状态/系数精确121，旧64经120传递精确保留。事件cutoff、body控制、实际前缀、负例、230源视频＋200新增叠加帧解码通过。122仍明显超20ms；本轮没有新RPM/推力收益证明。
- 保留核验修正：120初始检查错误要求每个启动tick均已初始化64路，改为逐项保持所有已存在旧分支，正例全段并集64；旧脚本与哈希/修正理由保留。119完整路径检查误认为首65无自然三桨，已改为只在隔离对照关闭新增partial，算法结果不变。119冻结说明edge_on误写35—104，冻结代码45—99，按55时刻报告；其中心扁平比.095—.150、最小椭圆轴比.114—.194，不是充分极端成线测试。
- 研究边界：已知共面曲线位姿、模型部分特征跟踪、在线跟踪/KF融合、交叉检验均已有；EVPeriscope明确原则上三个桨心足够。创新候选收敛为“事件桨盘模型失配的逐桨隐藏自检及缺桨观测选择”，尚不能证明首创/泛化，也不能称前沿精度。查阅链接见报告和122README；本轮未运行/移植LOPET作者MATLAB系统。

下一步冻结122，在此前未调参的自采连续RAW验证清晰→部分退化→恢复；按录像分组，分别评价中心一致性/人工机体中心/未来框、错误更新率和恢复时间。需要在同一真实录像验证选择器胜过两种固定策略，若真实一直仿射最好应简化。之后裁剪最小路径再测速度。未修改yolov10-main，E盘只读，无G/新依赖。本轮所有实验进程结束。

以下118及更早为历史。

'''
    status=ROOT.parent/'PROJECT_STATUS_20261022.md';text=status.read_text('utf-8')
    if not text.startswith('## 119—122：'):status.write_text(lead+text,encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text('utf-8')
    intro='''2026-10-23最新实验122：[缺桨与模型自检报告/RAW视频](<D:/drone forecast/output/stage122_RAW_validated_structure_20261023/第一百一十九至一百二十二阶段缺桨与模型自检报告.html>)、[复现与研究边界](research/stage122/README.md)。新入口`run_validated_structure_forecast.py`，实际RAW330时刻通过，保留原68并加因果自检。真实特征删除一致性11.52→2.26px不是真COM精度，侧视/实时仍未解决；默认54/API93不改。以下118及更早为历史。

'''
    if not text.startswith('2026-10-23最新实验122'):readme.write_text(intro+text,encoding='utf-8')
    log=ROOT.parent/'WORKLOG.md';text=log.read_text('utf-8')
    if '## 134. 119—122：' not in text:log.write_text(text+'\n\n## 134. 119—122：部分桨盘观测、模型适配性自检与实际RAW\n\n'+lead.split('\n\n',1)[1].split('以下118及更早为历史。')[0],encoding='utf-8')
    cp=ROOT.parent/'output/checkpoints';h=load(cp/'latest_handoff.json');prior=Path(h['archive']);assert study122.sha(prior)==h['archive_sha256']
    archive=cp/f'research_through_stage122_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.zip'
    h.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),last_completed_stage=122,completed=sorted(set(h['completed']+list(range(119,123)))),
        latest_full_forecast_stage=122,latest_cached_forecast_stage=121,latest_diagnostic_stage=121,latest_RAW_geometry_stage=122,
        latest_report=str(report),latest_full_forecast_report=str(report),experimental_RAW_entry=str(ROOT/'run_validated_structure_forecast.py'),experimental_RAW_stage=122,
        latest_checks=str(OUT/'delivery_checks122.json'),in_progress=[],experiments_running=[],default_promoted=False,
        completed_scope='119 partial3-conic/affine controls;120 RAW68;121 past full-view leave-one-out adequacy selection;122 RAW69 with selector.',
        completion_scope='Narrow innovation candidate implemented with controls; real consistency only, synthetic truth separately. No blind efficacy/trueCOM/extreme-side/realtime proof.',
        original64_numerically_preserved=True,stable_high_accuracy_realtime_achieved=False,independent_RPM_gain_proven=False,
        next_action='Freeze122 and evaluate unused continuous self-collected RAW learning full structure then naturally losing partial rotors and recovering. Group by video, compare fixed affine/fixed conic/body fallback/validated selector, score real body center separately from four-conic consistency. Simplify if real always affine; only then optimize minimal runtime.',
        archive=str(archive),archive_sha256=None,prior_archive=str(prior),prior_archive_sha256=h['archive_sha256'],zip_verified=False,
        source_summary_files=[str(status),str(log),str(readme),str(ROOT/'research/stage122/README.md')],report_local_links_checked=local,
        archive_scope='Verified118 snapshot plus119-122 code, protocols, JSON results and figures/current handoff. RAW/weights/JSONL/videos remain at original paths. Embedded handoff excludes self archive SHA.')
    h['counts_by_stage'].update({'119':dict(cached_ticks=1330,synthetic_frames=750,real_masks=4,distinct_real_sequences=1),'120':dict(RAW_parent=330,original_branches=64,new_branches=4),
        '121':dict(cached_actual_ticks=800,synthetic_frames=750,actual_prefix_cached_ticks=775),'122':dict(RAW_parent=330,original_branches=68,new_branches=1,original_preview_frames=230,overlay_frames=200)})
    h['retained_failed_attempts']=list(dict.fromkeys(h['retained_failed_attempts']+['119 conditional covariance/learned-shape spread admitted all actual feature-mask model biases; affine consistency2.264px better than conic3.980px',
        '120 verifier startup-all64 assertion repaired, inference unchanged; original source+SHA kept','119 isolation test confused native3 observations with allfour; partial-only isolation corrected',
        '119 frozen-prose edge_on interval mismatch35..104 vs frozen code45..99; reported actual55',
        '121 selector real always affine, ideal normal always conic; no same-real-sequence switching superiority proven']))
    save(cp/'latest_handoff.json',h)
    files=[status,log,readme,cp/'latest_handoff.json',ROOT/'run_partial_structure_forecast.py',ROOT/'run_validated_structure_forecast.py']
    folders=[ROOT/'research'/f'stage{s}' for s in range(119,123)]+[ROOT.parent/'output'/n for n in ['stage119_partial_conics_20261023','stage120_RAW_partial_structure_20261023','stage121_causal_model_validation_20261023','stage122_RAW_validated_structure_20261023']]
    for folder in folders:
        files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.jsonl','.mp4'] and p.name not in ['recovery_manifest122.json','final_checks122.json'])
    contents={}
    with zipfile.ZipFile(prior) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not name.endswith('/'):contents[name]=z.read(name)
    contents.update({p.relative_to(ROOT.parent).as_posix():p.read_bytes() for p in files});manifest={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    save(OUT/'recovery_manifest122.json',dict(archive=str(archive),files=manifest));contents[(OUT/'recovery_manifest122.json').relative_to(ROOT.parent).as_posix()]=(OUT/'recovery_manifest122.json').read_bytes()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in contents.items():z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    digest=study122.sha(archive);h.update(archive_sha256=digest,zip_verified=True,execution_status='checkpoint_ready');save(cp/'latest_handoff.json',h)
    save(OUT/'final_checks122.json',dict(archive=str(archive),archive_sha256=digest,verified_files=len(contents),local_links_checked=local,latest_full_RAW_stage=122,
        actual_RAW120_and122=660,original64_preserved_via120=True,old68_exact120=True,selector_exact121=True,decoded_original_video_frames=460,new_overlay_frames=200,
        no_default_promotion=True,real_COM_accuracy_proven=False,extreme_side_recovery_achieved=False,realtime_achieved=False))
    print(json.dumps(dict(report=str(report),archive=str(archive),verified_files=len(contents))),flush=True)
if __name__=='__main__':main()
