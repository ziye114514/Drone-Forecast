"""Finalize actual runtime evidence, artifact checks and handoff once."""
from pathlib import Path
import argparse,ast,datetime,hashlib,json
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
parser=argparse.ArgumentParser();parser.add_argument('--short-remaining',type=int,required=True);parser.add_argument('--weekly-remaining',type=int,required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/flip_consistency_20261026'
assert not (OUT/'artifact_checks.json').exists(), 'Finalization already recorded.'
comp=json.loads((OUT/'raw_smoke/completion.json').read_text('utf-8'))
protocol=json.loads((OUT/'raw_smoke/protocol.json').read_text('utf-8'))
assert comp['recipe_id']=='accumulated_v3_consistency' and comp['sources_unchanged']
assert not comp['labels_read'] and not comp['output_caches_read'] and not comp['center_ensemble_enabled']
assert comp['frame_descriptor_mismatches']==0
assert protocol['runtime']['detector_sha256']=='7e356b8b2e74eaf5f37eecdc9554071daaa78816d71b66b94bb2f2bdf92c8395'
runtime=dict(status='passed_short_entry_smoke',tracking_updates=comp['tracking_updates'],publication_slots=comp['publication_slots'],
    valid_publications=comp['valid_publications'],processing_median_ms=comp['processing_median_ms'],
    observation_age_p95_ms=comp['observation_age_p95_ms'],source_unchanged=comp['sources_unchanged'],detector_sha256=protocol['runtime']['detector_sha256'],
    precision='FP32 graph',scope='1.2s paced RAW, not proof of sustained live-camera behavior or fresh paired accuracy.')
(OUT/'runtime_status.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
decision=json.loads((OUT/'retention_decision.json').read_text('utf-8'));decision['raw_runtime_smoke']='passed'
(OUT/'retention_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
snap=OUT/'source_snapshot'
files=['consistency_detector.py','tools/train_flip_consistency.py','tools/check_flip_consistency.py','tools/probe_consistency_pair.py',
    'tools/score_consistency_stack.py','tools/record_consistency_run.py','tools/evaluate_adapted_transfer.py','tools/check_live_tracks.py','tools/finalize_consistency_run.py']
for n in files:
    p=ROOT/n;ast.parse(p.read_text(encoding='utf-8'));d=snap/n;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(p.read_bytes())
manifest=json.loads((ROOT/'configs/live_tracks.json').read_text('utf-8'))
for n in ['baseline_v0','accumulated_v1','accumulated_v3_consistency']:
    e=manifest['tracks'][n];assert hashlib.sha256((ROOT/e['model']).read_bytes()).hexdigest()==e['sha256']
class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and not urlparse(v).scheme and not v.startswith('#'):self.local.append(unquote(v.split('#')[0]))
links=Links();links.feed((OUT/'index.html').read_text(encoding='utf-8'))
missing=[n for n in links.local if not (OUT/n).exists()];assert not missing,missing
checks=dict(local_links=len(links.local),missing=missing,python_syntax_files=len(files),registered_model_hashes_exact=True,plot_and_one_side_overlay_visually_checked=True)
(OUT/'artifact_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
hpath=WORK/'output/checkpoints/latest_handoff.json';handoff=json.loads(hpath.read_text(encoding='utf-8'))
handoff.update(updated_utc=now,execution_status='active goal; cumulative v1 recommended, v2/v3 tradeoff revisions available',
    latest_report=str(OUT/'index.html'),latest_live_revision='accumulated_v1',available_tradeoff_revision='accumulated_v3_consistency',
    available_tradeoff_revisions=['accumulated_v2_finetune','accumulated_v3_consistency'],
    current_jobs=[],current_goal_turn_classification='progress',quota_remaining=dict(short=args.short_remaining,weekly=args.weekly_remaining),
    next_action='Continue from accumulated_v1; preserve selectable v3 small-gain consistency model. Two fixed five-epoch matched-control arms complete; no checkpoint/alpha sweep. Next research and reuse existing SDK/optical-flow implementations to obtain quality-controlled current local displacement/velocity without smoothing real turns. Keep original position branch and geometric eligibility separation. Check quota before batches and pause at10%.')
for k in ['currentjobs','running_jobs','active_jobs']:
    if k in handoff:handoff[k]=[]
hpath.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),encoding='utf-8')
(WORK/'output/latest_live_forecast.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=flip_consistency_20261026/index.html"><a href="flip_consistency_20261026/index.html">最新优化报告</a>',encoding='utf-8')
with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write(f'''

### 同帧一致性实际完成：小收益版本接入（2026-10-26）

本目标turn为progress：完成两臂五轮训练、导出、实际原生RAW探针、完整预测回放与可切换部署。control226.623s、consistency201.481s；父权重SHA保持，测试/预测标签在训练期间禁止。最后轮合格翻转对control410/431、一致性412/431，未以牺牲匹配数清空损失。按验证fitness选择，未挑预测分数最好checkpoint。

402张同原生画面SHA精确：side100、train2 101、test3 201。固定0.4s等权macro：累计v1 68.274、相同训练量对照69.613、一致性68.127px；新约束比对照省1.486px，比v1省0.147px。side51.368→54.664（回退3.296），train2 85.181→81.591（省3.590）；test3独立13个变时域目标61.770→59.461，不能混池。无丢失旧有效发布，side多1候选。侧视工程mean门槛失败如实保留，默认仍累计v1；新accumulated_v3_consistency可显式切换，不因小收益删除。

独立107可见图均检出，中心均误6.694→6.276、medianIoU.7838→.8025；录像3的11空图误报2→1，录像5另7空图均0。弱标注/历史已暴露样本，不称全局盲测、SOTA或显著性证明。

v3公开入口1.2s实际RAW完成：65native、16,981,921events，31tracking、6pub/5valid，processing median15.522ms、ageP95102.773ms；单模型、正确SHA、source不变、无标签/缓存读取。短接线不与旧负载测速混比，不称持续实时或新增端到端精度。

会话75159(control)、69219(consistency)、14421(pair RAW)、59312/80223(transfer)、50210(smoke)均exit0。新入口default/v3/original及通用入口检查通过；图和侧视叠框已看，{len(links.local)}个本地链接通过，{len(files)}个Python语法通过。报告output/flip_consistency_20261026/index.html；WORKLOG及latest_handoff已更新，当前无本轮后台任务，goal保持active。下一步局部位移/速度观测复用，避免继续扫检测损失。额度余短{args.short_remaining}%、周{args.weekly_remaining}%。
''')
print(json.dumps(dict(runtime=runtime,artifact_checks=checks,handoff_saved=True),ensure_ascii=False))
