"""Save the actual accumulated recipe and the user's incremental policy."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUTPUT=WORK/'output';OUT=OUTPUT/'accumulated_mainline_20261026'
def read(path):return json.loads(path.read_text('utf-8'))
single=read(OUTPUT/'single_model_optimized_RAW_20261026/scores.json')
heldout=read(OUTPUT/'heldout_motion_RAW_20261026/scores.json')
smoke=read(OUT/'raw_smoke/completion.json');protocol=read(OUT/'raw_smoke/protocol.json')
assert smoke['recipe_id']=='accumulated_v1' and smoke['sources_unchanged'] and smoke['native_batches']==65
assert protocol['runtime']['detector_sha256']=='ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4'
assert smoke['scale_rollout']=='compatible' and smoke['frame_transport']=='frames'
assert not smoke['center_ensemble_enabled'] and not smoke['labels_read'] and not smoke['output_caches_read']
rows=[]
for case,r in single['results'].items():
    m=r['scores']['0.4']['metrics']
    rows.append(dict(case=case,n=m['merged_scale']['n'],baseline=m['original_model_replay']['mean'],
        plus_merged_model=m['merged_curved']['mean'],plus_scale=m['merged_scale']['mean']))
macro={k:sum(r[k] for r in rows)/len(rows) for k in ['baseline','plus_merged_model','plus_scale']}
ledger=dict(revision='accumulated_v1',adopted_for='ongoing engineering development mainline',
    user_preference='Accumulate small gains; preserve original for comparison; do not require every clip to win.',
    original_reference_is_offline_same_frame=True,development_fixed04s=rows,development_equal_recording_macro=macro,
    heldout_variable_horizon=dict(original=heldout['metrics']['original_reference']['mean'],
        accumulated=heldout['metrics']['merged_scale']['mean'],n=heldout['metrics']['merged_scale']['n'],
        horizon_range_s=heldout['actual_horizon_range_s'],pooled_into_fixed04s=False),
    policy=read(ROOT/'configs/accumulation_policy.json'),recipe=read(ROOT/'configs/live_tracks.json'),
    entry_checks=read(OUT/'entry_checks.json'),actual_recipe_smoke=smoke,
    no_new_accuracy_experiment=True,mainline_adoption_is_not_SOTA_or_generalization_claim=True)
(OUT/'ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
table=''.join(f'<tr><td>{r["case"]}</td><td>{r["n"]}</td><td>{r["baseline"]:.2f}</td><td>{r["plus_merged_model"]:.2f}</td><td>{r["plus_scale"]:.2f}</td></tr>' for r in rows)
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>累计优化主线</title><style>body{{max-width:1000px;margin:40px auto;padding:0 20px;font:17px/1.8 system-ui;color:#203047}}aside{{padding:16px;background:#eaf2fa}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cdd7e3;padding:9px}}code{{background:#eef2f7;padding:2px 5px}}a{{color:#1460a0}}</style><h1>累计版现在是后续迭代主线</h1><aside>小幅改善可以积累。原版冻结做对照；新改动先与当前累计版比较。允许透明的小幅场景回退，不再要求每个片段全部获胜。原有“未证明普遍收益”的说明继续保留，这不阻止工程主线向前推进。</aside><h2>同一组实际目标上的累计结果</h2><p>固定0.4秒、原生像素误差，数据来自已完成的实际RAW；原模型列是同画面/同时间的离线参考。</p><table><tr><th>片段</th><th>共同目标</th><th>原模型</th><th>加合并模型</th><th>再加尺度修正</th></tr>{table}</table><p>两段均等计权平均：{macro['baseline']:.2f} → {macro['plus_merged_model']:.2f} → {macro['plus_scale']:.2f}像素，累计改善{macro['baseline']-macro['plus_scale']:.2f}像素。留出录像的13个变时域目标60.99→61.77，回退约0.78像素，单独报告，不混进固定0.4秒分数。</p><h2>现在怎样运行</h2><p>推荐入口：<code>forecast_mainline.py run ...</code>，默认累计版。原版：在同一命令后加 <code>--profile live-baseline</code>。原有 <code>forecast.py --profile live</code> 的行为也保留，方便复现旧实验。</p><p>累计版包括合并模型、条件尺度、共享原生画面、启动同步；启用周期诊断时自动使用语义保持的周期包去重。暂不把空间四分之一采样或仅19图检查过的FP16强行纳入默认。</p><p>新入口已完成1.2秒RAW接线：{smoke['native_batches']}原生片、{smoke['tracking_updates']}次跟踪、{smoke['valid_publications']}/{smoke['publication_slots']}有效发布。这里只确认入口和组合正确，没有把这次重复接线当新的精度证据。</p><h2>以后怎么取舍</h2><p>看完整组合的总收益、覆盖与明显退化；每段都赢不再是必要条件。初始工程容忍范围写在政策文件，后续试验前固定。仍拒绝明显伤害预测或破坏因果性的改动。两个独立实验省下的像素可能来自同一种误差，所以要测组合后的真实增量。</p><p><a href="ledger.json">累计版本记录</a> · <a href="../../event_drone_forecast/docs/ACCUMULATED_MAINLINE.md">运行与取舍说明</a> · <a href="../optimization_resume_20261026/index.html">此前完整实验、视频和负结果</a></p></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
doc='''# 累计优化主线

按用户2026-10-26反馈：小幅收益应累计，原版用于固定对照，后续改动以累计版为起点。不再把“每个片段必须全部胜出”作为工程主线的必要条件。

推荐入口（项目根目录）：

```powershell
.\\.venv-event311\\Scripts\\python.exe event_drone_forecast\\forecast_mainline.py run `
  --config event_drone_forecast\\configs\\local.json `
  --raw E:\\dataset\\train\\6\\media\\event.raw --start 110 --duration 4 `
  --output output\\my_accumulated_run --preview
```

不加profile时选configs/live_tracks.json中的recommended，当前accumulated_v1。加`--profile live-baseline`选冻结原版。原forecast.py的live入口仍保留历史行为；也可显式使用`--profile live-accumulated`。CLI显式模型/参数覆盖仍优先，模型覆盖会写入recipe_id，真实模型SHA始终单独记录。

模型已按SHA复制到项目models目录，运行不再依赖旧output历史目录；原文件没有覆盖。模型未公开上传，第三方条款继续适用。未重新构建wheel或源码ZIP。

当前累计：单模型参数平均+条件尺度+共享原生画面+启动同步；周期诊断启用时使用无状态141的周期包去重。四分之一空间采样、FP16和双模型推理保留实验入口，不自动叠入。

同目标0.4s：side52.583→51.475→51.368px，train2 88.629→85.181→85.181px。两片段均等平均70.606→68.328→68.274px。留出段0.317—0.451s的13目标60.994→61.770px，透明保留，不和固定0.4s混池。这是工程主线选择，不是统计显著或普遍泛化的声明。

后续规则在configs/accumulation_policy.json：比较完整新组合与当前累计版，按兼容时域的同起点记录均等评价；允许小幅场景回退，报告P95和覆盖；原有因果、缺测、几何资格不放宽。初始2px/5%均值、6px/10%尾部容忍是工程选择，不是测得的噪声置信区间。政策在用户反馈后制定，不冒充此前预注册门槛。

没有额外的运动精度重跑；只用1.2s RAW确认新主线入口接线。下一步先复审已有小收益的完整组合，再继续调研定位时序一致性。不要重新扫alpha或把普通平滑当作抑制真实变向。
'''
(ROOT/'docs/ACCUMULATED_MAINLINE.md').write_text(doc,encoding='utf-8')
p=ROOT/'README.md';s=p.read_text('utf-8');line='> 当前迭代主线：使用 [forecast_mainline.py](forecast_mainline.py)，默认累计版；原版通过 --profile live-baseline 保留。详见 [累计优化说明](docs/ACCUMULATED_MAINLINE.md)。\n\n'
if line not in s:s=s.split('\n',1)[0]+'\n\n'+line+s.split('\n',1)[1];p.write_text(s,encoding='utf-8')
previous=OUTPUT/'optimization_resume_20261026/index.html';s=previous.read_text('utf-8')
banner='<aside id="accumulated-mainline-update">更新：按你的要求，累计版已作为后续优化主线，原版保留作对照。<a href="../accumulated_mainline_20261026/index.html">查看累计版与运行入口</a>。以下保留此前实验原始结论。</aside>'
if 'id="accumulated-mainline-update"' not in s:
    position=s.find('<h1>');s=s[:position]+banner+s[position:];previous.write_text(s,encoding='utf-8')
handoff=OUTPUT/'checkpoints/latest_handoff.json';d=read(handoff)
d.update(utc=datetime.now(timezone.utc).isoformat(),execution_status='active goal; accumulated development mainline established',
    experiments_running=False,current_running_jobs=[],latest_report=str(OUT/'index.html'),
    latest_live_revision='accumulated_v1',current_mainline_entry=str(ROOT/'core/forecast_mainline.py'),
    original_baseline_entry='forecast_mainline.py run --profile live-baseline',
    accumulation_ledger=str(OUT/'ledger.json'),
    next_action='Use accumulated_v1 as the next optimization baseline; preserve baseline_v0 for reference. Revisit small prior gains under the user-requested aggregate policy; no all-recordings-win veto. Then research reusable temporal localization consistency before code. Check quota before each batch; pause at remaining10%.')
(handoff).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write('\n\n### 用户纠正：小收益累计为主线（2026-10-26）\n\n'+
        '上一轮真实进展归类为progress（已训练、接线、得到新证据）。本轮额度已恢复，goal工具确认active，继续工作；用户指出不应因几像素或单段轻微回退而始终冻结原版。按新偏好将累计版作为后续工程主线，原版独立冻结对照；不等于声称所有片段改善或前沿提升。\n\n'+
        '新增forecast_mainline.py默认live-accumulated，显式live-baseline选原版；原forecast.py --profile live历史行为保持。两个ONNX按SHA复制到models标准位置，未覆盖旧文件。configs/live_tracks.json记录版本和参数，推荐accumulated_v1。累计模型与尺度同原实际目标：side52.583→51.475→51.368，train2 88.629→85.181→85.181；均等记录平均70.606→68.328→68.274，固定0.4s总益2.331px。留出13个变时域目标+0.776px透明保留，不混池。\n\n'+
        '不再每段必须胜出；看完整组合总收益、覆盖和明显退化。configs/accumulation_policy.json记录用户反馈后制定的工程容忍规则，不冒充预注册或测得统计置信度。启用周期诊断时累计版默认只做语义保持的包去重，不强制四分之一空间采样、FP16或双模型。\n\n'+
        f'新入口1.2s实际RAW65片，{smoke["tracking_updates"]}跟踪、{smoke["valid_publications"]}/{smoke["publication_slots"]}有效发布，选中accumulated_v1/正确SHA/compatible/frames，源与共享图检查通过；这是接线而非新增精度验证。无本轮运行作业，goal保持active。报告output/accumulated_mainline_20261026/index.html，后续从累计版继续。\n')
(OUTPUT/'latest_live_forecast.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=accumulated_mainline_20261026/index.html"><a href="accumulated_mainline_20261026/index.html">累计优化主线</a>',encoding='utf-8')
print(json.dumps(dict(mainline='accumulated_v1',macro04s=macro,smoke_valid=smoke['valid_publications'],goal_left_active=True),ensure_ascii=False))
