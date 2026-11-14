"""Curate NN-led release with an explicitly unsuccessful optional rotor study."""
from pathlib import Path
import ast,hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
PREVIOUS=WORK/'output/github_research_closeout_20261105/event-drone-forecast'
DELIVERY=WORK/'output/github_nn_main_20261105';OUT=DELIVERY/'event-drone-forecast'
STUDY=WORK/'output/nn_rotor166_20261105'
FROZEN=WORK/'output/stability_audit_20261031/RAW_clear_final_clean4s/source_snapshot'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,text):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,'utf-8')


def portable(s):
    s=re.sub(r'sys\.path\[:0\]\s*=\s*\[.*?\]',"sys.path[:0]=[str(ROOT/'src')]",s,flags=re.S)
    replacements={
        'from research.stage146.branches146 import':'from rotor_geometry import',
        'from research.stage147.curvature147 import':'from curvature_split import',
        'from research.stage6.silhouette import':'from body_axis import',
        'from guided148 import':'from arc_parameters import',
        'from curvature147 import':'from curvature_split import',
        'from native_axis141 import':'from native_axis import',
        'from native79 import':'from frequency_features import',
        'import native79':'import frequency_features as native79',
        'research/stage6/silhouette.py':'body_axis.py',
        'research/stage146/branches146.py':'rotor_geometry.py',
        'research/stage147/curvature147.py':'curvature_split.py',
        'research/stage148/guided148.py':'arc_parameters.py',
        'research/stage79/native79.py':'frequency_features.py',
    }
    for a,b in replacements.items():s=s.replace(a,b)
    s=s.replace("sys.path.insert(0,str(ROOT/'research/stage141'))","sys.path.insert(0,str(ROOT/'src'))")
    s=s.replace("sys.path.insert(0,str(ROOT/'research/stage79'))","sys.path.insert(0,str(ROOT/'src'))")
    return s


if __name__=='__main__':
    if DELIVERY.exists():raise FileExistsError('New delivery only; do not overwrite historical packages')
    shutil.copytree(PREVIOUS,OUT)
    for name in ['README.md','docs/RESEARCH_SNAPSHOT.md']:
        dest='docs/PREVIOUS_GEOMETRY_'+Path(name).name
        shutil.copy2(OUT/name,OUT/dest)
    provenance=[]
    # Freeze a stronger recorded NN-led runtime rather than export current pure-rotor experiments.
    for p in FROZEN.rglob('*'):
        if not p.is_file():continue
        name=p.relative_to(FROZEN).as_posix()
        if p.suffix=='.py':write(name,portable(p.read_text('utf-8')))
        else:
            q=OUT/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        provenance.append(dict(destination=name,source='recorded_NN_clear_runtime_snapshot',source_sha256=sha(p),adaptation='public imports/paths; runtime configuration handled separately'))
    p=OUT/'run_fast_structure.py';s=p.read_text('utf-8');tree=ast.parse(s)
    for node in tree.body:
        pass
    protected=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='protected')
    s=s.replace(ast.get_source_segment(s,protected),'# Private workspace guards omitted from public export.')
    tree=ast.parse(s)
    resolve=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='resolve')
    s=s.replace(ast.get_source_segment(s,resolve),'resolve(args.gpu_python,args.detector_weights,args.detector_sha256,args.cache_dir)')
    s=s.replace("p.add_argument('--gpu-python');","p.add_argument('--gpu-python');p.add_argument('--detector-weights');p.add_argument('--detector-sha256');p.add_argument('--cache-dir');")
    tree=ast.parse(s)
    # Source fingerprinting now covers the curated public source, never missing private stage paths.
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='f'
              and isinstance(n.iter,ast.List) and any(isinstance(v,ast.Constant) and v.value=='run_fast_structure.py' for v in n.iter.elts))
    source_list="sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts and 'runs' not in p.parts and 'outputs' not in p.parts)"
    s=s.replace(ast.get_source_segment(s,loop.iter),source_list)
    write('run_fast_structure.py',s)
    p=OUT/'forecast_runtime.py';s=p.read_text('utf-8').replace('e2531556ad0431d904fb0d1d432727a2e2dd9eaed9e098a6ed3d9e55f474db0b','ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4')
    s=s.replace('cfg.get(key, default)','(cfg.get(key) if cfg.get(key) is not None else default)');write('forecast_runtime.py',s)

    # Separate namespace: reproducing the auxiliary arm must not modify the default NN runtime.
    pools=[STUDY/'evidence_source_snapshot',STUDY/'gated_source_snapshot',STUDY/'source_snapshot',FROZEN,ROOT]
    pool={}
    for folder in pools:
        for p in folder.glob('*.py'):pool.setdefault(p.stem,p)
    for stem in ['rotor_geometry','curvature_split','body_axis','arc_parameters','frequency_features','native_axis']:
        pool[stem]=OUT/(stem+'.py')
    selected={};queue=['nn_rotor_gate','nn_rotor_residual']
    while queue:
        name=queue.pop()
        if name in selected or name not in pool:continue
        p=pool[name];text=portable(p.read_text('utf-8'));selected[name]=(p,text)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,ast.ImportFrom) and not node.level and node.module in pool:queue.append(node.module)
            elif isinstance(node,ast.Import):queue.extend(a.name for a in node.names if a.name in pool)
    prefix='src/event_drone_forecast/experimental_nn_aux/'
    for name,(p,s) in selected.items():
        lines=s.splitlines(keepends=True);edits=[]
        for node in ast.walk(ast.parse(s)):
            if isinstance(node,ast.ImportFrom) and not node.level and node.module in selected:
                replacement='from .'+node.module+' import '+', '.join(a.name+(' as '+a.asname if a.asname else '') for a in node.names)
            elif isinstance(node,ast.Import) and any(a.name in selected for a in node.names):
                replacement='; '.join(('from . import ' if a.name in selected else 'import ')+a.name+(' as '+a.asname if a.asname else '') for a in node.names)
            else:continue
            # Imports are short ASCII statements; columns refer to their Python source bytes.
            edits.append((node.lineno-1,node.col_offset,node.end_lineno-1,node.end_col_offset,replacement))
        for first,col,last,end,replacement in sorted(edits,reverse=True):
            assert first==last,'Unexpected multiline local import'
            lines[first]=lines[first][:col]+replacement+lines[first][end:]
        result=''.join(lines);ast.parse(result)
        write(prefix+name+'.py',result)
        provenance.append(dict(destination=prefix+name+'.py',source='166_auxiliary_dependency_snapshot' if p.is_relative_to(STUDY) else 'own_dependency',source_sha256=sha(p),adaptation='namespace-local imports; no NN base overwrite'))
    write(prefix+'__init__.py','"""Experimental166 rotor residual; no stable accuracy gain demonstrated."""\nfrom .nn_rotor_gate import CausalRotorResidualGate, evaluate_gated\nfrom .nn_rotor_residual import PreservedNNRotorResidual\n')

    # Replay needs no SDK/GPU/weights; inputs are fixed NN outputs and native features, not labels.
    keys=('t','body','coefficient','valid','force','physical','image_motion','structure','force_observed_position',
          'axis_amplitude_bridge','normal_turn_forecast','adaptive_force_memory','thrust_trend_forecast')
    info=json.loads((STUDY/'summary.json').read_text('utf-8'));cases={}
    for clip,entry in info.items():
        observations=[json.loads(s) for s in (STUDY/f'{clip}_observations.jsonl').read_text('utf-8').splitlines()]
        lookup={r['t']:r for r in map(json.loads,(Path(entry['original_NN_run'])/'predictions.jsonl').read_text('utf-8').splitlines())}
        base=[{k:lookup[o['t']][k] for k in keys if k in lookup[o['t']]} for o in observations]
        feature=[{k:o[k] for k in ('t','body','structure','geometry','rates','pooled','current_pose') if k in o} for o in observations]
        write(f'examples/nn_aux/{clip}_nn.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in base))
        write(f'examples/nn_aux/{clip}_rotor.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in feature))
        cases[clip]=dict(frames=len(base),NN_input='recorded detector-driven state, not a neural trajectory predictor',labels_included=False)
    write('examples/nn_aux/README.md','''# 固定输入NN辅助回放

三个自采旧开发片段的固定NN状态与当前原生事件特征，不含RAW、模型权重或人工终点坐标。清晰120、侧视132、另一段141行；另一段部分时刻无完整原生窗口，保留其无资格状态。

上游NN用于目标检测定位，后续位置/速度/外推为模型状态；原NN基线自身已有旧几何/周期线索，不能把它叫彻底去掉所有桨叶信息的纯方框法。此处对照的是额外阶段166辅助增量。

所有文件来自维护者自采数据派生的数值记录，随本集成版AGPL-3.0-only提供。它们不是真实RPM/COM/推力真值，不构成公开盲测。回放不执行NN或读取RAW，不能拿速度当端到端FPS。
''')
    write('nn_assist_replay.py','''"""Replay fixed detector-led NN states, optionally add experimental166 rotor residual."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def replay(nn_path,output,rotor_path=None,mode='baseline'):
    if output.exists():raise FileExistsError('Use a new output: '+str(output))
    if mode!='baseline' and rotor_path is None:raise ValueError('Rotor features required for auxiliary mode')
    if mode!='baseline':
        from event_drone_forecast.experimental_nn_aux import CausalRotorResidualGate,PreservedNNRotorResidual
        candidate=PreservedNNRotorResidual('native_evidence_only' if mode=='matched-zero' else 'native_visible');gate=CausalRotorResidualGate()
    output.parent.mkdir(parents=True,exist_ok=True);n=0;applied=0;missing=0;previous=None
    features={r['t']:r for r in map(json.loads,rotor_path.read_text('utf-8').splitlines())} if rotor_path else {}
    with nn_path.open(encoding='utf-8') as inp,output.open('w',encoding='utf-8') as dest:
        for line in inp:
            base=json.loads(line);t=base['t']
            if previous is not None and t<=previous:raise ValueError('Increasing NN times required')
            previous=t;row=base
            if mode!='baseline':
                if t not in features:raise ValueError('Missing same-cutoff rotor feature; keep explicit unavailable rows instead of skipping')
                proposed=candidate.step(base,features[t]);row=gate.step(base,proposed,bool(proposed['force']['active']))
                applied+=row['rotor_residual_fusion']['applied']
            assert row['coefficient']==base['coefficient'] and row['valid']==base['valid']
            dest.write(json.dumps(row,allow_nan=False)+'\\n');n+=1
    return dict(frames=n,mode=mode,aux_applied=applied,RAW_or_NN_inference_executed=False,labels_read=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--nn',required=True,type=Path);p.add_argument('--rotor',type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--mode',choices=['baseline','rotor-assist','matched-zero'],default='baseline')
    a=p.parse_args();print(json.dumps(replay(a.nn,a.output,a.rotor,a.mode)))
if __name__=='__main__':main()
''')
    write('nn_forecast.py','''"""NN-led release entry. NN detects/localizes; tracking/model rollout predicts position."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo',help='Saved NN state replay; no SDK/GPU/weights, not live inference')
    d.add_argument('--case',choices=['clear','side','train2'],default='clear');d.add_argument('--output',required=True,type=Path)
    d.add_argument('--rotor-assist',action='store_true',help='Experimental incremental auxiliary; no stable gain demonstrated')
    r=sub.add_parser('predict',help='RAW NN-led recorded-runtime recipe; additional166 auxiliary is not wired into live RAW')
    r.add_argument('--raw',required=True,type=Path);r.add_argument('--output',required=True,type=Path)
    r.add_argument('--start',type=float,default=0.);r.add_argument('--duration',type=float,default=2.)
    r.add_argument('--config');r.add_argument('--sdk-python');r.add_argument('--gpu-python');r.add_argument('--detector-weights');r.add_argument('--detector-sha256');r.add_argument('--cache-dir')
    r.add_argument('--preview',action='store_true');r.add_argument('--sdk-avx2',action='store_true');r.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.command=='demo':
        from nn_assist_replay import replay
        folder=ROOT/'examples/nn_aux'
        print(json.dumps(replay(folder/(a.case+'_nn.jsonl'),a.output,folder/(a.case+'_rotor.jsonl'),'rotor-assist' if a.rotor_assist else 'baseline')));return 0
    from forecast import settings
    cfg=settings(a)
    if a.start<0 or a.duration<=0:p.error('Require start>=0 and duration>0')
    command=[cfg['sdk_python'],'-X','utf8',str(ROOT/'run_fast_structure.py'),'--raw',str(a.raw.resolve()),'--output',str(a.output.resolve()),
        '--start',str(a.start),'--duration',str(a.duration),'--hz','30','--realtime30','--gpu-python',cfg['gpu_python'],
        '--detector-weights',cfg['detector_weights'],'--detector-sha256',cfg['detector_sha256'],'--cache-dir',cfg['cache_dir']]
    if a.preview:command+=['--preview']
    if a.sdk_avx2:command+=['--sdk-avx2']
    if a.dry_run:print(json.dumps(dict(command=command,NN_led=True,additional166_rotor_assist=False,realtime30_is_profile_not_FPS_guarantee=True),ensure_ascii=False));return 0
    if not a.raw.is_file():p.error('RAW not found')
    if a.output.exists():p.error('Use a new output directory')
    return subprocess.call(command,cwd=ROOT)
if __name__=='__main__':raise SystemExit(main())
''')
    scores=json.loads((STUDY/'gated_scores.json').read_text('utf-8'))
    summary={name:dict(arms={a:v['error'] for a,v in c['arms'].items()},available=c['arms']['legacy_exact']['error']['n']) for name,c in scores.items()}
    summary.update(scope='Historical fixed source-time0.4s queries; configurations differ by clip; not new package-wide RAW benchmark',independent_test=False,stable30FPS_or30px_proven=False,auxiliary_innovation_result='No stable benefit demonstrated; failed attempt under current inputs and protocol',cases=cases)
    write('examples/nn_aux/results.json',json.dumps(summary,indent=2))
    write('README.md','''# Event Drone Forecast — NN-led Research Release

**事件相机无人机检测、跟踪与二维短时预测，附桨叶结构／周期辅助实验。**

本版以已有较强的**NN检测定位＋事件图像跟踪＋状态估计／运动外推**为主体；没有训练端到端神经轨迹预测器。五圆模型、可见桨数切换和“可测贡献＋不可测合力”作为可选研究辅助保留。**额外辅助的收益很小且不稳定，本次创新尝试未达到预期，未证明其稳定提升预测精度。**

研究快照已经收尾，无持续维护承诺；不是引用论文官方代码，不宣称SOTA或稳定30px／30FPS。

## 本版选择与默认行为

- 主入口：`nn_forecast.py`。默认保留NN主干结果，不强制开启额外阶段166辅助；不按片段挑选最好分支拼成成绩。
- RAW运行源码选自已经保存的较强NN配置快照（2026-10-31清晰段运行），保持检测/跟踪/外推计算，只适配公共导入、参数路径和源码指纹。详见[版本选择](docs/NN_RELEASE_SELECTION.md)。
- 阶段166新增辅助在独立命名空间，以固定NN状态＋同截止时刻原生特征回放；尚未串进此版实时RAW入口。`--rotor-assist`只用于`demo`，不是已验证在线融合。
- 原NN基线自身已有历史几何/周期线索，不能叫完全去除所有桨叶信息的纯方框模型。这里检验的是**额外**五圆／可见性／二力增量。

## 无硬件体验

Python3.11+，在源码目录：

```bash
python -m pip install -e .
python nn_forecast.py demo --case clear --output outputs/nn_clear.jsonl
python nn_forecast.py demo --case clear --rotor-assist --output outputs/nn_clear_assist.jsonl
```

`--case`还可选`side`和`train2`。这是随附已保存NN状态的回放，**不执行NN，不处理RAW，不证明端到端FPS**。输出0.1/0.2/0.4s图像位置；候选/状态来自旧开发数据，不是真值。

对自备的兼容记录运行：

```bash
python nn_assist_replay.py --nn nn_states.jsonl --rotor native_features.jsonl --mode rotor-assist --output outputs/assisted.jsonl
python nn_assist_replay.py --nn nn_states.jsonl --rotor native_features.jsonl --mode matched-zero --output outputs/zero_control.jsonl
```

默认`--mode baseline`保持输入记录；辅助需要严格递增、同一截止时刻的NN状态与原生特征。模块保留NN的位置、速度和有效性，只修正未来外推。字段例子与来源见[数值示例](examples/nn_aux/README.md)。

## RAW入口

需要独立安装Metavision SDK、CUDA/PyTorch/ONNX Runtime GPU/Ultralytics，并提供符合[模型契约](docs/MODEL_CARD.md)的权重。此NN配置默认使用Numba JIT算子，SDK侧还需按`requirements-counts.txt`安装兼容的Numba/llvmlite。参考权重没有随包提供，也没有已整理的一键训练/下载链。

```bash
python nn_forecast.py predict --config configs/local.json --raw data/event.raw --output outputs/nn_run --start 110 --duration 2
```

配置和依赖见[旧版环境说明](docs/GETTING_STARTED.md)。`--sdk-avx2`是显式可选的本地SDK适配优化，需合法SDK及本地构建；不分发厂商二进制。`--dry-run`只打印命令。配置名`realtime30`是30Hz目标调度，不保证持续有效30Hz。

## 已有对照：辅助作用有多大

下表为既有固定NN输入、源时刻未来0.4s的终点均值误差（px），均为旧开发片段。不同片段原NN配置略有差异，**不是本次发布包重新统一跑出的端到端成绩**。

| 片段 | 可评分/预定 | 原NN主干 | 加新辅助 | 同资格已知输入置零 |
|---|---:|---:|---:|---:|
| 清晰片段 |6/6|27.555|26.889|26.854|
| 侧视 |17/19|37.110|37.653|37.671|
| 另一录像 |7/7|68.193|68.193|68.193|

清晰约改善0.67px，侧视约退化0.54px，另一段无变化。严格置零控制与完整辅助几乎相同，清晰甚至更好，因此**不能把微小变化归功于周期反推推力数值**。本版将额外辅助设为实验选项，默认保留主干。完整统计见[结果JSON](examples/nn_aux/results.json)。

本次五圆结构/可见性切换/二力辅助预测是一次**未验证成功的创新尝试**。这不等于所有相关物理建模方向已被证明错误；只说明当前数据、观测代理和实现没有产生稳定优势。没有独立机械RPM/牛顿推力/姿态真值，没有新数据盲测，也没有未知录像泛化保证。

## 流程与贡献边界

```text
原生ON/OFF → NN检测ROI／位置 → 事件图像KLT跟踪 → 状态估计 → 主干未来位置
                    ↓
     原生弧/椭圆/周期 → 四桨／三桨／两桨／侧线资格 → 当前可测贡献＋未知项
                    ↓
     额外候选外推 → 已到期自身NN位置检查 → 因果0–1权重增量（实验回放）
```

NN和KLT/滤波等基础方法来自已有工作；维护者完成了事件数据流程适配、实验组合、状态/输出接口、工程修复及对照分析。五圆后端区分实测/补点/不可观测状态，但其受控几何一致性不等于轨迹提升。隐藏桨力归未知项，不伪造隐藏RPM；不宣称已辨识真实质量、风或推力系数。

保留的[几何数值示例](docs/RESEARCH_SNAPSHOT.md)是独立附加模块，不能拿其中0.325px布局一致性作为本系统0.4秒预测误差。旧文档/视频中版本和帧率只作历史记录，以本README的范围为准。

## 来源、许可与维护

集成版沿用**AGPL-3.0-only**，保留[LICENSE](LICENSE)、[原MIT声明](LICENSES/MIT-original.txt)、[NOTICE](NOTICE.md)、[第三方许可](THIRD_PARTY_NOTICES.md)。SDK、Ultralytics、模型和数据仍各守其条款；不分发RAW、权重、SDK、论文或未授权作者仓库。AI编码工具辅助实现/文档，代码和结论责任由维护者承担。

本项目作为研究原型冻结公开，保留负结果，不承诺后续优化、论文发表或稳定实时能力。
''')
    write('docs/NN_RELEASE_SELECTION.md','''# NN主体发布选择与结论

用户要求“把最好的那个版本当GitHub发布，就是NN主体、我的思路辅助，注明辅助作用不大，算一次失败创新”。本版选择有保存运行证据的NN主干，额外166辅助默认关闭；不拿弱的新主干小胜冒充成功，也不按片段开关拼出最好分数。

“较强版本”是工程发布选择，不是对所有历史入口在同一独立测试上证明全局最优。RAW源码来自2026-10-31清晰段NN运行快照；侧视和另一录像的历史对照有调度/恢复配置差异。历史表来源阶段166固定输入诊断，发布准备仅检查打包/轻量回放，不重跑优化或测得新FPS。

额外辅助的完整值比置零没有稳定优势，侧视也未超过主干，因此公开定位为“未获得稳定预测收益的创新尝试”。原NN基线已含历史几何/周期线索；此结论针对额外辅助，不能声称彻底否定所有桨叶方法。

RAW主入口与辅助回放是两条明确的运行能力：nn_forecast.py predict运行NN主干；demo --rotor-assist或nn_assist_replay.py运行固定NN状态上的辅助。辅助尚未正式实时串联，不用回放结果声称已部署在线融合。

当前纯桨叶/五圆研究源码及旧发布包未删除、未被改成NN成绩。此版停止主动算法优化，只准备本地发布，尚未上传GitHub。
''')
    write('docs/RESEARCH_SNAPSHOT.md','''# 附加几何模块

阶段169几何观测后端仍在event_drone_forecast.rotor_observation，可运行python rotor_demo.py --output outputs/geometry.jsonl。该后端不自动接入NN主体；历史受控缺测仅验证相对完整检测布局的一致性，不是轨迹FDE/COM/真实遮挡准确率。详细历史说明见[原几何快照说明](PREVIOUS_GEOMETRY_RESEARCH_SNAPSHOT.md)。当前发布定位以根README和[版本选择](NN_RELEASE_SELECTION.md)为准。
''')
    # Fix links in the moved old root README; its original target base was the repository root.
    old=OUT/'docs/PREVIOUS_GEOMETRY_README.md';t=old.read_text('utf-8')
    old.write_text(re.sub(r'(?P<pre>!?\[[^\]]*\]\()(?P<url>[^)\s]+)(?P<end>\))',lambda m:m.group(0) if '://' in m['url'] or m['url'].startswith('#') else m['pre']+'../'+m['url']+m['end'],t),'utf-8')
    write('RELEASE_NOTES.md','''# v0.5.0-research.1 — NN主体与辅助负结果

以记录较完整的NN检测定位/跟踪/状态外推为主入口，恢复较强NN运行源码；新增可运行的阶段166可选辅助回放及同资格置零控制，保留阶段169独立几何模块。辅助未有稳定预测增益，将本次创新尝试记录为未成功。所有成绩标明历史开发数据、小样本和源时刻口径，不宣称30px/30FPS。辅助尚未正式串联实时RAW入口。集成许可AGPL-3.0-only，SDK/模型/RAW不分发，项目冻结研究状态。
''')
    p=OUT/'pyproject.toml';t=p.read_text('utf-8').replace('version = "0.4.0a2"','version = "0.5.0a1"').replace('Five-circle event-camera drone modeling and image-plane forecast research prototype','NN-led event-camera drone tracking with experimental rotor assistance');p.write_text(t,'utf-8')
    p=OUT/'src/event_drone_forecast/__init__.py';p.write_text('"""NN-led event-camera tracking and experimental rotor assistance."""\n__version__ = "0.5.0a1"\n','utf-8')
    for name in ['CITATION.cff','CITATION.bib']:
        p=OUT/name;t=p.read_text('utf-8').replace('0.4.0-alpha.2','0.5.0-research.1').replace('2026-10-28','2026-11-05').replace('Five-Circle Event-Based Drone Modeling and Forecasting','NN-led Event-Camera Drone Forecasting with Experimental Rotor Assistance');p.write_text(t,'utf-8')
    write('SNAPSHOT_VERSION.txt','v0.5.0-research.1\nNN main: recorded2026-10-31 runtime; rotor auxiliary166 opt-in replay; geometry169 independent\n')
    (OUT/'NOTICE.md').write_text((OUT/'NOTICE.md').read_text('utf-8')+'\n\n## NN主体发布补充\n\n本版切换到已保存NN主干快照，附阶段166额外桨叶辅助及置零回放，辅助收益未稳定，默认关闭。辅助依赖在独立命名空间，避免覆盖主干。新来源指纹见NN_SOURCE_PROVENANCE.json。原RAW不变的旧说明仅适用于上一版几何快照，不适用于本次NN版本更换。\n','utf-8')
    write('NN_SOURCE_PROVENANCE.json',json.dumps(dict(version='0.5.0-research.1',files=provenance,
        scores_source_sha256=sha(STUDY/'gated_scores.json'),new_RAW_measurement=False,new_training=False,
        NN_source_selection='recorded clear NN-led runtime snapshot; historical clip configurations differ',
        additional_auxiliary_default=False,auxiliary_live_integration_proven=False),indent=2))
    write('tools/check_nn_aux.py',(ROOT/'tools/check_nn_rotor166.py').read_text('utf-8')
          .replace('sys.path.insert(0,str(ROOT))',"sys.path.insert(0,str(ROOT/'src'))")
          .replace('from nn_rotor_gate import','from event_drone_forecast.experimental_nn_aux.nn_rotor_gate import')
          .replace('from nn_rotor_residual import','from event_drone_forecast.experimental_nn_aux.nn_rotor_residual import')
          .replace('from adaptive_force_memory import','from event_drone_forecast.experimental_nn_aux.adaptive_force_memory import')
          .replace('from physical_publication import','from event_drone_forecast.experimental_nn_aux.physical_publication import'))
    workflow=OUT/'.github/workflows/synthetic.yml';workflow.write_text(workflow.read_text('utf-8')+'      - run: python tools/check_nn_aux.py\n      - run: python nn_forecast.py demo --case clear --output outputs/nn_demo.jsonl\n      - run: python nn_forecast.py demo --case clear --rotor-assist --output outputs/nn_aux_demo.jsonl\n','utf-8')
    # Ready-to-copy public text for existing nested GitHub layout.
    (DELIVERY/'GitHub简介与发布说明.txt').write_text('''仓库简介：
NN-led event-camera drone tracking and short-horizon forecasting, with experimental rotor geometry/periodicity assistance and documented negative results.

中文简介：
事件相机无人机检测、跟踪与二维短时预测。以NN定位及状态外推为主体，保留五圆／可见桨数／二力模型辅助实验；辅助收益很小且不稳定，本次创新尝试未验证成功。提供可运行回放、基线和负结果。

Release标签：v0.5.0-research.1
标题：NN-led research release with an unsuccessful rotor-assistance study
说明：
发布较强NN主干和默认关闭的额外桨叶辅助。清晰固定6终点27.555→26.889px，侧视17/19终点37.110→37.653px，另一段7终点68.193px不变；严格置零控制未证实周期数值净收益。记录为一次未获得稳定增益的创新尝试，不宣称首创/SOTA或30px/30FPS。NN用于检测定位，后续为跟踪和模型外推；辅助目前为固定输入因果回放，尚未实时串联。原数据、权重、SDK不分发。冻结研究状态。
''','utf-8')
    print(json.dumps(dict(output=str(OUT),auxiliary_modules=len(selected),cases=cases,new_experiments=False),ensure_ascii=False))
