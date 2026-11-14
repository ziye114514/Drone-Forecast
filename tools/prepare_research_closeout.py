"""Build a frozen public research snapshot. No RAW, training or remote publishing."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
BASE = WORK / 'output/github_release_update_20261028/event-drone-forecast'
DELIVERY = WORK / 'output/github_research_closeout_20261105'
OUT = DELIVERY / 'event-drone-forecast'


def write(name, text):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, 'utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    if DELIVERY.exists():
        raise FileExistsError('Use the existing prepared snapshot; do not overwrite.')
    old_manifest = json.loads((BASE / 'SOURCE_MANIFEST.json').read_text('utf-8'))
    assert all(sha(BASE / r['path']) == r['sha256'] for r in old_manifest['files'])
    shutil.copytree(BASE, OUT)
    # The old runnable RAW source stays frozen; new geometry is namespaced.
    shutil.copy2(OUT / 'README.md', OUT / 'docs/LEGACY_RAW_README_20261028.md')
    legacy = OUT / 'docs/LEGACY_RAW_README_20261028.md'
    legacy.write_text(re.sub(r'(?P<pre>!?\[[^\]]*\]\()(?P<url>[^)\s]+)(?P<end>\))',
        lambda m:m.group(0) if '://' in m['url'] or m['url'].startswith('#') else m['pre']+'../'+m['url']+m['end'], legacy.read_text('utf-8')), 'utf-8')
    (OUT / 'SOURCE_MANIFEST.json').unlink()  # replaced by a new, non-self-referential manifest
    package = 'src/event_drone_forecast/rotor_observation/'
    evidence = (ROOT / 'rotor_evidence.py').read_text('utf-8')
    original_import = 'from five_circle_model import diagonal_center,FiveCircleModel,ellipse_tensors'
    assert evidence.count(original_import) == 1
    write(package + 'evidence.py', evidence.replace(original_import, 'from .five_circle_model import diagonal_center,FiveCircleModel,ellipse_tensors'))
    shutil.copy2(ROOT / 'five_circle_model.py', OUT / package / 'five_circle_model.py')
    write(package + '__init__.py', '"""Visibility-conditioned rotor evidence, not metric thrust estimation."""\nfrom .evidence import RotorEvidence, VisibilityLayout\n\n__all__ = ["RotorEvidence", "VisibilityLayout"]\n')
    checks = (ROOT / 'tools/check_rotor_evidence169.py').read_text('utf-8')
    checks = checks.replace('sys.path.insert(0,str(ROOT))', "sys.path.insert(0,str(ROOT/'src'))")
    checks = checks.replace('from rotor_evidence import', 'from event_drone_forecast.rotor_observation import')
    write('tools/check_rotor_observation.py', checks)
    # Keep the base component version; snapshot tag describes the new source assembly.
    write('SNAPSHOT_VERSION.txt', 'research-snapshot-20261105\nRAW base: v0.4.0-alpha.2 (2026-10-28), unmodified runtime\nRotor observation backend: stage169 (2026-11-05), namespaced import adaptation\n')

    # Small numeric observations from the user's own recording; no event/media/label copying.
    rows = []
    source = WORK / 'output/nn_rotor166_20261105/clear_observations.jsonl'
    for line in source.read_text('utf-8').splitlines():
        r = json.loads(line)
        body = {k:r['body'][k] for k in ('detected', 'center') if k in r['body']}
        st = r['structure']
        structure = {k:st[k] for k in ('mode','axis_xy','pose_source','axis_support','observed_model_ids','inferred_model_ids') if k in st}
        rows.append(dict(t=r['t'], body=body, structure=structure,
                         rotors=r.get('geometry',{}).get('rotors',[]), rates=r.get('rates',[]),
                         upstream_NN_ROI_and_body_position_used=True))
    write('examples/rotor_candidates.jsonl', ''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write('examples/rotor_candidates_README.md', '''# 数值候选示例

120个历史开发时刻，来自维护者自采事件录像的已提取桨盘候选，仅保留几何、周期测试、方向及关联所需位置。没有RAW、人工位置标签、第三方数据或模型权重。该数值文件由维护者按根AGPL-3.0-only许可提供；旧视频媒体仍遵循examples/README.md的范围。

上游使用了原生ON/OFF和既有NN辅助ROI/位置；候选不是真值，可能漏检/误检/受历史形状先验约束。演示后端不加载NN，但不能将此示例称为纯桨叶检测成绩。局部来源关联可能使用输入body位置；VisibilityLayout不直接使用当前NN机体中心。

输出可查看four/three/two/side/unresolved及current_center=null等状态；数量只是区域假设，不保证等于物理桨数。没有真实机械RPM、COM、三维姿态或牛顿推力标签。
''')
    write('rotor_demo.py', '''"""SDK/GPU-free replay of extracted candidates; not RAW detection or forecasting."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.rotor_observation import RotorEvidence,VisibilityLayout

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default=str(ROOT/'examples/rotor_candidates.jsonl'))
    p.add_argument('--output',default=str(ROOT/'runs/rotor_demo.jsonl'))
    a=p.parse_args();out=Path(a.output);evidence=RotorEvidence();layout=VisibilityLayout();counts={};n=0
    if out.exists():raise FileExistsError('Choose a new output file: '+str(out))
    out.parent.mkdir(parents=True,exist_ok=True)
    with Path(a.input).open(encoding='utf-8') as inp,out.open('w',encoding='utf-8') as dest:
        for line in inp:
            r=json.loads(line);packet=evidence.step(r['t'],r.get('body',{}),r.get('structure',{}),r.get('rotors',[]),r.get('rates',[]))
            state=layout.step(packet);counts[state['model']]=counts.get(state['model'],0)+1;n+=1
            dest.write(json.dumps(dict(evidence=packet,layout=state),allow_nan=False)+'\\n')
    print(json.dumps(dict(frames=n,models=counts,output=str(out),RAW_read=False,NN_loaded=False,
                         input_is_extracted_candidates=True,accuracy_or_FPS_benchmark=False),ensure_ascii=False))

if __name__=='__main__':main()
''')
    # Include compact evidence without local paths or giant stage history.
    layout = json.loads((WORK / 'output/decision169_20261105/layout_summary.json').read_text('utf-8'))
    score = json.loads((WORK / 'output/decision169_20261105/decision_scores.json').read_text('utf-8'))
    results = dict(date='2026-11-05', geometry_test='controlled removal of one current detected center, not pixel occlusion',
        independent_reference=False, reference='same-frame complete detected layout, not physical COM',
        video_frames=16, related_cases=64, upstream_NN_ROI=True,
        visible_centroid=layout['baseline_common'], history_only=layout['history_only_common'],
        shared_translation=layout['shared_translation_common'], old_affine=layout['model_common'],
        force_diagnostic={name:dict(baseline=v['uncalibrated_baseline'], **{a:b['error'] for a,b in v['arms'].items()}) for name,v in score.items() if name!='decision'},
        force_diagnostic_passed=False, manual_calibration_used=True, historical_development_data=True,
        universal_performance_upper_bound=False, further_parameter_search_in_this_batch=False,
        geometry_is_trajectory_FDE=False, stable_RAW_30Hz_or_30px_achieved=False)
    write('examples/research_summary_20261105.json', json.dumps(results, ensure_ascii=False, indent=2))
    shutil.copy2(WORK / 'output/decision169_20261105/geometry_value.png', OUT / 'examples/geometry_value.png')

    write('README.md', '''# Event Drone Forecast

**事件相机桨叶观测、可见性几何恢复与二维短时预测研究原型。**

本项目探索：能否利用原生ON/OFF中的高速桨叶信号，在不同可见数量下建立简化无人机结构，并辅助预测运动。当前提供可运行的观测后端、既有RAW流程、演示与负结果。**这是冻结的研究快照，不是已验证的高精度实时预测产品，也不是所引用论文的官方实现。**

## 本快照包含什么

| 部分 | 功能与范围 |
|---|---|
| 桨叶证据后端（2026-11-05） | 当前弧/椭圆与周期测试→可见区域、合格周期、无序可见贡献；实测与推断分开 |
| 五圆／可见性布局 | 四点布局中心；三点借近期完整布局恢复；两点/侧线保留方向与有时效先验，不虚构完整中心 |
| SDK/GPU-free示例 | 用随附数值候选复现后端输出，无需相机、模型权重或GPU；它不直接处理RAW |
| 冻结RAW流程（2026-10-28） | 保留已整理的旧版检测/跟踪/短时外推代码及控制；依赖独立SDK、兼容模型和硬件 |
| 实验记录 | 几何一致性小试、周期/推力辅助预测未获得稳定增益的结果和限制 |

RAW基线与新增后端分别保留。**本快照没有把阶段169几何后端自动替换进旧RAW实时预测器，也不是整个本机实验工作区的全量导出。**

## 先运行一个小示例

Python 3.11+，在仓库源码目录执行：

```bash
python -m pip install "numpy>=2,<2.5" "scipy>=1.14,<2" "opencv-python-headless>=4.10,<6"
python rotor_demo.py --output runs/rotor_demo.jsonl
python tools/check_rotor_observation.py
```

输出JSONL含`evidence`和`layout`：当前可见区域、周期资格、推断点、当前中心、先验时效及拒绝原因。两点/侧线中`current_center=null`是明确的不可观测状态，不应用历史位置冒充当前检测。

示例候选上游用了NN辅助ROI/位置，候选不是真值；本示例不加载NN，不等于已证明纯桨叶检出率。详见[候选来源](examples/rotor_candidates_README.md)。

模块API位于`src/event_drone_forecast/rotor_observation`；从源码安装`python -m pip install -e .`后：

```python
from event_drone_forecast.rotor_observation import RotorEvidence, VisibilityLayout

evidence, layout = RotorEvidence(), VisibilityLayout()
# 输入为已有前端提取的当前候选，t严格递增；不是直接传入视频/RAW。
packet = evidence.step(t, body, structure, rotors, rates)
state = layout.step(packet)
```

原RAW流程另见[旧版使用说明](docs/LEGACY_RAW_README_20261028.md)与[快速开始](docs/GETTING_STARTED.md)。SDK、检测权重和原始数据不随包提供，模型获取/训练流程尚不完整；不承诺陌生机器clone后一键跑通RAW。

## 核心处理思路

```text
RAW原生ON/OFF → 周期/密度支持与弧/椭圆候选 → 当前证据资格
                                                     ↓
四区域：完整布局候选 / 三区域：近期布局补点 / 两区域或侧线：方向与部分证据
                                                     ↓
当前实测、历史推断、不可观测状态分别输出
                                                     ↓
研究分支：当前可测贡献 + 不可测合力 → 二维短时预测与残差反馈
```

“五圆”是四个桨盘加一个几何机体中心的简化表示，不要求机头/机尾。四路合格周期平方和不要求固定桨编号；少于四路时只记实测贡献，不补隐藏RPM。隐藏桨力、重力、风/阻力及模型误差归不可测项。这是建模分账，**不等于已从图像辨识真实力**。

## 结果应怎样理解

在16个真实视频候选帧中轮流删掉一个中心，共64个相关的受控缺测案例。相对同帧完整检测布局，中心一致性误差如下：

| 方法 | 均值px |
|---|---:|
| 可见三点直接平均 |21.258|
| 仅过去布局速度外推 |0.837|
| 近期布局＋共享平移 |0.325|
| 既有仿射恢复 |0.348|

![受控几何缺测示例](examples/geometry_value.png)

**这不是0.4秒轨迹预测误差、真实质心误差或真实遮挡检出精度。** 原图未做像素遮挡；候选可能有误且部分拟合受历史先验约束；参考不独立，上游ROI用了NN。结果仅说明布局约束可减少可见点组成变化带来的中心偏移，不证明算法首次性或整套系统优于已有方法。

周期辅助预测的一次固定标定—后段检验未稳定胜过置零/错时序控制。例如纯桨叶侧视后段7点，未标定状态外推39.50px、对齐响应头42.38px、置零42.38px、过去错时序50.58px。使用过标定段人工终点，不能叫无真值自适应。负结果见[完整简表](examples/research_summary_20261105.json)。

**目前未实现：** 稳定真实机械RPM/牛顿推力、质量/风的可靠辨识、独立验证的推力预测增益、跨录像稳定0.4秒30px误差、持续有效RAW30Hz。视频播放帧率、缓存处理FPS、实时新观测频率不能混用。旧RAW示例和指标属于旧版，不能作为新增后端的预测成绩。

## 贡献、来源与许可

项目贡献是原生事件处理流程的适配、可见性条件化几何观测接口、实测/推断/不可观测状态区分，以及可检查的实验和失败边界。刚体模板、椭圆拟合、滤波、频率平方推力关系本身不是本项目首次提出；新组合的学术新颖性尚未确认。

本项目使用AI编码工具辅助实现和文档，研究问题、数据与结果取舍由维护者负责。第三方代码不作为自己的原创贡献。

集成发布沿用**AGPL-3.0-only**；保留[LICENSE](LICENSE)、[原MIT声明](LICENSES/MIT-original.txt)、[NOTICE](NOTICE.md)与[第三方说明](THIRD_PARTY_NOTICES.md)。SDK、Ultralytics、模型及数据仍受各自条款约束；不附SDK/模型权重、FRED原始数据、论文PDF或无明确授权的研究仓库。媒体范围另见[说明](examples/README.md)。

## 项目状态

**研究快照已冻结，维护者暂不继续主动优化；无固定更新或支持承诺。** 保留源码供学习、复查与后续研究，不以继续试验作为发布前提。

[发布状态与边界](docs/RESEARCH_SNAPSHOT.md) · [版本组成](SNAPSHOT_VERSION.txt) · [来源指纹](SOURCE_PROVENANCE.json) · [AI与来源说明](NOTICE.md)
''')
    write('docs/RESEARCH_SNAPSHOT.md', '''# 冻结研究快照：2026-11-05

## 发布定位

这是事件相机桨叶观测、可见性几何恢复与短时预测的研究源码。没有确认全新算法或前沿精度优势；项目完成度与论文创新性分别描述。维护者停止本轮主动优化，保留成果并公开边界，不承诺后续开发。

发布组成：经过整理的2026-10-28 v0.4.0-alpha.2 RAW基线保持算法不变；增加2026-11-05阶段169后端到独立命名空间。包元数据0.4.0a2标识原RAW/位置组件；整体快照标签为research-snapshot-20261105，不将不同版本实验混成一个成绩。新增几何未自动接入旧RAW链。

## 可以用于项目介绍的文字

“开发事件相机四旋翼桨叶观测与二维短时预测研究原型，完成原生ON/OFF事件处理流程适配、五圆简化结构和四/三/二桨及侧视条件化几何接口，区分实测、历史补点与不可观测状态；提供可运行数值示例、基线对照和实验局限。受控缺测测试显示几何中心一致性改善，推力辅助预测尚无稳定增益。”

可以写实现/工程贡献，不要写首创、可靠推力恢复、持续30FPS/30px、SOTA或已解决极端侧飞。

## 已知限制

候选不是物理电机真值；侧视投影会压扁和合并。当前频率可能是谐波或混合代理，无独立机械转速真值。图像空间可见响应与任意未知合力之间存在可辨识性问题，隐藏桨力也可能快变。小样本旧开发片段、相关删点案例和受先验约束的候选不能证明泛化。

新示例只运行已提取候选后端；不读取RAW，不训练NN，不证明检测召回/FPS或未来轨迹精度。旧完整RAW链需用户独立安装SDK、提供兼容模型并按旧文档配置；本包未提供完整一键模型获取/训练链。

## 运行核验范围

本地交付验证记录放在交付目录外层verification，不把本机环境成功当全新机器部署证明。只进行源码/轻量示例检查，不再启动RAW/GPU优化。根README是本快照的说明入口，旧文档中的版本、目标和结果为历史。

## 许可依据与发布范围

沿用此前集成版AGPL-3.0-only与所有原版权/许可，不改变本机研究目录的MIT文本或第三方权利。可参考[GitHub许可说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)及[Ultralytics许可](https://www.ultralytics.com/license)，后者是原RAW检测依赖之一。这里只说明此包的已有选择，不将SDK、数据或权重重新授权。
''')
    with (OUT / 'NOTICE.md').open('a', encoding='utf-8') as f:
        f.write('\n\n## 2026-11-05研究快照补充\n\n新增event_drone_forecast.rotor_observation来自本项目阶段169自有研究源码；仅将模块导入改成相对命名空间，保留其几何/资格计算。示例是维护者自采录像已提取的数值候选，来源和非真值边界见examples/rotor_candidates_README.md。RAW运行文件保持2026-10-28集成版不变。新增代码随该集成快照AGPL-3.0-only发布，原MIT与第三方许可保留。未打包阶段169全部实验拟合工具或整个本机工作区。\n')
    workflow = OUT / '.github/workflows/synthetic.yml'
    with workflow.open('a', encoding='utf-8') as f:
        f.write('      - run: python tools/check_rotor_observation.py\n      - run: python rotor_demo.py --output runs/rotor_demo.jsonl\n')
    provenance = json.loads((OUT / 'SOURCE_PROVENANCE.json').read_text('utf-8'))
    provenance['research_snapshot_20261105'] = dict(
        frozen_base='v0.4.0-alpha.2 source manifest verified before copy',
        geometry_files=[dict(destination=package+'evidence.py', original='rotor_evidence.py', original_sha256=sha(ROOT/'rotor_evidence.py'), adaptation='one relative import only'),
                        dict(destination=package+'five_circle_model.py', original='five_circle_model.py', original_sha256=sha(ROOT/'five_circle_model.py'), adaptation='verbatim')],
        numeric_candidates_source_sha256=sha(source), numeric_candidates_filtered_fields=True,
        raw_runtime_unchanged=True, force_prediction_improvement_claimed=False)
    write('SOURCE_PROVENANCE.json', json.dumps(provenance, ensure_ascii=False, indent=2))
    # Root README for the user's existing nested repository layout.
    readme = (OUT / 'README.md').read_text('utf-8')
    nested = re.sub(r'(?P<pre>!?\[[^\]]*\]\()(?P<url>[^)\s]+)(?P<end>\))',
        lambda m:m.group(0) if '://' in m['url'] or m['url'].startswith('#') else m['pre']+'event-drone-forecast/'+m['url']+m['end'], readme)
    (DELIVERY / 'ROOT_README_FOR_NESTED.md').write_text(nested, 'utf-8')
    nested_dir = DELIVERY / 'root_files_for_nested_layout'
    nested_dir.mkdir()
    (nested_dir / 'README.md').write_text(nested, 'utf-8')
    shutil.copy2(OUT / 'LICENSE', nested_dir / 'LICENSE')
    (nested_dir / 'NOTICE.md').write_text('研究源码位于event-drone-forecast。完整版权、原MIT与第三方范围见[event-drone-forecast/NOTICE.md](event-drone-forecast/NOTICE.md)和[第三方说明](event-drone-forecast/THIRD_PARTY_NOTICES.md)。根AGPL许可与源码内相同，README不代替LICENSE。\n','utf-8')
    (DELIVERY / 'GitHub项目简介与发布文字.txt').write_text('''建议仓库简介：
Event-camera rotor observations, visibility-aware quadrotor geometry, and experimental 2D short-horizon forecasting. Frozen research prototype.

建议中文介绍：
事件相机桨叶观测与二维短时预测研究原型。实现原生ON/OFF处理流程、五圆简化模型及不同可见桨数下的几何接口，保留基线、数值示例和负结果。受控缺测中观察到布局中心一致性改善；真实推力估计、稳定预测增益和30px/30FPS尚未证实。

建议Release标签：research-snapshot-20261105
建议Release标题：Frozen research snapshot: rotor observations and visibility-aware geometry
发布说明：
公开冻结的RAW研究基线及新增桨叶证据/布局后端。提供无需SDK/GPU的候选回放示例，明确分开实测、推断与不可观测状态；保留推力响应诊断负结果。旧RAW运行代码与新几何模块独立保留，新模块未自动接入旧RAW实时预测。项目停止本轮主动优化，无持续更新承诺。

简历可用（不要加首创或达标字样）：
开发事件相机四旋翼观测与短时预测研究原型，完成ON/OFF流程适配、可见性条件化几何后端、数值回放及基线对照；分析侧视退化和力学参数不可辨识等局限，公开代码与实验边界。

发布应注明：功能状态；测试数据/样本数及开发集口径；几何一致性不等于轨迹精度；无稳定推力预测收益；硬件/SDK/模型依赖；上游代码和许可证；AI辅助实现；数据/权重排除范围；冻结维护状态。
''','utf-8')
    manifest = dict(snapshot='research-snapshot-20261105', files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()])
    write('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2))
    # Two batches for GitHub's current 100-files web-upload limit; keep nesting.
    files = [p for p in sorted(OUT.rglob('*')) if p.is_file()]
    batches=[]
    for i in range(0,len(files),90):
        name=f'网页上传第{i//90+1}批'
        dest=DELIVERY/name/'event-drone-forecast'
        for p in files[i:i+90]:
            q=dest/p.relative_to(OUT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        batches.append(dict(folder=name,files=len(files[i:i+90])))
    (DELIVERY/'发布操作说明.txt').write_text('''本目录是本地发布准备，没有自动上传GitHub。

完整标准仓库源码：event-drone-forecast。用Git/GitHub Desktop时，将其中的内容放在仓库根目录。
ZIP用于GitHub Releases附件；仅把ZIP作为仓库文件上传不会让GitHub自动解压或显示内部README。

你之前仓库是外部README + event-drone-forecast子目录布局：
1. 不必先删整个旧仓库。建议新建一个分支保存这次快照，保留旧历史。
2. 浏览器一次最多100文件，按网页上传第1批、第2批中同名event-drone-forecast文件夹逐批上传到仓库根，更新相同路径。
3. 将root_files_for_nested_layout里的README.md、LICENSE、NOTICE.md放在仓库根。外部README源代码另存为ROOT_README_FOR_NESTED.md，可直接复制全文替换。
4. 确认event-drone-forecast/src/event_drone_forecast/rotor_observation等文件已出现；旧仓库可能保留本包未包含的旧文件，它们不属于本次快照。ZIP和SOURCE_MANIFEST定义本次交付范围。
5. 自愿发布Release，标签research-snapshot-20261105；简介和发布文字已写好。不承诺达到30px/30FPS，不以继续试验为发布条件。

GitHub当前说明：https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
许可与来源以包内LICENSE/NOTICE/THIRD_PARTY_NOTICES为准；不上传本机RAW、模型、SDK、虚拟环境或整个output。
''','utf-8')
    archive=DELIVERY/'event-drone-forecast-research-snapshot-20261105.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'event-drone-forecast/'+p.relative_to(OUT).as_posix())
    (DELIVERY/(archive.name+'.sha256')).write_text(sha(archive)+'  '+archive.name+'\n','utf-8')
    (DELIVERY/'delivery.json').write_text(json.dumps(dict(source=str(OUT),zip=str(archive),files=len(files),batches=batches,
        archive_sha256=sha(archive),new_RAW_experiment=False,remote_upload=False,smoke_verification_pending=True),ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(folder=str(OUT),files=len(files),zip_MB=round(archive.stat().st_size/1048576,2),batches=batches),ensure_ascii=False))
