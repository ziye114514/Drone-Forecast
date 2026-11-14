"""Curated public source export; never publishes, trains, or touches protected projects."""
from pathlib import Path
import argparse,ast,hashlib,json,re,shutil,zipfile,csv
ROOT=Path(__file__).resolve().parents[1];TEMPLATES=ROOT/'release_templates/github'
VERSION='0.4.0-alpha.2'
COPY=('normal_turn_forecast.py','temporal_period_support.py','rotor_thrust_bank.py','propeller_physics.py','curved_flow_motion.py','flow_anchor_gauge.py','adaptive_force_memory.py','thrust_trend_forecast.py','aligned_period_recovery.py','body_period_stream.py','endpoint_force_balance.py','fast_raw_reader.py','period_process.py','sdk_activity_backend.py','native/sdk_activity_avx2.cpp','forecast_runtime.py','fast_structure_front.py','fast_geometry_compute.py','native_counts_numba.py','native_geometry_numba.py','native_latest_frequency.py','requirements-counts.txt','shared_image_motion.py','motion_observability.py','live_flow_spatial.py','flow_force_balance.py','projected_unknown_force.py','axis_amplitude_bridge.py','live_local_flow.py','live_flow_acceleration.py','native_render.py','responsive_force.py','continuous_five_circle.py',
    'five_circle_model.py','shared_arc_recovery.py','partial_arc_information.py','fast_force_response.py','visibility_period.py',
    'shared_phase_fft.py','live_decode.py','live_pixel_axis.py','live_frequency_pixels.py','live_scale_rollout.py',
    'live_motion.py','live_measurement_kf.py','live_covariance.py','live_worker.py','body_association.py',
    'live_roi_refine.py','live_center_ensemble.py','forecast_output.py','position_history_gauge.py','nn_search_schedule.py','run_fast_structure.py')
TESTS=('check_five_circle_v3.py','check_three_continuation.py','check_shared_phase_fft.py',
    'check_force_gap_hold.py','check_shared_arc_recovery.py','check_partial_arc_information.py',
    'check_anchor_and_amplitude.py','build_sdk_activity.py','check_disk_prediction.py','check_rotor_force_model.py','check_motion_response.py','check_position_history_gauge.py')


COPY += ('rigid_center_translation.py','physical_publication.py','live_publication.py','body_core_motion.py','fast_huber.py','native_huber.py','native_conic_residual.py','nn_recovery_budget.py','position_net_force.py','native_body_workspace.py','component_flow_acceleration.py')
COPY += ('singer_position_state.py',)
COPY += ('phase_rate_bridge.py',)
COPY += ('rotor_nvrtc.py','native/rotor_harmonics.cu','local_pixel_phase.py')
COPY += ('native_axis_workspace.py','native_plane_motion.py','native_correlation.py','native_histogram_front.py','native_history_buffer.py','relative_rotor_observer.py')
COPY += ('current_box_gate.py',)
COPY += ('rotor_only_front.py','rotor_support_front.py','rotor_period_worker.py','run_rotor_only.py',
         'predictive_harmonics.py','predictive_harmonics_cuda.py','rotor_force_observer.py',
         'rotor_cuda_device.py','rotor_cuda_worker.py')

def digest(data):return hashlib.sha256(data).hexdigest()
def write(root,name,data):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(data if isinstance(data,bytes) else data.encode('utf-8'));return p
def extract(source,names,imports):
    text=(ROOT/source).read_text('utf-8');tree=ast.parse(text);parts=[];selected=[]
    for n in tree.body:
        name=getattr(n,'name',None)
        if isinstance(n,ast.Assign):name=next((t.id for t in n.targets if isinstance(t,ast.Name)),None)
        if name in names:
            parts.append(ast.get_source_segment(text,n));selected.append(dict(name=name,ast_sha256=digest(ast.dump(n,include_attributes=False).encode())))
    if set(names)!={q['name'] for q in selected}:raise ValueError((source,names,selected))
    result='"""Current runtime definitions extracted without arithmetic changes; see SOURCE_PROVENANCE.json."""\n'+imports+'\n\n'+'\n\n'.join(parts)+'\n'
    ast.parse(result);return result,selected


def build(out):
    out=out.resolve()
    if out.exists():raise FileExistsError('Use a new export directory: '+str(out))
    archive=out.parent/f'event-drone-forecast-v{VERSION}.zip'
    if archive.exists():raise FileExistsError(archive)
    out.mkdir(parents=True);provenance=[]
    for f in COPY:
        b=(ROOT/f).read_bytes();write(out,f,b)
        provenance.append(dict(destination=f,original=f,original_sha256=digest(b),mode='copy_or_path_interface_adaptation'))
    package=ROOT/'src/event_drone_forecast'
    selected=[package/f for f in ['__init__.py','geometry.py','detector_backend.py','raw_io.py']]
    selected+=list((package/'position').glob('*.py'))
    selected += [p for p in (package/'_vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    for p in selected:
        name=p.relative_to(ROOT).as_posix();write(out,name,p.read_bytes())
        provenance.append(dict(destination=name,original=name,original_sha256=digest(p.read_bytes()),mode='verbatim'))
    exports=[('rotor_geometry.py','research/stage146/branches146.py',
        ['PARAMS','centres','unit','metric','ellipse_tensor','three_affine_hypotheses','two_axis','plane_axis','BranchGeometry','rotor_rates'],
        'from itertools import permutations\nimport numpy as np\nimport cv2'),
        ('curvature_split.py','research/stage147/curvature147.py',
        ['PARAMS','notches','fit_arc','pair_evidence','split_contour'],
        'from itertools import combinations\nimport numpy as np\nimport cv2\nfrom scipy.ndimage import gaussian_filter1d\nfrom event_drone_forecast.geometry import _fit,_distance,_coverage,_uncertainty'),
        ('arc_parameters.py','research/stage148/guided148.py',['PARAMS'],''),
        ('body_axis.py','research/stage6/silhouette.py',['PARAMS','angle_distance','principal','body_measurement'],'import cv2,numpy as np'),
        ('frequency_features.py','research/stage79/native79.py',['PARAMS','engine','copy_frequency','checked_process','summarize'],
        'import numpy as np\nimport metavision_sdk_cv as sdk'),
        ('native_axis.py','research/stage141/native_axis141.py',['PARAMS','NativeAxis','measure_current'],
        'import numpy as np\nimport frequency_features as native79\nfrom event_drone_forecast.geometry import _line'),
        ('live_structure_bridge.py','live_structure_bridge.py',['PARAMS','QualifiedCenter','NoiseKF','GeometryMotion'],
        'import copy,numpy as np\nfrom live_scale_rollout import ScaleRolloutMotion,scale_predictions\nfrom live_measurement_kf import MeasurementKF'),
        ('lumped_disturbance.py','lumped_disturbance.py',['PARAMS','LumpedDisturbance','forecasts'],
        'import numpy as np\nfrom event_drone_forecast.position._filterpy import predict,update')]
    for dest,source,names,imports in exports:
        text,defs=extract(source,names,imports);write(out,dest,text)
        provenance.append(dict(destination=dest,original=source,original_sha256=digest((ROOT/source).read_bytes()),
            mode='AST_verified_definition_extraction',definitions=defs))
    for f in TESTS:write(out,'tools/'+f,(ROOT/'tools'/f).read_bytes())
    for p in TEMPLATES.rglob('*'):
        if p.is_file() and p.name!='LICENSE_AGPL.txt':write(out,p.relative_to(TEMPLATES).as_posix(),p.read_bytes())
    license_text=(TEMPLATES/'LICENSE_AGPL.txt').read_bytes()
    assert b'GNU AFFERO GENERAL PUBLIC LICENSE' in license_text and b'END OF TERMS AND CONDITIONS' in license_text
    write(out,'LICENSE',license_text);write(out,'LICENSES/MIT-original.txt',(ROOT/'LICENSE').read_bytes())
    # Export-only import/path adaptations. Working research sources are not edited.
    def change(name,old,new):
        p=out/name;s=p.read_text('utf-8')
        if old not in s:raise ValueError('Expected source fragment absent: '+name+' '+old[:70])
        p.write_text(s.replace(old,new),'utf-8')
    for f in ['fast_structure_front.py','fast_geometry_compute.py','shared_arc_recovery.py','live_pixel_axis.py','run_fast_structure.py','aligned_period_recovery.py','body_period_stream.py','rotor_only_front.py','run_rotor_only.py']:
        p=out/f;s=p.read_text('utf-8')
        s=re.sub(r'sys\.path\[:0\]\s*=\s*\[.*?\]',"sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src')]",s,flags=re.S)
        s=s.replace("sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage141'))","sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))")
        s=s.replace('from research.stage146.branches146 import','from rotor_geometry import')
        s=s.replace('from research.stage147.curvature147 import','from curvature_split import')
        s=s.replace('from research.stage6.silhouette import','from body_axis import')
        s=s.replace('research/stage6/silhouette.py','body_axis.py')
        for old,new in [('research/stage146/branches146.py','rotor_geometry.py'),
                ('research/stage147/curvature147.py','curvature_split.py'),
                ('research/stage148/guided148.py','arc_parameters.py'),
                ('research/stage79/native79.py','frequency_features.py')]:s=s.replace(old,new)
        s=s.replace('from guided148 import','from arc_parameters import').replace('from curvature147 import','from curvature_split import')
        s=s.replace('from native_axis141 import','from native_axis import').replace('import native79','import frequency_features as native79')
        s=s.replace('from native79 import','from frequency_features import')
        s=s.replace("sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage79'))","sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))")
        p.write_text(s,'utf-8')
    change('forecast_runtime.py',"REFERENCE_SHA256 = 'e2531556ad0431d904fb0d1d432727a2e2dd9eaed9e098a6ed3d9e55f474db0b'", "REFERENCE_SHA256 = 'ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4'")
    change('forecast_runtime.py',"cfg.get(key, default)","(cfg.get(key) if cfg.get(key) is not None else default)")
    p=out/'run_fast_structure.py';s=p.read_text('utf-8');tree=ast.parse(s)
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='resolve')
    s=s.replace(ast.get_source_segment(s,call),'resolve(args.gpu_python,args.detector_weights,args.detector_sha256,args.cache_dir)')
    tree=ast.parse(s)
    protected=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='protected')
    s=s.replace(ast.get_source_segment(s,protected),'# Private workspace guards are not part of the public source export.')
    s=s.replace("p.add_argument('--gpu-python');", "p.add_argument('--gpu-python');p.add_argument('--detector-weights');p.add_argument('--detector-sha256');p.add_argument('--cache-dir');")
    tree=ast.parse(s)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='f'
        and isinstance(n.iter,ast.List) and any(isinstance(v,ast.Constant) and v.value=='run_fast_structure.py' for v in n.iter.elts))
    core=sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p.suffix in ('.py','.cpp') and not any(q in p.relative_to(out).parts for q in ['tests','tools']))
    s=s.replace(ast.get_source_segment(s,loop.iter),repr(core));p.write_text(s,'utf-8')
    change('src/event_drone_forecast/__init__.py',"'0.3.1'","'0.4.0a2'")
    write(out,'configs/runtime.example.json',json.dumps(dict(sdk_python=None,gpu_python=None,
        detector_weights='../weights/best.onnx',detector_sha256='ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4',cache_dir='../.cache'),indent=2))
    write(out,'configs/dataset.example.yaml','path: ../data/yolo\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n')
    write(out,'requirements-gpu.txt','# Install a CUDA-compatible torch/torchvision build first. SDK is separate.\nultralytics==8.4.110\nonnxruntime-gpu==1.30.0\nonnx==1.23.1\n')
    write(out,'pyproject.toml','''[build-system]
requires = ["setuptools>=65"]
build-backend = "setuptools.build_meta"
[project]
name = "event-drone-forecast"
version = "0.4.0a2"
description = "Five-circle event-camera drone modeling and image-plane forecast research prototype"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "AGPL-3.0-only"}
dependencies = ["numpy>=2,<2.5", "scipy>=1.14,<2", "opencv-python-headless>=4.10,<6", "pillow>=10"]
[project.optional-dependencies]
report = ["matplotlib>=3.9"]
[tool.setuptools]
license-files = ["LICENSE", "NOTICE.md", "LICENSES/*"]
[tool.setuptools.packages.find]
where = ["src"]
include = ["event_drone_forecast*"]
[tool.setuptools.package-data]
event_drone_forecast = ["_vendor/filterpy_linear_upstream/*", "_vendor/filterpy_process_upstream/*"]
''')
    write(out,'.gitignore','__pycache__/\n*.py[cod]\n*.egg-info/\n.venv*/\nbuild/\ndist/\n.cache/\noutputs/\ndata/\nweights/\nconfigs/local.json\nconfigs/*.local.yaml\n*.raw\n*.pt\n*.onnx\n*.dll\n*.pyd\n*.exe\n.env\n')
    write(out,'.gitattributes','* text=auto\n*.py text eol=lf\n*.md text eol=lf\n*.webm binary\n*.gif binary\n*.png binary\n')
    write(out,'CITATION.cff','''cff-version: 1.2.0
message: "Please cite this software version and the upstream methods you actually use."
type: software
title: "Event Drone Forecast: Five-Circle Event-Based Drone Modeling and Forecasting"
authors:
  - name: "Event Drone Forecast contributors"
version: "0.4.0-alpha.2"
date-released: 2026-10-28
license: AGPL-3.0-only
''')
    write(out,'CITATION.bib','''@software{event_drone_forecast_2026,
  author = {{Event Drone Forecast contributors}},
  title = {Event Drone Forecast: Five-Circle Event-Based Drone Modeling and Forecasting},
  version = {0.4.0-alpha.2},
  year = {2026},
  license = {AGPL-3.0-only}
}
''')
    write(out,'.github/workflows/synthetic.yml','''name: Synthetic controls
on: [push, pull_request]
permissions:
  contents: read
jobs:
  controls:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
      - run: python -m pip install -e .
      - run: python forecast.py doctor --base-only
      - run: python -m unittest discover -s tests -v
''')
    write(out,'CONTRIBUTING.md','# 贡献\n\n请提供新输出目录中的最小复现、环境/模型契约、因果时刻和同输入对照。不要上传无权分发的RAW、权重、SDK或个人日志。新增优化保留旧控制，区分合成机制、观测代理和独立真值；不以播放FPS代替运行时间。\n')
    write(out,'CHANGELOG.md',(TEMPLATES/'RELEASE_NOTES.md').read_text('utf-8')+'\n\n## 0.4.0-alpha.1 — 2026-10-27\n\n首次公开整理五圆主线、模型配置、自检与合成演示。旧版本源码包保持不变。\n')
    evidence=ROOT.parent/'output/three_continuation_20261026'
    for name in ('clear','side'):
        folder=evidence/f'raw_release_{name}'
        write(out,f'examples/{name}.webm',(folder/'preview.webm').read_bytes())
        write(out,f'examples/{name}_poster.jpg',(folder/'poster.jpg').read_bytes())
    write(out,'examples/forecast_comparison.png',(evidence/'forecast_comparison.png').read_bytes())
    write(out,'examples/phase_tracks.png',(ROOT.parent/'output/shared_angular_phase_20261027/phase_tracks.png').read_bytes())
    metrics=json.loads((evidence/'summary.json').read_text('utf-8'))
    clean=dict(version=VERSION,reference_model_sha256='ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4',
        scope='Previously exposed self-recorded development clips; sparse bbox references; no independent RPM/force truth',
        current_side_13target=metrics['current_side_accuracy']['scores'],rate_definitions=metrics['rate_definitions'],
        stable30_proven=False,physics_advantage_proven=False)
    clean['reference_model_sha256']='ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4'
    write(out,'examples/metrics.json',json.dumps(clean,ensure_ascii=False,indent=2))
    # Historical evidence stays separate from the newly measured candidate.
    write(out,'examples/historical_alpha1_metrics.json',json.dumps(dict(clean,version='0.4.0-alpha.1'),ensure_ascii=False,indent=2))
    latest=ROOT.parent/'output/shared_template_budget_20261027'
    ext=json.loads((latest/'side_visibility_final8s/extension_summary.json').read_text('utf-8'))
    def summary(name):return json.loads((latest/name/'summary.json').read_text('utf-8'))
    side=summary('side_visibility_final8s');clear=summary('clear_fused_line_goal');other=summary('train2_visibility_final')
    scores=json.loads((latest/'train2_visibility_final/shared_motion_accuracy.json').read_text('utf-8'))
    # Whitelist aggregates only: never copy protocols, private paths or per-event data.
    public_keys=('frames','fresh_image_observations','valid_forecasts','current_force_inputs','recent_amplitude_current_axis_inputs',
        'steady_capacity_fps','paced_fresh_observation_rate','first_result_after_s','paced_deadline_fraction')
    current=dict(version=VERSION,reference_model_sha256=clean['reference_model_sha256'],
        scope=clean['scope'],independent_test=False,stable30_proven=False,goal_complete=False,
        candidate_flags='--target-candidate --sdk-avx2 --rigid-motion --tangent-disturbance --structure-guided-nn --paced',
        side_summary={k:side[k] for k in public_keys if k in side},
        side_scores=ext['full_scores'],side_fixed_and_extension=ext['scopes'],
        clear_summary={k:clear[k] for k in public_keys if k in clear},clear_accuracy='not independently scored',
        train2_summary={k:other[k] for k in public_keys if k in other},train2_scores=scores,
        warning='Capacity includes empty frames; compare the fixed target-present window, current observation Hz and latency separately. Labels score only; not read by runtime.')
    # score summaries can contain local input/run paths; retain only numeric score trees.
    def numeric_tree(x):
        if isinstance(x,dict):return {k:v for k,a in x.items() if (v:=numeric_tree(a)) is not None}
        if isinstance(x,list):return [v for a in x if (v:=numeric_tree(a)) is not None]
        return x if isinstance(x,(int,float,bool)) or x is None else None
    current['train2_scores']=numeric_tree(scores)
    current['side_fixed_and_extension']=numeric_tree(ext['scopes'])
    write(out,'examples/metrics.json',json.dumps(current,ensure_ascii=False,indent=2))
    write(out,'examples/latest_prediction_and_latency.png',(latest/'side_visibility_final8s/prediction_and_latency.png').read_bytes())
    write(out,'examples/README.md','# 演示媒体\n\nclear/side.webm来自自采train6约26–28秒/110–113秒已保存因果输出，30FPS只是播放时间。红点为当前中心、灰色为模型推断，黄色为条件物理预测。metrics.json为同起点历史开发结果，不是公开盲测。\n\n只包含短可视化，不含原RAW或真实RPM/推力标注。示例媒体版权由项目原作者保留，仅用于本仓库演示；根代码许可证不自动向第三方授予媒体再分发/其他用途权利。phase_tracks.png表示同源事件估计，不是真实转速误差。\n')
    write(out,'examples/index.html','<!doctype html><meta charset="utf-8"><title>Event Drone Forecast demos</title><body style="max-width:1000px;margin:30px auto;font-family:system-ui"><h1>Saved causal ON/OFF outputs</h1><p>Playback FPS is not measured processing FPS. No mechanical RPM or Newton-force truth.</p><video controls style="width:100%" src="clear.webm" poster="clear_poster.jpg"></video><video controls style="width:100%" src="side.webm" poster="side_poster.jpg"></video></body>')
    # A small readable GIF derived solely from the saved self-recorded output.
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(str(evidence/'raw_release_clear/results.mp4'));frames=[];i=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if i%3==0:
            crop=frame[:500,700:1200];frames.append(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((400,400)))
        i+=1
    cap.release();assert frames
    frames[0].save(out/'examples/clear.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
    write(out,'SOURCE_PROVENANCE.json',json.dumps(dict(version=VERSION,scope='Curated current source; no downloaded author repository',
        license_text_source='GNU AGPLv3 full text from installed ultralytics8.4.110 distribution licenses/LICENSE',
        license_text_sha256=digest(license_text),
        original_own_code_notice='LICENSES/MIT-original.txt',integrated_license='AGPL-3.0-only',
        third_party_code='FilterPy MIT snapshots only',files=provenance),indent=2))
    write(out,'docs/ENVIRONMENT.md','# 本机验证环境\n\n基础/SDK侧：Windows，Python3.11，NumPy2.4.6，SciPy1.17.1，OpenCV headless5.0.0.93，Pillow12.3.0。GPU侧：Python3.13，Torch2.13.0+cu130，Torchvision0.28.0+cu130，Ultralytics8.4.110，ONNX Runtime GPU1.30.0，ONNX1.23.1。\n\n这是已用本机环境记录，不是干净机器或所有CUDA/SDK组合的安装保证。合成CI不需要GPU/SDK；GitHub Actions配置不等于已经在GitHub运行通过。真实模型/数据分发边界见MODEL_CARD.md。\n')
    # Every public file is inspectable; the final integrity manifest excludes itself.
    files=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        name=p.relative_to(out).as_posix();data=p.read_bytes()
        if p.suffix=='.py':ast.parse(data,filename=name)
        assert p.suffix.lower() not in ('.raw','.onnx','.pt','.exe','.dll','.pyd','.pyc')
        if p.suffix.lower() in ('.py','.md','.json','.toml','.txt','.yml','.yaml','.cff','.bib'):
            text=data.decode('utf-8');assert not re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]',text),(name,'absolute Windows path')
            assert '子夜' not in text,(name,'personal name')
        files.append(dict(path=name,bytes=len(data),sha256=digest(data)))
    write(out,'SOURCE_MANIFEST.json',json.dumps(dict(version=VERSION,files=files,weights_included=False,
        RAW_included=False,SDK_or_author_DLL_included=False,public_source_not_published=True),indent=2))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,out.name+'/'+p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    report=dict(directory=str(out),archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,sha256=digest(archive.read_bytes()),
        integrated_license='AGPL-3.0-only',private_files_excluded=True,source_math_extraction_checks=True,published=False)
    (out.parent/'delivery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);build(p.parse_args().output)
