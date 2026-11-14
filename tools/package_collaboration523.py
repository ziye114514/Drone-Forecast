"""Create a dependency-complete collaboration snapshot without moving sources."""
from pathlib import Path
import ast,collections,copy,hashlib,json,os,re,shutil,sys,textwrap,zipfile
BASE=Path(__file__).resolve().parents[2];ROOT=BASE/'event_drone_forecast'
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
from run_rotor_only import SOURCE_FILES

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(text):return textwrap.dedent(text).lstrip()

if __name__=='__main__':
    parent=BASE/'output/collaboration_20261113';parent.mkdir(exist_ok=True)
    out=parent/'event-drone-forecast-collab';out.mkdir(exist_ok=True)
    if any(p.is_file() for p in out.rglob('*')):
        provenance=json.loads((out/'SOURCE_PROVENANCE.json').read_text('utf8'))
        assert provenance['version']=='research517-collab-20261113', 'Only rebuild this generated collaboration snapshot'
    def write(name,text):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(clean(text),'utf8')
    source_log={}
    def cp(source,target,reason):
        p=out/target;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,p)
        source_log[target]=dict(source_relative=str(source.relative_to(BASE)).replace('\\','/'),
            original_sha256=sha(source),packaged_sha256=sha(p),reason=reason)
    # SOURCE_FILES are required by the runtime's reproducibility snapshot,
    # even for optional old branches. Resolve their Python dependencies too.
    index={}
    for p in sorted(ROOT.rglob('*.py')):
        rel=p.relative_to(ROOT)
        if any(x in rel.parts for x in ('build','__pycache__','.cache','Ultralytics')):continue
        if rel.parts[:2]==('research','helix_reference'):continue
        parts=list(rel.with_suffix('').parts)
        if parts[0]=='src':parts=parts[1:]
        if parts[-1]=='__init__':parts=parts[:-1]
        key='.'.join(parts)
        if key not in index or rel.parts[0]=='src':index[key]=p
    reverse={p:k for k,p in index.items()}
    by_stem=collections.defaultdict(list)
    for p in set(index.values()):
        if p.stem!='__init__':by_stem[p.stem].append(p)
    for name,paths in by_stem.items():
        if len(paths)==1:index.setdefault(name,paths[0])
    selected={ROOT/f for f in SOURCE_FILES}
    selected.update(ROOT/f for f in ('run_rotor_background_memory_candidate.py','run_rotor_local_contrast_candidate.py',
        'tools/check_background499.py','tools/check_background_endpoints510.py',
        'tools/score_rotor164.py','tools/compare_final_pipeline.py'))
    pending=collections.deque(p for p in selected if p.suffix=='.py');seen=set()
    while pending:
        p=pending.popleft()
        if p in seen:continue
        seen.add(p);module=reverse.get(p,'');package=module if p.name=='__init__.py' else module.rpartition('.')[0]
        for node in ast.walk(ast.parse(p.read_text('utf-8-sig'),filename=str(p))):
            names=[]
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                value=node.value.replace('\\','/')
                if '\n' not in value and len(value)<300 and Path(value).suffix in ('.py','.json','.yaml','.yml'):
                    for q in (ROOT/value,p.parent/value):
                        if q.is_file() and q.resolve().is_relative_to(ROOT.resolve()) and q not in selected:
                            selected.add(q)
                            if q.suffix=='.py':pending.append(q)
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    parts=package.split('.') if package else []
                    prefix=parts[:len(parts)-node.level+1]
                    name='.'.join(prefix+([node.module] if node.module else []))
                else:name=node.module or ''
                names=[name]+[name+'.'+a.name for a in node.names if name]
            for name in names:
                for length in range(1,len(name.split('.'))+1):
                    q=index.get('.'.join(name.split('.')[:length]))
                    if q is not None and q not in selected:selected.add(q);pending.append(q)
    # Original own C++ include chain and CUDA code, without author HelixTrack.
    selected.update(p for p in (ROOT/'native').iterdir() if p.suffix in ('.cpp','.cu') and p.name!='helixtrack_bridge.cpp')
    for p in sorted(selected):
        assert p.is_file(),p
        cp(p,'code/'+str(p.relative_to(ROOT)).replace('\\','/'),'registered runtime source or transitive/worker/native dependency')
    # Runtime AST loaders and vendored resources use filesystem assets rather
    # than Python imports. Keep their complete small license-bearing snapshots.
    for name in ('filterpy_linear_upstream','filterpy_process_upstream','filterpy_imm_upstream','filterpy_mmae_upstream',
                 'circular_center_calibration_168b85c1','evperiscope'):
        folder=ROOT/'external'/name
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ('.git','__pycache__','.cache')):continue
            if p.suffix.lower() in ('.py','.json','.md','.txt','.toml') or p.name.upper() in ('LICENSE','NOTICE','COPYING'):
                cp(p,'code/'+str(p.relative_to(ROOT)).replace('\\','/'),'third-party snapshot with original notices')
    for p in (ROOT/'src/event_drone_forecast/_vendor').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            cp(p,'code/'+str(p.relative_to(ROOT)).replace('\\','/'),'vendor resource and original license')
    cp(ROOT/'LICENSE','LICENSE','existing project license retained')
    cp(BASE/'output/uncertainty216_20261108/geometry_kernels.dll','binaries/windows-x64/geometry_kernels.dll',
       'own compiled216 kernel; OpenCV/VC runtime must be installed separately')
    # A single explicitly logged path-only adaptation in the distributed copy.
    p=out/'code/native_ellipse_hypotheses.py';text=p.read_text('utf8')
    old="sdk=Path('D:/Prophesee/third_party/bin')"
    assert old in text
    text=text.replace(old,"sdk=Path(os.environ.get('EDRF_OPENCV_BIN','D:/Prophesee/third_party/bin'))")
    p.write_text(text,'utf8');source_log['code/native_ellipse_hypotheses.py']['packaged_sha256']=sha(p)
    source_log['code/native_ellipse_hypotheses.py']['adaptation']='Only SDK/OpenCV DLL directory is configurable via EDRF_OPENCV_BIN; numerical code unchanged.'
    actual=json.loads((BASE/'output/raw_bg_memory517_20261113/results.json').read_text('utf8'))
    compact=dict(pooled=actual['pooled'],sources={n:{a:s['original_available'] for a,s in c.items()} for n,c in actual['cases'].items()},
        replay_parity=actual['replay_parity'],all6_actual_RAW=True,original102_preserved=True,
        independent_new_video_accuracy_passed=False,FPS30_proven=False)
    write('results/development_summary.json',json.dumps(compact,ensure_ascii=False,indent=2))
    fred=json.loads((BASE/'output/frozen_fred522_20261113/results_pending_review.json').read_text('utf8'))
    write('results/fred11_summary.json',json.dumps(dict(cases={a:{k:c[k] for k in ('eligible','available','error','positive_coverage')} for a,c in fred['cases'].items()},
        fixed_interval_s=[10,16],algorithm_frozen=True,native_alignment_review_pending=True,
        independent_accuracy_passed=False,missing_output_is_failure_not_zero_error=True),indent=2))
    write('results/development_source_hashes.json',json.dumps(json.loads((BASE/'output/raw_bg_memory517_20261113/protocol.json').read_text('utf8'))['source_sha256'],indent=2))
    from physical_publication import freeze,evaluate
    rows=[json.loads(s) for s in (BASE/'output/raw_bg_memory517_20261113/recording4/predictions.jsonl').read_text('utf8').splitlines()]
    row=[r for r in rows if r['t']<=12][-1];snapshot=freeze(row,row['t'])
    write('examples/frozen_prediction.json',json.dumps(dict(snapshot=snapshot,horizons=[.1,.2,.4],expected=evaluate(snapshot,[.1,.2,.4]),
        saved_state_demo_not_fresh_detection=True,source='development4 near12s',metric_force_measured=False),indent=2))
    write('configs/local.example.json',json.dumps(dict(sdk_python='C:/path/to/sdk-python/python.exe',gpu_python='C:/path/to/cuda-python/python.exe',
        sdk_pythonpath='C:/path/to/Metavision/lib/python3/site-packages',sdk_bin='C:/path/to/Metavision/bin',
        opencv_bin='C:/path/to/Metavision/third_party/bin',cuda_path='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2',
        geometry_dll='binaries/windows-x64/geometry_kernels.dll'),indent=2))
    write('requirements-cpu.txt',r'''
        # Tested SDK interpreter: Python3.11.6. SDK itself is installed separately.
        numpy==2.4.6
        scipy==1.17.1
        opencv-python-headless==5.0.0.93
        pillow==12.3.0
    ''')
    write('requirements-report.txt','matplotlib==3.10.8\n')
    write('requirements-gpu.txt',r'''
        # Tested separate GPU interpreter: Python3.13.2.
        # Install CUDA-enabled PyTorch for your own GPU/driver separately.
        # Tested: torch2.13.0+cu130. No neural detector weights are loaded.
        numpy==2.5.1
        opencv-python==5.0.0.93
    ''')
    write('scripts/demo_forecast.py',r'''
        """CPU-only saved-state demonstration; no RAW/SDK/GPU required."""
        from pathlib import Path
        import sys,json
        root=Path(__file__).resolve().parents[1]
        sys.path[:0]=[str(root/'code'),str(root/'code/src')]
        from physical_publication import evaluate
        import numpy as np
        data=json.loads((root/'examples/frozen_prediction.json').read_text('utf8'))
        actual=evaluate(data['snapshot'],data['horizons'])
        assert all(np.max(abs(np.asarray(actual[k])-data['expected'][k]))<1e-7 for k in actual)
        for name,m in list(sys.modules.items()):
            p=getattr(m,'__file__',None)
            if p and name in ('physical_publication','camera_reference_selection','background_relative_force'):
                assert Path(p).resolve().is_relative_to(root/'code'),(name,p)
        print(json.dumps(dict(saved_state_demo=True,physical=actual,real_time_or_new_video_validation=False),indent=2))
    ''')
    write('scripts/run_forecast.py',r'''
        """Portable launcher; inference and original baseline are unmodified."""
        from pathlib import Path
        import argparse,json,os,subprocess,sys
        root=Path(__file__).resolve().parents[1]
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('--config',type=Path,default=root/'configs/local.json')
        p.add_argument('--variant',choices=['latest','baseline'],default='latest')
        p.add_argument('--raw',type=Path,required=True);p.add_argument('--start',type=float,default=10.)
        p.add_argument('--duration',type=float,default=6.);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--dry-run',action='store_true');a=p.parse_args()
        c=json.loads(a.config.read_text('utf8'));env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join([str(root/'code'),str(root/'code/src'),c.get('sdk_pythonpath',''),env.get('PYTHONPATH','')])
        env['EDRF_OPENCV_BIN']=c['opencv_bin'];env['CUDA_PATH']=c['cuda_path']
        env['PATH']=os.pathsep.join([c.get('sdk_bin',''),c['opencv_bin'],env.get('PATH','')])
        entry='run_rotor_background_memory_candidate.py' if a.variant=='latest' else 'run_rotor_local_contrast_candidate.py'
        dll=Path(c.get('geometry_dll','binaries/windows-x64/geometry_kernels.dll'))
        if not dll.is_absolute():dll=root/dll
        cmd=[c['sdk_python'],str(root/'code'/entry),'--raw',str(a.raw.resolve()),'--start',str(a.start),
             '--duration',str(a.duration),'--output',str(a.output.resolve()),'--gpu-python',c['gpu_python'],
             '--rotor-native-geometry-dll',str(dll.resolve())]
        if a.dry_run:
            print(json.dumps(dict(command=cmd,variant=a.variant,config=str(a.config)),ensure_ascii=False,indent=2));sys.exit(0)
        for item in (a.raw,dll,Path(c['sdk_python']),Path(c['gpu_python'])):
            if not item.is_file():p.error('Missing file: '+str(item))
        if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh output folder to preserve results')
        sys.exit(subprocess.run(cmd,cwd=root,env=env).returncode)
    ''')
    write('scripts/evaluate_run.py',r'''
        """0.4s source-time FDE, with missing outputs retained in denominator."""
        from pathlib import Path
        import argparse,sys,json,numpy as np
        root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'code'),str(root/'code/src'),str(root/'code/tools')]
        from score_rotor164 import score
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True)
        p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--start',type=float,default=10.8);p.add_argument('--end',type=float,default=15.6);p.add_argument('--step',type=float,default=.2)
        a=p.parse_args();r=score(a.predictions,a.references,np.round(np.arange(a.start,a.end+1e-8,a.step),6))
        r.update(coverage=r['available']/max(1,r['eligible']),mean_is_conditional_on_available=True,
            reference_kind_must_be_declared_by_collaborator=True,new_temporal_extension_is_not_a_new_recording=False)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2),'utf8')
        print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2))
    ''')
    write('scripts/verify_bundle.py',r'''
        """Standard-library-only archive integrity and syntax check."""
        from pathlib import Path
        import hashlib,json
        root=Path(__file__).resolve().parents[1];manifest=json.loads((root/'MANIFEST.sha256.json').read_text('utf8'))
        for name,h in manifest['files'].items():
            p=root/name
            assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h,name
            if p.suffix=='.py':compile(p.read_text('utf-8-sig'),name,'exec')
        print(f"Verified {len(manifest['files'])} files; Python syntax OK. No RAW/GPU jobs started.")
    ''')
    write('scripts/build_geometry.ps1',r'''
        param([Parameter(Mandatory=$true)][string]$OpenCvInclude,
              [Parameter(Mandatory=$true)][string]$OpenCvLib,
              [string]$CoreLibrary='opencv_core4.lib',[string]$ImgprocLibrary='opencv_imgproc4.lib')
        $ErrorActionPreference='Stop'
        $bundleRoot=Split-Path -Parent $PSScriptRoot
        $buildDir=Join-Path $bundleRoot 'build\geometry'
        New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
        if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'Run in an x64 Visual Studio Developer PowerShell.' }
        $source=Join-Path $bundleRoot 'code\native\arc_uncertainty216.cpp'
        $arguments=@('/nologo','/std:c++17','/EHsc','/O2','/fp:strict','/arch:AVX2','/openmp','/MD','/LD',
            "/I$OpenCvInclude",$source,"/Fo$buildDir\geometry.obj","/Fe$buildDir\geometry_kernels.dll",'/link',"/LIBPATH:$OpenCvLib",$CoreLibrary,$ImgprocLibrary)
        & cl.exe @arguments
        if ($LASTEXITCODE -ne 0) { throw 'Geometry compilation failed.' }
        Write-Output "Set geometry_dll to $buildDir\geometry_kernels.dll"
    ''')
    write('README.md',r'''
        # Event Drone Forecast — 协作源码包

        当前研究主线：原生ON/OFF桨叶信号 → 弧/椭圆或侧视线 → 五圆与可见数量分工 → 可见贡献＋不可测合力 → 未来0.4秒图像位置。
        本包用于继续研发，版本标识 `research-517 / 2026-11-13`。推荐从 [交接](docs/HANDOFF.md) 和 [环境与运行](docs/SETUP.md) 开始。

        当前结果：同六段开发录像、原102起点，实际RAW均FDE **31.471473 → 29.997218px**。开发门槛只低0.002782px，不能称为稳定跨录像30px。
        新测FRED11固定10–16秒**0/25有效预测**，没有可报告的FDE；原生画面与官方标注对齐仍待复核。新录像泛化、真实RPM/牛顿推力与持续30FPS均未通过。

        ## 先跑一个小示例

        ```powershell
        python -m pip install -r requirements-cpu.txt
        python scripts/verify_bundle.py
        python scripts/demo_forecast.py
        ```

        这是已有状态的CPU预测示例，不需要RAW、Metavision或GPU；它不代表在新录像检测成功。

        ## 运行自己的RAW

        依照[SETUP](docs/SETUP.md)分别配置SDK Python、CUDA Python与OpenCV DLL路径，将`configs/local.example.json`复制为`configs/local.json`后填写。

        ```powershell
        python scripts/run_forecast.py --config configs/local.json --variant latest --raw "E:\dataset\train\6\media\event.raw" --start 10 --duration 6 --output runs/latest_001
        python scripts/run_forecast.py --config configs/local.json --variant baseline --raw "E:\dataset\train\6\media\event.raw" --start 10 --duration 6 --output runs/baseline_001
        ```

        输入仅支持当前1280×720的Metavision RAW；流程在Windows环境验证。Linux、不同分辨率、不同SDK版本需要适配。

        ## 目录

        - `code/`：当前入口、运行依赖、C++/CUDA源码和固定小型响应参数，保留相对布局以免改变数值逻辑。
        - `scripts/`：统一运行、评分、CPU示例、包校验、原生几何编译。
        - `docs/`：架构、交接、协作约定、现有结果和限制。
        - `results/`：精简结果与原开发源码指纹。
        - `binaries/windows-x64/`：本机自有几何DLL，依赖另行安装的OpenCV和VC运行库；可从源码重编译。
        - `examples/`：一个已有开发状态的数值预测示例。

        原始RAW/FRED、SDK、虚拟环境、CUDA/PyTorch安装器、NN检测权重和大量历史实验输出独立管理。详见[第三方说明](THIRD_PARTY_NOTICES.md)、[来源清单](SOURCE_PROVENANCE.json)。
    ''')
    write('docs/SETUP.md',r'''
        # 环境与运行

        当前研究入口不是旧`pip install .`的历史CLI。使用本包`scripts/run_forecast.py`。

        ## 两个解释器

        1. SDK/模型进程：本机验证Python3.11.6；依赖见`requirements-cpu.txt`。另行安装可读取EVT3的Metavision SDK，使`metavision_core`、`metavision_hal`能导入。SDK不包含在包内。
        2. GPU进程：本机验证Python3.13.2、PyTorch2.13.0+cu130、NumPy2.5.1、OpenCV5.0.0.93。需要NVIDIA显卡/驱动和可用NVRTC；当前加载器使用`CUDA_PATH`下的`nvrtc64_*.dll`。PyTorch用于事件计算与自有CUDA核，不加载神经检测器。

        这些是已测版本记录，并不代表所有设备都应安装同一CUDA组合。可使用同一个兼容SDK且支持CUDA的解释器，尚未在其他组合完整验收。

        ## 配置字段

        `sdk_python`、`gpu_python`填写解释器路径；`sdk_pythonpath`填写SDK的Python目录；`sdk_bin`填写SDK DLL目录；`opencv_bin`填写SDK自带OpenCV DLL目录；`cuda_path`填写本机CUDA根目录。`geometry_dll`可用包内自有DLL或重编译路径。

        包内唯一针对原运行文件的适配是`native_ellipse_hypotheses.py`增加`EDRF_OPENCV_BIN`环境变量读取；数值代码未改。原工程文件没有被该打包程序修改。

        ```powershell
        Copy-Item configs/local.example.json configs/local.json
        # 用编辑器填写本机路径后，先只检查命令：
        python scripts/run_forecast.py --config configs/local.json --raw "E:\data\event.raw" --output runs/check --dry-run
        ```

        ## 几何DLL

        在x64 Visual Studio Developer PowerShell中，提供OpenCV头文件、库目录与实际库名：

        ```powershell
        .\scripts\build_geometry.ps1 -OpenCvInclude "C:\SDK\third_party\include" -OpenCvLib "C:\SDK\third_party\lib"
        ```

        源码入口`code/native/arc_uncertainty216.cpp`逐层包含215、206与椭圆拟合实现；需要OpenCV core/imgproc、MSVC OpenMP及AVX2。不同OpenCV发行版需通过`-CoreLibrary`、`-ImgprocLibrary`指定库文件。

        ## 输出与评分

        每次使用新输出目录。`predictions.jsonl`保存各源时间状态、力学代理和0.1/0.2/0.4秒预测；`summary.json`保存处理包数、遗漏与运行信息。默认精度入口不丢包、不按实时墙钟追赶；不能把它称作30Hz实时预测。

        `scripts/evaluate_run.py --predictions ... --references ... --output ...`读取参考JSON：`{"records":{"10.8":{"t":10.8,"state":"present","box":[x1,y1,x2,y2]}}}`。必须记录参考类型、固定全部查询起点，同时报告可用比例与缺失数量；0个输出时FDE为null，不能记0px。
    ''')
    write('docs/ARCHITECTURE.md',r'''
        # 当前检测与预测链

        ```mermaid
        flowchart TD
          A[原生RAW x y polarity timestamp] --> B[GPU每像素ON/OFF与周期证据]
          B --> C[原生支持重建与弧/椭圆/线几何]
          C --> D[五圆可见性和四/三/二/侧视资格]
          D --> E[当前平移观测与周期代理]
          E --> F[可见贡献 + 不可测合力状态]
          A --> G[目标区域外的共同背景位移]
          G --> H[共享端点噪声与参考坐标补偿]
          H --> F
          F --> I[固定小型响应参数与未来0.4秒传播]
          I --> J[窗口/FOV观测投影与参考假设权重]
        ```

        ## 从哪里改

        |环节|主要源码（位于code）|
        |---|---|
        |统一原入口与读流|run_rotor_only.py、live_decode.py、cooperative_raw_reader.py|
        |桨叶事件前端|rotor_only_front.py、rotor_cuda_device.py、rotor_cuda_worker.py、predictive_harmonics.py|
        |几何/五圆/缺桨|fast_structure_front.py、continuous_five_circle.py、five_circle_model.py、curvature_visibility.py、native/|
        |周期与可见贡献|rotor_thrust_bank.py、anonymous_rotor_group.py、equilibrium_gain_prior.py、visibility_period.py|
        |二力与方向|relative_rotor_observer.py、shared_endpoint_observer.py、inverse_normal_response.py、anisotropic_response.py|
        |固定响应与观测投影|learned_rotor_predictor.py、plane_conditioned_response.py、native_fov_observation.py、window_position_observation.py|
        |本轮背景参考|background_reference_predictor.py、background_relative_force.py、background_endpoint_observer.py、background_selection_predictor.py、camera_reference_selection.py|
        |最终传播|physical_publication.py|

        四/三/二是可观测几何资格，不保证等同真实可见电机数量。三桨缺失位置的几何假设不等于观测到第四桨转速；侧视不强制完整五圆。无需机头/机尾标签。

        二力语义：可用当前周期和几何只提供像素域可见贡献代理；隐藏桨推力、重力、风和阻力归不可测合力。背景参考是坐标运动，不是第三种机械力。未知质量、尺度和推力系数不能由单目像素误差唯一分开。

        最新背景参考版本共享一个桨叶前端，在静止/运动背景参考坐标下各运行原力学状态。权重由当前及过去背景事件决定，并沿用原0.35秒状态尺度保留资格；不是整机NN或传统框运动预测支路。两状态有共同观测，不能当独立测量增加置信度。

        固定JSON是小型源训练响应参数，非NN检测权重。当前实际在线周期—力增益尚未稳定学到，不能称真实RPM/牛顿推力已经测准。
    ''')
    write('docs/RESULTS.md',r'''
        # 结果与边界（2026-11-13）

        实际六段RAW，源时间未来0.4秒，原102个起点：

        |入口|均FDE px|中位px|30.5px内|
        |---|---:|---:|---:|
        |487局部归一化基线|31.471473|25.379331|66/102|
        |509背景共享端点|30.600825|25.082446|64/102|
        |512当前背景证据|30.138842|24.754987|67/102|
        |517补偿资格记忆（当前）|29.997218|24.904973|67/102|

        每段181包、零跳包；517与冻结观测回放逐帧预测最大差约2.84e-14px。改善1.474255px，约4.68%。收益主要来自原较差的3、4号；其他四段基本保持原值。不是每段都低于30px。

        六段已暴露开发，参考是弱桨叶轮廓中心，非重心/姿态/飞控真值。30px余量仅0.002782px，不能把边界通过写成稳定达标。完整数据见results/development_summary.json。

        ## 新录像

        原预留recordings文件解封后是标定板，27个原生检查帧无无人机；25个固定查询两入口均0误报，只能作阴性检查，不能验证无人机轨迹精度。

        随后按用户要求选本地未查到使用记录的FRED11，固定10–16秒、两入口先冻结再看全量官方标注。两入口均0/25有效预测、FDE为null；原生画面与官方事件框对应关系尚未最终复核。不能称独立泛化通过，不能拿缺失输出当0误差。

        自采train1–6历史上均抽样或试跑过。FRED11已经本轮暴露，下一次不得继续称它全新盲测。

        ## 实时性

        当前入口是保留全部观测的离线精度运行。背景LK与双参考力学增加了成本，尚未完成30FPS验收；旧阶段某个吞吐数不能代表本版本。先解决新录像有效输出与精度，再优化FPS。
    ''')
    write('docs/HANDOFF.md',r'''
        # 接手顺序与当前卡点

        推荐入口：`code/run_rotor_background_memory_candidate.py`（517）。保留对照：`code/run_rotor_local_contrast_candidate.py`（487）。统一启动见README。先看RESULTS，不把开发集29.997px称为项目全面成功。

        ## 用户必须保留的研究方向

        原生ON/OFF先看桨叶；通过原弧/椭圆推桨心、五圆可见性分工；侧视采用线/局部周期而非强求四桨；二力是“可见贡献＋不可测合力”，隐藏桨推力归未知项。无需机头机尾。NN即使以后引入，也不能掩盖用户这条主线的独立贡献。

        ## 本轮只攻了一个根因

        背景共同运动污染目标位移/速度与不可测力。先查到仿射参考在跟踪点支撑外外推，改共同平移；再保留相邻图像端点噪声相关性；再用背景证据在两种参考坐标假设间分配权重；最后修正当前背景变弱就立即丢掉历史补偿资格的时序问题。没有扫录像阈值或模型系数，原前端/周期/几何保持。

        这一环在开发集有实际收益，但共同平移仍不能覆盖强视差、旋转、景深差与无背景纹理，先保留此边界，不再靠目标误差扫背景权重。

        ## 接手后第一条工作

        **先查FRED11固定10–16秒为什么没有任何有效桨叶预测。** 从原始事件检查“有无人机吗、事件目标位置与官方框是否对齐、目标是否小到没有周期/轮廓支持”。本包results保留0/25；原工作区`output/frozen_fred522_20261113`有冻结预测、日志及27张原生检查帧，未放入源码包，可由项目持有人另行提供。

        先定位丢失在严格周期种子、原生ON/OFF传播、几何/启动资格中的哪一环，再查相应文献和现有代码，选择一个根因修改。不要直接降阈值到检出背景，也不要只在FRED11这个时间/坐标打补丁。FRED11已经看过，后续调过算法后需要另选未暴露来源作冻结验证。

        ## 其他尚未解决的问题

        - 周期代理到有效推力的可辨识性：输入合格时段短、激励弱，在线增益尚未稳定学习；几何补出隐藏桨不等于补出推力真值。
        - 部分开发起点当前位置较准，但0.4秒速度/合力延续仍差；要区分观测错误与预测发出后新的操纵，不要求预知未来，也不删除这类起点评分。
        - 固定模型对未见尺寸/运动条件的有效性未证实；开发30px不等于新视频30px。
        - 背景参考开销与实际发布延迟尚未优化；持续30Hz暂未通过。

        用户无同步ESC、飞控、IMU或测速真值，无法补采这些日志；不要将它们作为继续工作的前提。不要改两处yolov10-main、原始数据/SDK/权重或已发布NN包。

        本包整理时所有RAW/GPU作业均已退出；没有自动继续运行的后台任务。
    ''')
    write('docs/COLLABORATION.md',r'''
        # 协作约定

        一个根因一个分支/PR。开始前写清：哪条证据表明它是当前最大问题、改动怎样影响最终精度、如何排除只适配某段录像。不要并行改同一前端与力学状态后无法归因。

        建议先由一位协作者负责FRED无输出的可见性/前端诊断，另一位只负责环境移植与复现517/487，两者不同时改算法。推力可辨识性需要独立论证，避免把欠观测频率当力真值。

        运行RAW前冻结源码和参数哈希；任务运行中不得编辑`run_rotor_only.py`的`SOURCE_FILES`中任何文件，包括未启用的候选。每次完整结束后再改。保留旧基线，所有原查询起点不变；同时报告均值、可用比例、缺失输出与逐录像结果。不可用样本不得记0px。

        新录像先登记来源/区间/评价口径，再锁定算法；不据测试结果改时间、选择最好片段或扫权重。需要调过的素材转开发集，再找新来源验证。若只有旧素材可用，直接写明是开发结果。

        对突然机动用新观测更新下一条0.4秒预测，不倒改此前预测。优先精度；新视频通过前不以优化FPS替代算法问题。

        PR最少交付：根因证据、改动文件、完整同起点结果和缺失数、失败/退化场景、源码/模型指纹。无需堆大量重复验证。
    ''')
    write('AGENTS.md',r'''
        # 协作入口

        先读README.md、docs/HANDOFF.md、docs/RESULTS.md。当前主线是517纯ON/OFF桨叶、五圆可见性、可见贡献＋不可测合力；原487对照必须保留。
        一次只攻最大且有证据的根因，不按录像编号/时间/坐标打补丁，不删失败起点，不把缺失输出当0误差。RAW作业期间冻结全部SOURCE_FILES。
        本包用于协作，不含原RAW/SDK/NN权重。新视频泛化和30FPS未通过，真实机械RPM/牛顿推力未测准。
        优先查FRED11固定10–16秒0/25有效输出的阶段性失效；它现已暴露，不再视作盲测。
    ''')
    write('THIRD_PARTY_NOTICES.md',r'''
        # 第三方来源与包范围

        项目自有源码沿用随附MIT LICENSE；第三方源码保持各自原许可证与声明，不以本项目MIT重新授权。

        - FilterPy固定快照：`code/external/filterpy_*`及`code/src/event_drone_forecast/_vendor/`，保留MIT LICENSE与已有UPSTREAM_SNAPSHOT资源。上游 https://github.com/rlabbe/filterpy 。
        - circular-center-calibration：`code/external/circular_center_calibration_168b85c1`保留Apache-2.0 LICENSE、NOTICE和已有快照。上游 https://github.com/xiahaa/circular-center-calibration 。这是历史可选参考，当前自采数据不宣称有已知相机内参。
        - evperiscope小型历史快照若随依赖保留，附原MIT声明；并非当前新增算法贡献。
        - NumPy、SciPy、OpenCV、Pillow、Matplotlib、PyTorch、CUDA、Metavision SDK由接手人独立安装，遵循各自条款。本包不附带它们的安装器/运行库，也没有NN检测权重。
        - `binaries/windows-x64/geometry_kernels.dll`来自项目自有216几何源码；动态链接依赖需另行安装。作者HelixTrack代码、HelixTrack适配DLL与未确认重分发许可的整仓库不随包分发。
        - RAW、FRED下载文件、原始官方图像和人工原始数据不在本源码包内。数据访问和发布授权由数据提供方另行管理。

        方法参考见原论文：Relative State Estimation using Event-Based Propeller Sensing（https://arxiv.org/abs/2604.18289）；Event-Based Motion Segmentation by Motion Compensation（https://arxiv.org/abs/1904.01293）；FAST-Dynamic-Vision（https://arxiv.org/abs/2103.05903）。本包不是这些论文的官方实现，不借用其成绩作为本项目结果。
    ''')
    write('CHANGES_FROM_RESEARCH.md',r'''
        # 与原工程的区别

        算法源码按SOURCE_FILES与传递依赖整理到code/，保持原相对布局与原数字后缀；没有把数百次研究重排成看似全新的算法。
        唯一路径适配：`code/native_ellipse_hypotheses.py`增加`EDRF_OPENCV_BIN`读取，原默认路径保留；对比原文件只有这一行差异。SOURCE_PROVENANCE.json逐份记录原与打包后SHA256。
        新增统一运行/评分脚本、CPU数值示例、配置、文档、精简结果与完整性清单。未附原始数据、虚拟环境、历史output、已下载整仓库和NN权重。
        原工程目录和既有发布包保留原状。主源码树中的历史入口在必要的依赖/快照范围内仍有保留；协作者只从README两个统一变体开始。
    ''')
    write('.gitignore',r'''
        __pycache__/
        *.py[cod]
        .venv*/
        configs/local.json
        runs/
        build/
        *.raw
        *.tmp_index
        *.onnx
        *.pt
        .env
    ''')
    write('SOURCE_PROVENANCE.json',json.dumps(dict(version='research517-collab-20261113',files=source_log,
        algorithm_source_is_frozen517_with_one_documented_path_adaptation=True,
        source_registry_count=len(SOURCE_FILES),bundled_original_paths=len(source_log),original_workspace_untouched=True),ensure_ascii=False,indent=2))
    files={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write('MANIFEST.sha256.json',json.dumps(dict(files=files,manifest_excludes_itself=True),indent=2))
    # Do not archive until the copied source has passed separate checks.
    print(json.dumps(dict(folder=str(out),files=len(files)+1,bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        registered=len(SOURCE_FILES),dependency_python=len(seen),source_files=len(source_log)),indent=2))
