"""Prepare a local reviewable source candidate; never publishes or downloads."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEXT = {'.py', '.md', '.json', '.toml', '.txt', '.yaml', '.yml'}
VENDORS = ('filterpy_linear_upstream', 'filterpy_process_upstream', 'filterpy_imm_upstream',
           'filterpy_mmae_upstream', 'circular_center_calibration_168b85c1', 'evperiscope')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(target, demos=None, live_demos=None):
    target = Path(target).expanduser().resolve()
    if target.exists(): raise ValueError('Candidate directory must not exist: ' + str(target))
    target.mkdir(parents=True, exist_ok=False)
    selected = set()
    # Keep historical source layout: runtime currently imports earlier stages
    # and adapts their source ASTs. This is intentionally not a minimal core.
    selected.update(p for p in ROOT.glob('*.py'))
    selected.update(p for p in ROOT.glob('*.md'))
    selected.update(ROOT / n for n in ('LICENSE', '.gitignore', 'pyproject.toml',
                                      'requirements-lock.txt', 'requirements-report.txt',
                                      'configs/runtime.example.json', 'tools/build_source_candidate.py'))
    selected.update(ROOT/'tools'/n for n in ('check_live_controls.py','compare_native_readers.py',
                                            'run_live_pilot.py','summarize_live_pilot.py',
                                            'check_timed_publication.py','check_body_association.py',
                                            'run_body_association_pilot.py','evaluate_body_association.py',
                                            'replay_live_noise.py','replay_live_gain.py',
                                            'check_position_core.py','build_position_wheel.py',
                                            'install_position_wheel.py','build-wheel-requirements.txt'))
    for directory in ('src', 'research', 'docs'):
        selected.update(p for p in (ROOT / directory).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts
                        and not any(part.endswith('.egg-info') for part in p.parts)
                        and (p.suffix in TEXT or p.name in ('LICENSE','NOTICE'))
                        and 'helix_reference' not in p.parts)
    for vendor in VENDORS:
        selected.update(p for p in (ROOT / 'external' / vendor).rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts
                        and (p.suffix in TEXT or p.name in ('LICENSE', 'NOTICE')))
    for p in sorted(selected):
        dest = target / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    example = target / 'examples'
    example.mkdir()
    demo_files = {}
    if demos is not None:
        demos = Path(demos).resolve()
        for case in ('clear', 'side'):
            src = demos / case / 'predictions.mp4'
            if not src.is_file(): raise ValueError('Demo not found: ' + str(src))
            dest = example / (case + '.mp4')
            shutil.copy2(src, dest)
            demo_files[dest.name] = dict(sha256=digest(dest), source_stage=156,
                                        role='Previously recorded development demonstration, not a new evaluation')
        src = demos / 'forecast_comparison156.png'
        if src.is_file(): shutil.copy2(src, example / 'forecast_comparison.png')
    (example / 'README.md').write_text(
        '# 已有演示\n\nclear.mp4与side.mp4来自整理前156实验：清晰22—30s、侧视110—119.5s。'
        '绿色为原预测，紫色为可选曲线回退，灰色为整机匀速；并非每一帧均有效发布。'
        '50fps为录像播放时间轴，不是在线处理速度。没有新精度、姿态、RPM或真实推力结论。'
        '\n\n未指定 --demos 时，源码包仅包含本说明；数据和权重不随包提供。\n', encoding='utf-8')
    if live_demos is not None:
        live_demos=Path(live_demos).resolve()
        for src,name in [(live_demos/'side/predictions.mp4','live_side.mp4'),
                         (live_demos/'timing.png','live_timing.png')]:
            if not src.is_file():raise ValueError('Live demo missing: '+str(src))
            shutil.copy2(src,example/name)
            demo_files[name]=dict(sha256=digest(example/name),role='Optional low-latency position branch, not a thrust/pose result')
        with (example/'README.md').open('a',encoding='utf-8') as stream:
            stream.write('\n## 新增live模式\n\nlive_side.mp4为4秒侧视低延迟位置通道；紫线是补偿处理延迟后的预测。目标5Hz发布，实际频率和有效空档见docs/LIVE_MODE.md；视频固定FPS不能作为实时性能证据。live_timing.png保留全部发布与有效发布的区别。\n')
    (target / 'SOURCE_SCOPE.md').write_text(
        '# 候选包范围\n\n自有源码、文档与已随附第三方小型源码快照。为保留传递导入，research历史布局尚未裁减。'
        '唯一推荐完整RAW源码入口forecast.py；新增0.3.1可安装position运动核心，原pip控制台入口仍是历史流程。'
        '\n\n不含SDK、虚拟环境、私有configs/local.json、RAW、模型权重、其他下载整仓库或整套历史output。'
        'position核心已离开原目录用本机依赖运行；完整RAW尚未进行干净机器安装，参考模型获取路径尚不完整；不能宣称clone即用。\n', encoding='utf-8')
    rows = []
    for path in sorted(target.rglob('*')):
        if not path.is_file(): continue
        relative = path.relative_to(target)
        assert relative.as_posix() != 'configs/local.json'
        assert path.suffix.lower() not in ('.raw', '.onnx', '.pt', '.exe', '.dll', '.pyd')
        assert '__pycache__' not in path.parts
        assert 'helix_reference' not in relative.parts
        rows.append(dict(path=relative.as_posix(), size=path.stat().st_size, sha256=digest(path)))
    manifest = dict(status='local source candidate, not published', minimal_core_extracted=False,
                    minimal_position_core_extracted=True,full_raw_runtime_packaged=False,
                    weights_included=False, raw_included=False, private_config_included=False,
                    files=rows, demo_files=demo_files)
    (target / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = target.with_suffix('.zip')
    if archive.exists(): raise ValueError('Archive already exists: ' + str(archive))
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(target.rglob('*')):
            if path.is_file(): z.write(path, target.name + '/' + path.relative_to(target).as_posix())
    result = dict(directory=str(target), archive=str(archive), files=len(rows) + 1,
                  archive_bytes=archive.stat().st_size, archive_sha256=digest(archive))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--demos', type=Path, help='Optional existing156 output directory; no new inference')
    p.add_argument('--live-demos',type=Path,help='Optional completed live pilot directory')
    a = p.parse_args()
    build(a.output, a.demos, a.live_demos)
