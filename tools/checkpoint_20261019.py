"""Verify current reports/source supplements and archive an explicit goal status."""
from pathlib import Path
import argparse,datetime,hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUTPUT=PROJECT/'output'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def run(snapshot=False, goal_status='active'):
    statuses=[];completed=[];pending=[]
    for n in range(19,100):
        folders=[p for p in OUTPUT.glob(f'stage{n}_*') if p.is_dir() and (p/'protocol.json').exists()]
        if not folders:continue
        assert len(folders)==1,(n,folders);folder=folders[0];protocol=load(folder/'protocol.json')
        for name,value in protocol['source_hashes'].items():assert sha(ROOT/name)==value,(n,name)
        if n>=43:
            for p in folder.rglob('protocol.json'):
                for name,value in load(p).get('source_hashes',{}).items():assert sha(ROOT/name)==value,(p,name)
        pages=list(folder.glob('*.html'));bad=[]
        for p in pages:
            for ref in re.findall(r'(?:href|src)="([^"]+)"',p.read_text(encoding='utf-8')):
                if '://' in ref or ref.startswith('#'):continue
                if not (p.parent/ref).exists():bad.append(ref)
        assert not bad,(n,bad)
        if pages:
            completed.append(n);save(folder/'delivery_checks_20261019.json',dict(current_source_hashes_unchanged=True,nested_protocol_source_hashes_unchanged=True,html_local_links_valid=True,reports=[p.name for p in pages]));statuses.append(dict(stage=n,folder=str(folder),complete_reports=[p.name for p in pages]))
        else:pending.append(dict(stage=n,folder=str(folder),status='Protocol exists but no report; inspect actual tool handles before restarting.'))
    # Executed supplements are separate from immutable original protocol cores.
    supplement_checks=[]
    for n in [44,45,46,47]:
        folder=next(p for p in OUTPUT.glob(f'stage{n}_*') if (p/'protocol.json').exists())
        for name,key,source in ({44:[('unprofiled_supplement.json','helper_sha256','benchmark_unprofiled44.py')],45:[('inference_context_supplement.json','helper_sha256','inference_context45.py')],46:[('topology_supplement.json','helper_sha256','convert_sorted46.py')],47:[]}[n]):
            assert load(folder/name)[key]==sha(ROOT/f'research/stage{n}'/source);supplement_checks.append(dict(stage=n,name=name,source_unchanged=True))
        if n==46:
            for name,value in load(folder/'path_scope_supplement.json')['helper_hashes'].items():assert sha(ROOT/f'research/stage46'/name)==value
        if n==47:
            for name,value in load(folder/'pipe_proxy_supplement.json')['source_hashes'].items():assert sha(ROOT/name)==value
    summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status=goal_status,last_completed_stage=max(completed),completed=completed,in_progress=pending,supplement_checks=supplement_checks,
        next_action='Preserve48 same-position physics control. Investigate repeated rotor-reference resets starving0.6s warmup: retain axis-only state; for frequency models test coordinate-consistent gain/covariance/queued-ratio transfer when S0 changes, with algebraic invariance tests before real causal replay. Verify prior art; do not claim unknown true mass/wind/RPM identified.',
        scope='Source/protocol/report checkpoint only; no claim entire long-term goal achieved. No usage reset/purchase, no automation or external messaging.')
    if goal_status == 'paused':
        summary.update(user_intent='User explicitly requested wrapping up current work and stopping. No new experiments until a new user instruction.',
            running_project_jobs=0,
            process_check='Read-only Win32_Process inventory found no project Python or FFmpeg processes at wrap-up.',
            next_action='Await explicit user resume. Suggested first experiment only: preserve axis-only state and test coordinate-consistent frequency-reference transfer with algebraic invariance checks, then frozen same-input causal comparison against stage48. Stage49 has not been started.')
    save(OUTPUT/'research_overview_20261018/current_delivery_checks_20261019.json',summary)
    if not snapshot:print('Current source/report/supplement checks passed through'+str(max(completed)),flush=True);return
    files=set()
    for folder in [ROOT/'src',ROOT/'research',ROOT/'tools']:
        for pattern in ['*.py','*.md']:files.update(p for p in folder.rglob(pattern) if '__pycache__' not in p.parts)
    files.update(ROOT.glob('*.py'));files.update(ROOT.glob('*.md'));files.update(PROJECT.glob('WORKLOG*.md'));files.update(PROJECT.glob('PROJECT_STATUS*.md'))
    for item in statuses+pending:
        folder=Path(item['folder'])
        for pattern in ['*.json','*.md','*.html']:
            files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in p.parts)
    files.update((OUTPUT/'research_overview_20261018').glob('*.html'));files.update((OUTPUT/'research_overview_20261018').glob('*.json'))
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');archive=OUTPUT/'checkpoints'/f'research_through_stage{max(completed)}_{stamp}.zip';archive.parent.mkdir(exist_ok=True)
    manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(PROJECT).as_posix())
        z.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));z.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True,archive_scope='No RAW/weights/PNG/MP4/JSONL/large diagnostic JSON; retained in original workspace directories.')
    save(OUTPUT/'checkpoints/latest_handoff.json',summary);save(OUTPUT/'checkpoints/latest_progress_20261019.json',summary);print(json.dumps(dict(stage=max(completed),archive=str(archive),files=len(files),verified=True),ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',action='store_true');p.add_argument('--goal-status',choices=['active','paused'],default='active');a=p.parse_args();run(a.snapshot,a.goal_status)
