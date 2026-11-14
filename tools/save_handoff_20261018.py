"""Non-destructive local source/protocol snapshot; excludes RAW, videos and JSONL."""
from pathlib import Path
import datetime,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent


def main():
    files=set()
    for folder in [ROOT/'research',ROOT/'src',ROOT/'tools']:
        for pattern in ['*.py','*.md']:
            files.update(p for p in folder.rglob(pattern) if '__pycache__' not in p.parts)
    files.update(ROOT.glob('*.py'));files.update(ROOT.glob('*.md'))
    files.update(PROJECT/p for p in ['WORKLOG.md','PROJECT_STATUS_20261018.md'])
    for pattern in ['*.html','*.json']:
        files.update((PROJECT/'output/research_overview_20261018').glob(pattern))
    statuses=[]
    active=[]
    for stage in range(19,43):
        folders=list((PROJECT/'output').glob(f'stage{stage}_*20261018'));assert len(folders)==1,(stage,folders)
        folder=folders[0];assert (folder/'protocol.json').exists()
        if not list(folder.glob('*.html')):
            active.append(dict(stage=stage,folder=str(folder),status='In progress; inspect process state and completion files before resuming.'))
            files.add(folder/'protocol.json')
            continue
        for pattern in ['*.json','*.md','*.html']:
            files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000 and 'failed_border_preparation_v1' not in p.parts)
        statuses.append(dict(stage=stage,folder=str(folder),html=[p.name for p in folder.glob('*.html')],analysis_exists=(folder/'analysis.json').exists()))
    checkpoint=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_status='paused',user_intent='User explicitly requested wrap up current work and stop. No further experiments or background jobs.',
        last_completed_stage=max(s['stage'] for s in statuses),in_progress=active,reset_credit_consumed=False,
        next_action='Await user resume. Completed42 RAW body integration,40 detector adaptation and41 continuous future validation. If resumed: profile single-frame102ms latency, investigate turn/stop forecast overshoot and separate horizon feedback; then test whether existing rotor physics adds value on improved body state. Do not promote default or claim unseen-airframe/RPM/pose accuracy.',
        stages=statuses,scope='Source, docs, protocols and summary reports only. RAW, MP4, PNG and full JSONL stay in original directories, not included in snapshot.')
    out=PROJECT/'output/checkpoints';out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');archive=out/f'research_through_stage{checkpoint["last_completed_stage"]}_{stamp}.zip'
    manifest={p.relative_to(PROJECT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.relative_to(PROJECT).as_posix())
        z.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('HANDOFF_CHECKPOINT.json',json.dumps(checkpoint,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(name)).hexdigest()==value for name,value in manifest.items())
    checkpoint.update(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source_summary_files=len(files),zip_verified=True)
    (out/'latest_handoff.json').write_text(json.dumps(checkpoint,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(archive=str(archive),files=len(files),verified=True),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
