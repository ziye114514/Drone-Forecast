"""Resume only the empty interrupted negative case; preserve all prior outputs."""
import datetime
from pathlib import Path
from study26 import OUT,CASES,verify,run,analyze,load,save
import study26

if __name__=='__main__':
    verify()
    for case in ('clear6','side6','review4'):
        assert load(OUT/case/'completion.json')['frames']==round(CASES[case]['duration']/.02)
    folder=OUT/'negative5'
    if not (folder/'completion.json').exists():
        files=list(folder.glob('*'))
        assert len(files)==2 and all(p.stat().st_size==0 for p in files),files
        target=OUT/'interrupted_negative5_empty'
        assert folder.resolve().parent==OUT.resolve() and not target.exists()
        folder.rename(target)
        save(OUT/'resume_record.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preserved=str(target),reason='Prior call ended with three completed cases and two zero-byte negative files; no Python process remained. Cause not established. No completed outputs overwritten.'))
        study26.CASES={'negative5':CASES['negative5']}
        run()
    study26.CASES=CASES
    analyze()
