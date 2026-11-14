"""Bounded read-only scan for an independent propeller/RPM dataset already on disk.

No file is opened or modified; only names inside a fixed depth are matched.
The public dataset stays unreachable because its certificate does not verify,
and that is recorded rather than bypassed.
"""
from pathlib import Path
import json,re,time
ROOTS=['E:/dataset','E:/software','E:/tools','E:/experiment','E:/recordings',
       'E:/Downloads','E:/dataset_share_20261006','D:/drone forecast/tmp']
PATTERN=re.compile(r'propeller|qpdet|rpm|blade|rotor',re.I)
DEPTH=3
OUT=Path(__file__).resolve().parents[2]/'output/peak_ownership_20261029'


def walk(root,depth):
    stack=[(Path(root),0)]
    while stack:
        current,level=stack.pop()
        try:entries=list(current.iterdir())
        except (PermissionError,OSError):continue
        for entry in entries:
            yield entry,level
            if entry.is_dir() and level+1<depth:stack.append((entry,level+1))


def main():
    started=time.perf_counter();checked=[];hits=[]
    for root in ROOTS:
        path=Path(root)
        if not path.exists():continue
        checked.append(root)
        for entry,level in walk(root,DEPTH):
            if PATTERN.search(entry.name):hits.append(str(entry))
    result=dict(roots_requested=ROOTS,roots_present=checked,depth=DEPTH,
        name_pattern=PATTERN.pattern,matches=sorted(hits)[:200],match_count=len(hits),
        files_opened=0,public_dataset=dict(url='https://ptak.felk.cvut.cz/personal/spetlrad/propeller_dataset.tar.gz',
            status='unreachable: self-signed certificate, not bypassed, archive not downloaded'),
        wall_s=round(time.perf_counter()-started,1),
        independent_RPM_values_available=False,
        note='Name search only. Absence here does not prove the dataset is nowhere else on the machine.')
    (OUT/'local_rpm_dataset_scan.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='matches'},indent=2,ensure_ascii=False))
    print('matches:',result['matches'][:20])


if __name__=='__main__':main()
