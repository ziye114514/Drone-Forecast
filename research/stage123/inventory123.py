"""Audit historical RAW exposure from saved metadata, never infer unseen by name."""
from pathlib import Path
import datetime,json,re,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage109')]
from audit109 import save,sha
OUT=ROOT.parent/'output/stage123_frozen_extension_20261023'

def normalize(value):return str(value).replace('\\','/').lower()
def exposure():
    evidence=[];output=ROOT.parent/'output'
    for path in output.rglob('*.json'):
        if path.is_relative_to(OUT) or path.stat().st_size>300000:continue
        if not any(w in path.name for w in ['protocol','manifest','config']):continue
        try:data=json.loads(path.read_text('utf-8-sig'))
        except (ValueError,OSError):continue
        def walk(x,location=''):
            if isinstance(x,dict):
                for key,value in x.items():
                    if isinstance(value,str) and value.lower().endswith('.raw'):
                        match=re.search(r'e:/dataset/train/(\d+)/',normalize(value))
                        if match:
                            start=next((x[k] for k in ['start','start_s','start_seconds'] if isinstance(x.get(k),(int,float))),None)
                            duration=next((x[k] for k in ['duration','duration_s','duration_seconds'] if isinstance(x.get(k),(int,float))),None)
                            evidence.append(dict(recording=int(match[1]),raw=value,start=start,duration=duration,
                                end=None if start is None or duration is None or duration==0 else start+duration,source=str(path.relative_to(ROOT.parent)),location=location))
                    elif isinstance(value,(dict,list)):walk(value,location+'/'+str(key))
            elif isinstance(x,list):
                for i,value in enumerate(x):walk(value,location+'/'+str(i))
        walk(data)
    result=[]
    for recording in range(1,7):
        ev=[x for x in evidence if x['recording']==recording];known=[x for x in ev if x['end'] is not None]
        ranges=sorted(set((float(x['start']),float(x['end'])) for x in known));merged=[]
        for lo,hi in ranges:
            if merged and lo<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],hi)
            else:merged.append([lo,hi])
        result.append(dict(recording=recording,known_used_ranges=merged,metadata_with_unknown_extent=sum(x['end'] is None for x in ev),metadata_entries=len(ev)))
    save(OUT/'historical_exposure123.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),recordings=result,evidence=evidence,
        caution='Metadata audit is incomplete for ad-hoc/manual inspection and detector training; outside known ranges is not proof of unseen data. Freeze model and call this extension, not blind test.'))
    print(json.dumps(result),flush=True)

if __name__=='__main__':exposure()
