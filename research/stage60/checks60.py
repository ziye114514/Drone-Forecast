"""Supplemental audit fixes invalid-frame handling, without editing frozen inference."""
from pathlib import Path
import argparse,datetime,hashlib,inspect
import forecast60 as current
PROTOCOL=current.OUT/'audit_supplement.json'

def source():
    source=inspect.getsource(current.audit)
    old="        for name in history:assert qa['coefficients'][name]==qb['coefficients'][name],(case,a['t'],name)"
    new="""        if qa['valid']:
            assert all(name in qa['coefficients'] and name in qb['coefficients'] for name in history)
            for name in history:assert qa['coefficients'][name]==qb['coefficients'][name],(case,a['t'],name)
        else:
            assert qa['coefficients']==qb['coefficients']=={}"""
    assert source.count(old)==1;return source.replace(old,new)

def freeze():
    current.verify();assert not PROTOCOL.exists()
    current.save(PROTOCOL,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=current.sha(Path(__file__)),
       derived_audit_sha256=hashlib.sha256(source().encode()).hexdigest(),
       reason='Initial full pilot inference completed40/80 and video. Old new-audit indexed hold coefficients on valid=False frames whose coefficients are {} by existing contract, raising KeyError before regression saved. Fix checker only: invalid must be empty in both; valid requires all six keys and exact comparison. Complete15 off comparison and every other assertion retained. Frozen forecast60/inference untouched.',
       initial_terminal_batch=1447,pilot_predictions_retained=True))
    (current.OUT/'derived_audit60.py').write_text(source(),encoding='utf-8');print('60 strict audit supplement frozen',flush=True)

def verify():
    current.verify();p=current.load(PROTOCOL);assert p['helper_sha256']==current.sha(Path(__file__)) and p['derived_audit_sha256']==hashlib.sha256(source().encode()).hexdigest()

def audit(case):
    verify();namespace=dict(current.__dict__);exec(compile(source(),'<60 exact audit invalid-frame fix>','exec'),namespace);namespace['audit'](case)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','audit']);p.add_argument('--case',choices=list(current.CASES));a=p.parse_args()
    if a.action=='audit':audit(a.case)
    else:freeze()
