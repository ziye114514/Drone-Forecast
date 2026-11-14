"""Frozen supplementary multi-peak periodicity control and same RAW shadow replay."""
from pathlib import Path
import argparse,datetime,inspect,sys
from multipeak67 import *
import study66 as previous
ROOT=previous.ROOT;OUT=ROOT.parent/'output/stage67_multipeak_periodicity_20261021'
load=previous.load;save=previous.save;sha=previous.sha;rows=previous.rows;CASES=previous.CASES


def sources():
    result=previous.sources()
    for p in [Path(__file__),Path(__file__).with_name('multipeak67.py')]:result[p.relative_to(ROOT).as_posix()]=sha(p)
    return result


def freeze():
    previous.verify();assert not OUT.exists()
    controls=controlled()
    # Remove added diagnostic keys: every old spectral output must stay exact.
    assert strip_new(controls)==load(previous.OUT/'controlled66.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        cases=CASES,parameters=PARAMS,reference_protocol_sha256=sha(previous.OUT/'protocol.json'),
        reference_hashes={c:sha(previous.OUT/c/'observations.jsonl') for c in CASES},
        design='Add per-peak SNR and matching to unchanged66 FFT/top5 candidate sets. '
               'Original dominant/half-stable spectra remain bit-for-bit equivalent. '
               'Run same3.2s actual RAW through unchanged66 run function in isolated '
               'namespace with new tile diagnostic only; no hidden cache inputs.',
        prior_art=['https://arxiv.org/html/2604.18289v1','https://github.com/ros-event-camera/frequency_cam'],
        reuse='Original rotor_sensing top5 spectral hypotheses,66 polarity periodogram, '
              'same half-window bin tolerance andSNR8. Routine multi-peak consistency, '
              'not a new algorithm/true RPM claim.',
        scope='Follow-up after observing66 mixed-signal failure. Controlled development '
              'evidence, not blind validation; same recordings, no new future forecast.'))
    save(OUT/'controlled67.json',controls);print('67 multi-peak protocol frozen, original controls exact66',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for c,h in p['reference_hashes'].items():assert h==sha(previous.OUT/c/'observations.jsonl')
    previous.verify();return p


def run(case):
    verify()
    namespace=dict(previous.__dict__);namespace.update(OUT=OUT,verify=verify,tile_measurement=tile_measurement)
    exec(compile(inspect.getsource(previous.run),'<67 unchanged66 RAW shadow loop>','exec'),namespace)
    namespace['run'](case)


def audit(case):
    verify();a=rows(OUT/case/'observations.jsonl');b=rows(previous.OUT/case/'observations.jsonl')
    assert strip_new(a)==b,(case,'original66 spectral outputs changed')
    data=[]
    for method in ['raw','aligned']:
        observed=[r['methods'][method] for r in a]
        data.append(dict(case=case,method=method,parent=len(a),
             original_spectral_frames=sum(any(q['valid'] for q in r['tiles']) for r in observed),
             dominant_half_stable_frames=sum(any(q['stable'] for q in r['tiles']) for r in observed),
             multi_half_stable_frames=sum(any(q['multi_half_stable'] for q in r['tiles']) for r in observed),
             multi_half_stable_tiles=sum(sum(q['multi_half_stable'] for q in r['tiles']) for r in observed),
             matched_frequency_hypotheses=sum(sum(len(q['matched_frequencies']) for q in r['tiles']) for r in observed)))
    save(OUT/case/'regression67.json',dict(original66_current_spectra_velocity_and_sampling_exact=True,table=data))
    print(data,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','audit']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','audit']:globals()[a.action](a.case)
    else:freeze()
