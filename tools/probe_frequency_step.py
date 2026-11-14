"""Step response of the causal frequency selector, and why the first version latches.

Round 5's selector picks the hypothesis closest to the running median. That rule
has a fixed point: as long as the old value stays in the candidate set it keeps
re-selecting itself, so the reported spread looks small partly because the value
is latched, not because it is tracked. This probe measures that and compares a
consensus-switch rule:

  A  median-closest (round 5)     keep the option nearest the running median
  B  consensus switch             keep the current value until the reported value
                                  has deviated in the same direction for K frames

Cases are synthetic sequences of reported values, so this measures the selector
alone; a detector that never reports the new value cannot be rescued by either
rule. No rotor identity or RPM truth is claimed.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
from rotor_support_front import ConsensusFrequencySelector,select_frequency

OUT=ROOT.parent/'output/rotor_support_20261101'
FRAMES=30
K=3
DEVIATION=0.10


def rule_a(reported_series,candidate_series,history):
    assigned=[];changes=0
    history=list(history)
    for reported,candidates in zip(reported_series,candidate_series):
        chosen,changed=select_frequency(reported,candidates,history)
        history.append(chosen);assigned.append(chosen);changes+=int(changed)
    return assigned,changes


def rule_b(reported_series,candidate_series,history):
    follower=ConsensusFrequencySelector(deviation=DEVIATION,k=K)
    if history:follower.current=float(history[-1])
    assigned=[];changes=0;previous=None
    for reported,_ in zip(reported_series,candidate_series):
        value=follower.step(reported)
        if previous is not None and abs(np.log(value/previous))>1e-9:changes+=1
        previous=value;assigned.append(value)
    return assigned,changes


def frames_to_follow(assigned,target,tolerance=.02):
    for index,value in enumerate(assigned):
        if value and abs(np.log(value/target))<=np.log(1+tolerance):return index
    return None


def cases():
    base=270.
    series=[]
    for label,step_to in [('step_1p11',300.),('step_1p48',400.),('step_0p50',135.)]:
        reported=[base]*10+[step_to]*(FRAMES-10)
        candidates=[[(base,1),(step_to,1)] for _ in range(FRAMES)]
        series.append(dict(name=label,reported=reported,candidates=candidates,target=step_to))
    ramp=[base+(330.-base)*min(max((i-5)/20.,0.),1.) for i in range(FRAMES)]
    series.append(dict(name='ramp_270_330',reported=ramp,
        candidates=[[(value,1),(base,1)] for value in ramp],target=330.))
    rng=np.random.default_rng(0)
    noisy=[base*(1+rng.uniform(-.08,.08)) for _ in range(FRAMES)]
    series.append(dict(name='noise_only',reported=noisy,
        candidates=[[(value,1),(base,1)] for value in noisy],target=None))
    only_new=[base]*10+[300.]*(FRAMES-10)
    series.append(dict(name='step_1p11_new_only',reported=only_new,
        candidates=[[(value,1)] for value in only_new],target=300.))
    return series


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(frames=FRAMES,consensus_k=K,deviation=DEVIATION,
        assumption='the detector reports the true value each frame; this isolates the selector',
        cases={},latch_note='rule A re-selects the past value whenever it remains among the candidates')
    for case in cases():
        past=[case['reported'][0]]*2
        a,changes_a=rule_a(case['reported'],case['candidates'],past)
        b,changes_b=rule_b(case['reported'],case['candidates'],past)
        entry=dict(reported=[round(v,1) for v in case['reported']],
            assigned_a=[None if v is None else round(v,1) for v in a],
            assigned_b=[round(v,1) for v in b],changes_a=changes_a,changes_b=changes_b,
            frames_to_follow_a=None if case['target'] is None else frames_to_follow(a,case['target']),
            frames_to_follow_b=None if case['target'] is None else frames_to_follow(b,case['target']))
        result['cases'][case['name']]=entry
        print(f"{case['name']:20s} A follow {entry['frames_to_follow_a']} changes {changes_a} | "
              f"B follow {entry['frames_to_follow_b']} changes {changes_b}",flush=True)
    (OUT/'frequency_step.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'frequency_step.json')


if __name__=='__main__':
    main()
