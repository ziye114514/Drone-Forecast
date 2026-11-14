from slots33 import ReplaceStale,RotorSensing,PaperConfig,geometry_record


def observation(xs):
    return dict(rotors=[dict(cx=float(x),cy=100.,major=20.,minor=8.,angle=0.,ratio=.4,geometry_quality=.9,activity=100.,mode='ellipse') for x in xs])


def run():
    a=RotorSensing(PaperConfig());b=ReplaceStale(PaperConfig())
    for t in [.01,.02,.03,.04]:
        o=observation([0,50,100,150]);assert geometry_record(a.step(o,[],t))==geometry_record(b.step(o,[],t))
    b.reference_total=123.;o=observation([0,50,100,500])
    for t in [.05,.08,.11]:
        a.step(o,[],t);b.step(o,[],t);assert not b.audit['retired_ids']
    old_ids={r.id for r in b.tracks};q=b.step(o,[],.12)
    assert b.audit['retired_ids']==[3] and b.calibration_epoch==1 and b.reference_total is None
    fresh=[r for r in q['rotors'] if r['rotor_id'] not in old_ids]
    assert len(fresh)==1 and fresh[0]['frequency_valid'] is False and fresh[0]['reference_hz'] is None
    assert q['four_visible']
    # An unmatched candidate cannot displace a fresh matched ID.
    assert {r.id for r in b.tracks if r.id in {0,1,2}}=={0,1,2}
    q=b.step(dict(rotors=[]),[],.6);assert not q['rotors']
    return dict(unchanged_when_matches_available=True,grace_period_respected=True,stale_slot_replaced_only_for_current_candidate=True,
        new_identity_frequency_empty=True,thrust_reference_reset=True,matched_id_preserved=True,empty_input_expires=True)
