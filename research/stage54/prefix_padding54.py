"""Diagnose opaque structured padding without changing the frozen predictor."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import study54 as current
from packets54 import dtype_spec
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=current.OUT
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze():
    current.verify();p=OUT/'prefix_validation_supplement.json';assert not p.exists()
    current.save(p,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),
        reason='Original frozen prefix check failed at108.04s on event_packet_sha256. All54 run audits still have exact53 observations/forecasts; per-transfer parent/worker payload SHA passed. Diagnose structured padding, keep original source and failed validation intact.',
        design='Two independent native RAW readers at identical first40ms side cutoffs, compare every named field/order and changed byte offsets. Explicit padding-only perturbation control. Revised full prefix compares all scientific fields and byte count, excludes only opaque cross-decode payload SHA. Within-transfer checksum never relaxed.'))
    a=current.rows(OUT/'prefix_side6/observations_forecasts.jsonl');b=current.rows(OUT/'side6/observations_forecasts.jsonl')
    differences=[dict(t=x['t'],prefix_sha=x['event_packet_sha256'],full_sha=y['event_packet_sha256']) for x,y in zip(a,b) if x['event_packet_sha256']!=y['event_packet_sha256']]
    current.save(OUT/'failed_prefix_byte_sha_v1.json',dict(original_source_unchanged=True,original_failure_t=108.04,field='event_packet_sha256',cross_decode_sha_differences=differences,not_a_prediction_tolerance_change=True))
def verify():
    assert current.load(OUT/'prefix_validation_supplement.json')['helper_sha256']==sha(Path(__file__));current.verify()
def diagnose():
    verify();cfg=current.CASES['side6'];start=cfg['start'];begin=start-.1
    readers=[open_raw(cfg['raw']),open_raw(cfg['raw'])];streams=[raw_chunks(r,begin,.14,20000) for r in readers];records=[]
    for (ta,a),(tb,b) in zip(*streams):
        assert abs(ta-tb)<1e-8 and a.dtype==b.dtype and len(a)==len(b)
        for field in a.dtype.names:np.testing.assert_array_equal(a[field],b[field])
        if ta<=start+1e-8:continue
        dtype=a.dtype;used=np.zeros(dtype.itemsize,bool)
        for name in dtype.names:
            dt,offset=dtype.fields[name][:2];used[offset:offset+dt.itemsize]=True
        unused=np.flatnonzero(~used);aa=a.tobytes();bb=b.tobytes();x=np.frombuffer(aa,np.uint8).reshape(-1,dtype.itemsize);y=np.frombuffer(bb,np.uint8).reshape(-1,dtype.itemsize)
        diff=np.flatnonzero(np.any(x!=y,axis=0)).tolist() if len(x) else []
        assert all(not used[k] for k in diff),('actual named bytes differ',ta,diff)
        # Demonstrate that opaque padding alone can change whole-struct SHA.
        changed=x.copy()
        if len(changed) and len(unused):changed[:,unused]^=np.uint8(0xA5)
        altered=np.frombuffer(changed.tobytes(),dtype=dtype)
        for field in dtype.names:np.testing.assert_array_equal(a[field],altered[field])
        control=bool(len(changed) and len(unused) and hashlib.sha256(aa).hexdigest()!=hashlib.sha256(changed.tobytes()).hexdigest())
        records.append(dict(t=ta,count=len(a),dtype=dtype_spec(dtype),unused_byte_offsets=unused.tolist(),actual_changed_byte_offsets=diff,
            actual_named_fields_and_order_exact=True,actual_whole_struct_sha_equal=hashlib.sha256(aa).hexdigest()==hashlib.sha256(bb).hexdigest(),padding_only_perturbation_changes_sha=control))
    assert len(records)==2 and all(r['padding_only_perturbation_changes_sha'] for r in records)
    current.save(OUT/'structured_padding_diagnostic.json',dict(records=records,
        conclusion='Native named event values/order agree; padding bytes are outside event semantics. Whole-struct byte SHA is mandatory within one transfer but unsuitable as a required cross-decode prefix invariant. Original full-run bytes were not archived, so the diagnostic does not reconstruct those opaque bytes.'))
    print(json.dumps(records,ensure_ascii=False),flush=True)
def prefix():
    verify();diagnostic=current.load(OUT/'structured_padding_diagnostic.json');assert all(r['actual_named_fields_and_order_exact'] for r in diagnostic['records'])
    a=current.rows(OUT/'prefix_side6/observations_forecasts.jsonl');b=current.rows(OUT/'side6/observations_forecasts.jsonl');assert len(a)==225
    keys=['t','event_count','filtered_event_count','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','frontends','lifecycle','event_packet_nbytes']
    for x,y in zip(a,b):
        for key in keys:assert x[key]==y[key],(x['t'],key)
        assert len(x['event_packet_sha256'])==len(y['event_packet_sha256'])==64
    aa=current.rows(OUT/'prefix_side6/rotor_ticks.jsonl');bb=current.rows(OUT/'side6/rotor_ticks.jsonl');assert len(aa)==450
    for x,y in zip(aa,bb):
        for key in ['t','event_count','observation','sensing','slot_audit']:assert x[key]==y[key],(x['t'],key)
    current.save(OUT/'raw_prefix_checks.json',dict(parent_ticks=225,native10_ticks=450,actual_RAW_decode=True,complete_data_forecasts_lifecycle_exact=True,
        per_transfer_byte_sha_verified=True,cross_decode_opaque_padding_sha_not_required=True,named_fields_decoder_diagnostic=diagnostic,
        original_byte_sha_failure_preserved='failed_prefix_byte_sha_v1.json',validation_supplement='prefix_validation_supplement.json'))
    print('225/450 actual RAW semantic/model prefix exact; per-transfer byte integrity retained; original opaque-padding failure preserved',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','diagnose','prefix']);globals()[p.parse_args().action]()
