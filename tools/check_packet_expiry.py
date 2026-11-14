"""Check causal discard and SDK-update ordering, without RAW or wall sleeps."""
from pathlib import Path
import ast,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_stream import packet_expired
cap=.16;checked=0
for start in [0.,22.,110.]:
    for dt in [.02,.4,4.]:
        for epoch in [0.,10000.]:
            t=start+dt
            for age in [.0,.15,.159,.161,.4,3.]:
                now=epoch+dt+age
                expired=packet_expired(t,start,epoch,now,cap)
                if expired:
                    for transit in [0.,.001,.02,.5,3.]:
                        receiver_now=start+(now+transit-epoch)
                        assert receiver_now-t>cap+1e-8
                        checked+=1
assert not packet_expired(1.,0.,None,0.,cap)
assert not packet_expired(1.,0.,0.,5.,None)
tree=ast.parse((ROOT/'core/live_process.py').read_text('utf-8'))
producer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_producer')
lines={}
for n in ast.walk(producer):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='process_events':lines['native_noise']=n.lineno
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='packet_expired':lines['expiry']=n.lineno
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='clean' for t in n.targets):lines['copy']=n.lineno
assert lines['native_noise']<lines['expiry']<lines['copy']
base=ROOT.parent/'output/live_reader_index_20261026'
default=json.loads((base/'default.json').read_text('utf-8'))
disabled=json.loads((base/'disabled.json').read_text('utf-8'))
canonical=lambda d:[{k:r[k] for k in ['t','n','sha256']} for r in d['rows']]
assert canonical(default)==canonical(disabled)
report=dict(expired_producer_packets_rejected_by_original_consumer_for_nonnegative_transit=checked,
    native_SDK_update_before_expiry_and_copy_after_expiry=True,
    index_disabled_canonical_native_slices_exact=len(default['rows']),
    native_events=default['native_events'],index_mode_timing_gain_proven=False,
    scope='Analytic nonnegative-transit property plus native205-slice decoder equivalence; no forecast-accuracy claim')
(base/'equivalence_and_expiry.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
