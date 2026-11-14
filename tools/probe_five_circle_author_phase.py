"""Reuse the existing local author-core phase observer with current five-circle outputs.

Geometry/body inputs are frozen causal outputs, not manual or future labels.
This is an offline module pilot, not an end-to-end timing or mechanical-RPM test.
"""
from pathlib import Path
import sys,json,hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/helix_reference'),str(ROOT/'research/stage79'),str(ROOT/'research/stage141')]
from live_decode import native_batches
from phase_state import RotorPhaseObserver
from reference_api import AuthorTracker


class PhaseOnlyTracker(AuthorTracker):
    def feed(self,x,y,t,p,solve_pose=False):
        return super().feed(x,y,t,p,solve_pose=False)


def run(source,output,qualified_arcs=False,serial=True,quarter=False,phase_only=False):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text('utf-8'))
    rows=list(map(json.loads,(source/'predictions.jsonl').read_text('utf-8').splitlines()))
    observer=RotorPhaseObserver(allow_current_reanchor=True,allow_partial_reanchor=True)
    if serial:
        observer.pool.shutdown(wait=True);observer.pool=ThreadPoolExecutor(max_workers=1)
    if phase_only:
        observer.factory=lambda rpm,blades:PhaseOnlyTracker(rpm,blades,dll_path=observer.dll,
            canvas=(256,256,128,128,80),signed_coordinates=True)
    if quarter:
        from metavision_sdk_core import RoiMaskAlgorithm
        mask=np.random.default_rng(141000).integers(0,4,size=(720,1280),dtype=np.uint8)==0
        pixel_filter=RoiMaskAlgorithm(mask.astype(np.float64));pixel_buffer=pixel_filter.get_empty_output_buffer()
    config=observer.protocol();config.update(source=str(source),saved_upstream_causal_geometry=True,
        labels_read=False,force_prediction_changed=False,end_to_end_FPS=False,
        qualified_current_arc_initialization=qualified_arcs,
        serial_author_core=serial,
        four_independent_rotor_threads=not serial,
        phase_only_without_author_pose_solver=phase_only,
        quarter_fixed_native_pixels=quarter,temporal_event_sampling=False,
        author_core_unmodified=True,partial_shapes_are_initialization_hypotheses=True)
    (out/'protocol.json').write_text(json.dumps(config,indent=2),'utf-8');results=[];native_count=selected_count=0
    try:
        with (out/'observations.jsonl').open('w',encoding='utf-8') as stream:
            for r,(t,e) in zip(rows,native_batches(protocol['raw'],protocol['actual_iterator_start'],protocol['actual_duration'],False,protocol['native_step_us'])):
                assert abs(t-r['t'])<1e-8
                native_count+=len(e)
                if quarter:
                    pixel_filter.process_events(e,pixel_buffer);e=pixel_buffer.numpy()
                selected_count+=len(e)
                rotors=r['geometry'].get('rotors',[])
                adapted=False
                if (qualified_arcs and r['structure']['mode']=='four' and len(rotors)==4 and
                        all(z.get('mode')=='ellipse' or z.get('arc_center_qualified') or z.get('curvature_validated') for z in rotors)
                        and sum(not z.get('shared_arc_recovered',False) for z in rotors)>=2):
                    # The API's eligibility flag encodes "complete ellipse".
                    # Pass the qualified fitted shape as an EXPLICIT initial
                    # hypothesis. No missing centre or frequency is invented.
                    rotors=[dict(z,mode='ellipse',inferred_shape=False) for z in rotors]
                    adapted=True
                value=observer.step(e,t,r['body'],dict(count=rotors),dict(count=dict(current=True)),dict(matched_body=r['coefficient']))
                value['adapter_used_current_qualified_arc_shapes']=adapted
                results.append(value);stream.write(json.dumps(value,default=lambda x:x.tolist() if hasattr(x,'tolist') else x)+'\n');stream.flush()
    finally:observer.close()
    result=dict(source_ticks=len(results),initialized=sum(r.get('initialized_now',False) for r in results),
        native_event_count=native_count,phase_input_event_count=selected_count,
        full_supported_phase_ticks=sum(r.get('valid',False) for r in results),
        supported_candidate_counts=dict(Counter(r.get('supported_candidates',0) for r in results)),
        modes=dict(Counter(r['observation_mode'] for r in results)),
        mean_module_ms=float(np.mean([r['processing_ms'] for r in results])),
        p95_module_ms=float(np.percentile([r['processing_ms'] for r in results],95)),
        reasons=dict(Counter(r['reason'] for r in results)),
        no_mechanical_RPM_truth=True,no_forecast_accuracy_claim=True,production_default_changed=False,
        author_DLL_redistributed=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1],sys.argv[2],qualified_arcs='--qualified-arc-init' in sys.argv[3:],serial='--parallel-core' not in sys.argv[3:],quarter='--quarter-phase-events' in sys.argv[3:],phase_only='--phase-only' in sys.argv[3:])
