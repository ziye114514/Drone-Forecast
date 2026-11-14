"""Bounded CUDA component benchmark against frozen same-event CPU outputs."""
from pathlib import Path
import sys,json,time,os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
import numpy as np,torch,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from predictive_harmonics_cuda import predictive_harmonics_cuda


def main(output,sparse=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    assert torch.cuda.is_available();source=out.parent;results=[]
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['predictive_harmonics.py','predictive_harmonics_cuda.py','tools/bench_predictive_harmonics_cuda.py']:
        p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    # Initialization is explicit and excluded from steady component timings.
    start=time.perf_counter();torch.ones(1,device='cuda').sum().item();startup=(time.perf_counter()-start)*1000
    for name in ['clear','partial','side']:
        p=np.load(source/'aligned'/name/'aligned_events.npz');e=p['events'];t=float(p['t'])
        expected=cv2.imread(str(source/'aligned'/name/'predictive_mask.png'),0)
        pixel_mask=np.random.default_rng(141000).integers(0,4,size=(720,1280),dtype=np.uint8)==0 if sparse else None
        if sparse:expected=np.uint8((expected>0)&pixel_mask)*255
        records=[];first=None
        for repeat in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            result=predictive_harmonics_cuda(e,t,pixel_mask=pixel_mask)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            records.append(elapsed)
            if first is None:first=result
            else:assert np.array_equal(first['mask'],result['mask'])
        difference=int(np.count_nonzero(first['mask']!=expected))
        row=dict(name=name,aligned_native_events=len(e),CPU_accepted=int(np.count_nonzero(expected)),
            fixed_pixel_seed=141000 if sparse else None,fixed_spatial_fraction=.25 if sparse else 1.,
            temporal_events_at_selected_pixels_not_downsampled=True,
            CUDA_accepted=first['accepted_pixels'],mask_different_pixels=difference,
            timing_ms=records,steady_mean_ms=float(np.mean(records[1:])),
            component_with_CPU_grouping_transfers_and_CUDA_sync=True,
            no_full_system_or_RAW_decode_NN_geometry_FPS_claim=True)
        target=out/name;target.mkdir();cv2.imwrite(str(target/'mask.png'),first['mask'])
        np.savez_compressed(target/'evidence.npz',frequency_hz=first['frequency_hz'],evidence=first['evidence'],spatial_phase=first['spatial_phase'])
        results.append(row);print(json.dumps(row),flush=True)
        assert difference<=2,'Investigate numerical disagreement before connecting the CUDA branch'
    (out/'summary.json').write_text(json.dumps(dict(device=torch.cuda.get_device_name(),torch_version=torch.__version__,
        CUDA_initialization_ms=startup,records=results,cold_import_and_RAW_not_timed=True,
        original_CPU_control_preserved=True),indent=2),'utf-8')


if __name__=='__main__':main(sys.argv[1],'--sparse' in sys.argv)
