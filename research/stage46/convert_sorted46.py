"""Official ORT graph sort after converter appends input casts after consumers."""
from collections import Counter
import datetime,time
from backend46 import *
def main():
    p=verify();assert not MODEL.exists();supplement=OUT/'topology_supplement.json';assert not supplement.exists()
    save(supplement,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),reason='Installed ORT float16 converter with keep_io_types appended input cast after its first Conv consumer. Required ONNX checker rejected unsorted graph before original artifact save. Retain an unsorted copy and error, use same official ORT OnnxModel.topological_sort. Assert identical node multiset/initializers/IO before and after; only node order changes. No weights/edges/precision tuning.'))
    import onnx
    from onnxruntime.transformers import float16
    from onnxruntime.transformers.onnx_model import OnnxModel
    began=time.perf_counter();model=float16.convert_float_to_float16(onnx.load(common44.OUT/'weights/best.onnx'),keep_io_types=True,min_positive_val=1e-7,max_finite_val=1e4)
    unsorted=OUT/'weights/best_fp16_before_sort.onnx';unsorted.parent.mkdir(exist_ok=True);onnx.save(model,unsorted)
    try:onnx.checker.check_model(model);error=None
    except onnx.checker.ValidationError as e:error=str(e)
    assert error and 'topologically sorted' in error
    before_nodes=Counter(n.SerializeToString() for n in model.graph.node);before_weights=[v.SerializeToString() for v in model.graph.initializer];before_io=[v.SerializeToString() for v in list(model.graph.input)+list(model.graph.output)]
    wrapper=OnnxModel(model);wrapper.topological_sort(is_deterministic=True);assert Counter(n.SerializeToString() for n in model.graph.node)==before_nodes;assert [v.SerializeToString() for v in model.graph.initializer]==before_weights;assert [v.SerializeToString() for v in list(model.graph.input)+list(model.graph.output)]==before_io
    onnx.checker.check_model(model);onnx.save(model,MODEL)
    save(OUT/'conversion.json',dict(sha256=sha(MODEL),bytes=MODEL.stat().st_size,original_sha256=p['original_onnx_sha256'],wall_s=time.perf_counter()-began,initializer_types=dict(Counter(str(r.data_type) for r in model.graph.initializer)),input_element_type=model.graph.input[0].type.tensor_type.elem_type,output_element_type=model.graph.output[0].type.tensor_type.elem_type,unsorted_copy_sha256=sha(unsorted),unsorted_checker_error=error,sort_node_multiset_weights_edges_io_unchanged=True,helper_sha256=sha(Path(__file__)),sources_unchanged=verify()['source_hashes']==p['source_hashes']))
    print('FP16 node-order repair verified; arithmetic unchanged',flush=True)
if __name__=='__main__':main()
