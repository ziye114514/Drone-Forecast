"""Validation-only adapter: equivalent4D/6D paths differ at floating precision."""
import ast
import study107 as original


def main():
    original.verify();path=original.Path(original.__file__);source=path.read_text('utf-8')
    before="for r in data:assert r['output']['coefficients'][NAMES[0]]==r['output']['coefficients'][NAMES[2]]"
    after="for r in data: assert_coefficients_close(r['output']['coefficients'][NAMES[0]],r['output']['coefficients'][NAMES[2]])"
    assert source.count(before)==1
    def close(a,b):
        original.np.testing.assert_allclose(original.np.r_[a['position'],a['velocity']],original.np.r_[b['position'],b['velocity']],rtol=0.,atol=1e-8)
    namespace=dict(original.__dict__,assert_coefficients_close=close)
    node=next(n for n in ast.parse(source.replace(before,after)).body if isinstance(n,ast.FunctionDef) and n.name=='synthetic')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    original.save(original.OUT/'synthetic_validation_adapter107.json',dict(
        original_source_sha256=original.sha(path),adapter_sha256=original.sha(__file__),
        reason='Initial no-geometry control compared4D vs6D dict floats exactly. Separate reproduction maximum difference4.55e-13. Replace only this validator with1e-8 absolute numerical tolerance; all deterministic same-dimensional prefixes still exact. No inference/scoring/parameter changes.'))
    namespace['synthetic']()
    p=original.OUT/'synthetic107.json';q=original.load(p);q['no_geometry_fixed_control_exact']=False;q['no_geometry_fixed_control_numerically_equivalent']=True;q['equivalence_absolute_tolerance']=1e-8;original.save(p,q)


if __name__=='__main__':main()
