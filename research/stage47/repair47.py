"""Frozen transport-only PIPE correction; preserve empty initial failed folder."""
import argparse,datetime,importlib.util
import study47 as s
def module():
    path=s.ROOT/'run_shared_body_forecast_v2.py';spec=importlib.util.spec_from_file_location('usable_shared47',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def freeze():
    s.verify();supplement=s.OUT/'pipe_proxy_supplement.json';assert not supplement.exists();m=module()
    for body in [False,True]:
        ns,memory,workers=m.build(32,body,True);assert not memory and not workers
        assert all(hasattr(ns['subprocess'],x) for x in ['Popen','PIPE','TimeoutExpired'])
    s.save(supplement,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes={'run_shared_body_forecast_v2.py':s.sha(s.ROOT/'run_shared_body_forecast_v2.py'),'research/stage47/repair47.py':s.sha(__import__('pathlib').Path(__file__))},reason='Original47 proxy omitted subprocess.PIPE; first case stopped before child creation or any image inference. New v2 wrapper adds only PIPE constant and validates required proxy attributes, preserving old frozen files. Preserve initial empty folder, then execute same frozen five cases.'))
    folder=s.OUT/'train3_full32';target=s.OUT/'failed_train3_full32_pipe_proxy_v1';assert folder.resolve().is_relative_to(s.OUT.resolve()) and target.resolve().is_relative_to(s.OUT.resolve()) and not target.exists();folder.rename(target)
def run():
    s.verify();p=s.load(s.OUT/'pipe_proxy_supplement.json')
    for name,value in p['source_hashes'].items():assert s.sha(s.ROOT/name)==value
    m=module()
    for c,cfg in s.CASES.items():m.run(output=s.OUT/c,verify_frames=True,**cfg);print('Shared-memory v2 case completed',c,flush=True)
    s.verify()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);globals()[p.parse_args().action]()
