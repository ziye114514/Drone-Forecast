"""Restore the archived native20 startup schedule without altering frozen core."""
from common41 import *
def main():
    verify();assert not (OUT/'inputs.json').exists();supplement=OUT/'startup_supplement.json';assert not supplement.exists()
    save(supplement,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),audit_sha256=sha(OUT/'baseline_failure_audit.json'),
        reason='Preparation omitted the geometry update exactly at segment start used by frozen21/24; at22.04 only2 rather than3 confirmation calls. Audit with/without10ms noise warm both failed identically. Restore original startup: one geometry-only20ms call at segment start, predictions strictly afterwards; negative0–80ms emits archived minimal abstention records.',
        scope='No model weights, thresholds, samples or target scoring changed. Preserve failed_clear6_initialization. Geometry exact regression remains mandatory.'))
    (OUT/'clear6').rename(OUT/'failed_clear6_initialization')
    source=Path(__file__).with_name('prepare41.py').read_text(encoding='utf-8')
    a="reader.seek_time(round(cfg['start']*1e6)-10000);warm=reader.load_delta_t(10000);alg.process_events(warm,buf)"
    b="reader.seek_time(round(cfg['start']*1e6)-20000);warm=reader.load_delta_t(20000);alg.process_events(warm,buf);front.step(warm,cfg['start'])"
    assert source.count(a)==1;source=source.replace(a,b)
    a="o=front.step(e,t)['observation'];q,_=model.step";b="o=front.step(e,t)['observation'] if cfg['start'] or t>=.1-1e-8 else dict(t=t,detected=False,rotors=[]);q,_=model.step"
    assert source.count(a)==1;source=source.replace(a,b)
    namespace={'__name__':'prepare41_bridge','__file__':str(Path(__file__).with_name('prepare41.py'))};exec(compile(source,namespace['__file__'],'exec'),namespace);namespace['main']()
    prep=load(OUT/'preparation.json');prep['startup_supplement_sha256']=sha(supplement);save(OUT/'preparation.json',prep)
if __name__=='__main__':main()
