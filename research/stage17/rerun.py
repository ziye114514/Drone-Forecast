"""Run the fixed four-case protocol in a NEW folder, preserving frozen sources."""
import argparse,hashlib,json,importlib.util
from pathlib import Path
import study


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();target=Path(a.output).resolve();study.OUT=target
    if a.resume:
        if not (target/'protocol.json').is_file():p.error('resume requires an existing frozen protocol')
        study.verify()
    else:
        target.mkdir(parents=True,exist_ok=False);study.freeze()
    study.save(target/'runner.json',dict(wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        argv=vars(a),note='No changes to frozen detector/parameters. Handles known final console serialization issue.'))
    for case in study.CASES:
        if (target/case/'completion.json').is_file():
            if not a.resume:raise RuntimeError('Unexpected completed case')
            print('Keep completed '+case,flush=True);continue
        if (target/case/'observations.jsonl').exists():raise RuntimeError('Incomplete case found; use another new output folder')
        study.run(case)
    try:study.analyze()
    except TypeError as error:
        if str(error)!='Object of type int64 is not JSON serializable':raise
        # Frozen study writes and validates analysis BEFORE this console-only error.
        data=study.load(target/'analysis.json')
        if len(data['table'])!=len(study.CASES)*2:raise RuntimeError('Incomplete analysis') from error
        print('Analysis saved; skipped frozen console-only numpy int64 print failure.',flush=True)
    check=[]
    for case,cfg in study.CASES.items():
        rows=list(map(json.loads,(target/case/'observations.jsonl').read_text().splitlines()))
        completion=study.load(target/case/'completion.json')
        assert len(rows)==completion['published']==round(cfg['duration']/.02)
        assert completion['model_sources_unchanged']
        check.append(dict(case=case,frames=len(rows),complete=True))
    study.verify();study.save(target/'delivery_checks.json',dict(cases=check,frozen_sources_match=True,json_complete=True,
        runner_note='Known frozen console-only int64 exception handled; inference and analysis complete.'))
    spec=importlib.util.spec_from_file_location('stage17_report',Path(__file__).with_name('report.py'))
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    report.main()


if __name__=='__main__':main()
