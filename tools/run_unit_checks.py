"""Capture real tests for a new result directory; reuse matching passed evidence."""
from pathlib import Path
import argparse,ast,hashlib,importlib.metadata,json,re,subprocess,sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    tests=sorted((root/'tests').glob('test_*.py'))
    hashes={str(p.relative_to(root.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tests}
    names=[n.name for path in tests for n in ast.walk(ast.parse(path.read_text(encoding='utf-8')))
           if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
    result_path=out/'unit_test_result.json'
    old=json.loads(result_path.read_text()) if result_path.exists() else None
    # Paths in early session evidence are relative to the same workspace.
    if old and old.get('failed')==0 and old.get('passed')==len(names) and old.get('test_file_sha256')==hashes:
        print('Reused matching actual unit-test evidence: '+str(len(names))+' passed',flush=True)
    else:
        command=[sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-v']
        process=subprocess.run(command,text=True,capture_output=True,encoding='utf-8',errors='replace')
        text=process.stdout+process.stderr;(out/'unit_test_output.txt').write_text(text,encoding='utf-8')
        match=re.search(r'Ran (\d+) tests? in ([\d.]+)s',text)
        passed=bool(process.returncode==0 and match and int(match[1])==len(names))
        result=dict(command=command,passed=int(match[1]) if passed else 0,failed=0 if passed else 1,
                    elapsed_test_seconds=float(match[2]) if match else None,
                    record_type='Captured real unittest subprocess output.',test_names=names,test_file_sha256=hashes)
        result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(text,flush=True)
        if not passed:raise SystemExit('Unit checks failed')
    environment={name:importlib.metadata.version(name) for name in ['numpy','scipy','opencv-python-headless','pillow','matplotlib']}
    (out/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')


if __name__=='__main__':main()
