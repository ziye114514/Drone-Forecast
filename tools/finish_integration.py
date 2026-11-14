"""Finish a frozen regression suite sequentially, including read-only RAW media."""
from pathlib import Path
import argparse,subprocess,sys


def main():
    root=Path(__file__).resolve().parents[1];work=root.parent
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=work/'output/raw_only_project_integrated')
    parser.add_argument('--skip-media',action='store_true');args=parser.parse_args();out=args.output.resolve()
    def execute(name,*arguments):
        subprocess.run([sys.executable,str(root/'tools'/name),*map(str,arguments)],check=True)
    execute('run_unit_checks.py','--output',out)
    for sid in (6,2,3,4):
        execute('evaluate.py','--run',out/'runs'/f'train{sid}',
                '--labels',work/'output/raw_only_project/evaluation_data'/f'train{sid}.json')
    execute('geometry_examples.py','--output',out/'visuals')
    if not args.skip_media:
        for name,seq,start,duration in [('train6_close','train6',109,5),('train3','train3',1.38,8),('train6','train6',None,8)]:
            command=['--run',out/'runs'/seq,'--output',out/'visuals'/name,'--duration',duration,
                     '--baseline',work/'output/raw_only_project/runs'/seq]
            if start is not None:command+=['--start',start]
            execute('render_results.py',*command)
    execute('check_raw_prefix.py','--output',out)
    execute('benchmark_integrated.py','--output',out)
    execute('build_integration_report.py','--output',out)
    execute('validate_integrated.py','--output',out)
    execute('package_integrated.py','--output',out)
    print('DELIVERY COMPLETE: '+str(out/'查看整合结果.html'),flush=True)


if __name__=='__main__':main()
