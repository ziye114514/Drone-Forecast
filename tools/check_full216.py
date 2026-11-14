from pathlib import Path
import json
BASE=Path(__file__).resolve().parents[2]/'output'
a=[json.loads(s) for s in (BASE/'arc215_20261108/recording13/regional_adaptive.jsonl').read_text('utf-8').splitlines()]
b=[json.loads(s) for s in (BASE/'uncertainty216_20261108/recording13/regional_adaptive.jsonl').read_text('utf-8').splitlines()]
errors=[];different=[]
def compare(a,b,path=''):
    if isinstance(a,dict):
        if set(a)!=set(b):different.append(path+':keys');return
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        if len(a)!=len(b):different.append(path+':length');return
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)) and not isinstance(b,bool):errors.append((abs(a-b),path))
    elif a!=b:different.append(path)
assert len(a)==len(b)==181
for i,(x,y) in enumerate(zip(a,b)):
    for k in ['coefficient','valid','force','physical','body','geometry','structure']:compare(x[k],y[k],f'{i}.{k}')
mx=max(errors);result=dict(frames=len(a),max_numeric_difference=mx[0],max_numeric_path=mx[1],
    non_numeric_differences=different,acceptance_and_forecasts_preserved=not different and mx[0]<1e-6)
(BASE/'uncertainty216_20261108/recording13/full_parity.json').write_text(json.dumps(result,indent=2),'utf-8')
print(result);assert result['acceptance_and_forecasts_preserved']
