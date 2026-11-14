"""Prepare two browser upload batches for the existing nested GitHub layout."""
from pathlib import Path
import hashlib,json,re,shutil
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/github_release_update_20261028';PUBLIC=BASE/'event-drone-forecast'

def main():
    files=sorted(p for p in PUBLIC.rglob('*') if p.is_file());groups={}
    for p in files:
        rel=p.relative_to(PUBLIC);key=rel.parts[0] if len(rel.parts)>1 else '__ROOT__'
        groups.setdefault(key,[]).append(p)
    batches=[[],[]]
    order=['__ROOT__','src','native','configs','LICENSES','.github','tools','tests','docs','examples']
    assert set(groups).issubset(order)
    for name in order:
        part=groups.get(name,[]);i=0 if len(batches[0])+len(part)<=80 else 1
        batches[i].extend(part)
    counts=[]
    for i,part in enumerate(batches,1):
        assert 0<len(part)<=100
        target=BASE/('上传第'+str(i)+'批')/'event-drone-forecast'
        if target.exists():raise FileExistsError(target)
        for p in part:
            q=target/p.relative_to(PUBLIC);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
            assert q.read_bytes()==p.read_bytes()
        counts.append(dict(batch=i,files=len(part),drag_this_folder=str(target)))
    assert sum(c['files'] for c in counts)==len(files)
    content=(BASE/'ROOT_README_FOR_NESTED.md').read_text('utf-8')
    for match in re.finditer(r'!?\[[^\]]*\]\(([^\s)]+)\)',content):
        value=match.group(1)
        if '://' not in value and not value.startswith('#'):assert (BASE/value).exists(),value
    doc='现有仓库采用根README＋event-drone-forecast子目录。\n\n'
    for c in counts:
        doc+='第'+str(c['batch'])+'次：打开GitHub仓库根目录的Add file → Upload files，拖入以下文件夹（'+str(c['files'])+'文件）：\n'+c['drag_this_folder']+'\n提交后再上传下一批。\n\n'
    doc+='然后将root_files_for_nested_layout中的文件和LICENSES文件夹全选，拖到GitHub根目录提交；不要把root_files_for_nested_layout本身拖进去。\n也可直接将ROOT_README_FOR_NESTED.md全文复制到根README.md。\n不要上传整个output、verification或外层上传第X批文件夹。\n'
    (BASE/'拖拽上传说明.txt').write_text(doc,'utf-8')
    (BASE/'upload_batches.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(counts,ensure_ascii=False))

if __name__=='__main__':main()
