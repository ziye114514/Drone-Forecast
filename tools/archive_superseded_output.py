"""Archive superseded outputs with per-file SHA256; deletion is a separate native PowerShell step."""
from pathlib import Path
import argparse,hashlib,json,os,stat,time,zipfile,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output';RECORD=OUT/'cleanup_20261028'
ARCHIVE=ROOT/'archive/output_history_20261028.zip'
KEEP={'checkpoints','github_release_final_20261027','github_release_update_20261028',
      'shared_template_budget_20261027','three_continuation_20261026','shared_angular_phase_20261027',
      'cleanup_20261028','结果入口.md','查看实验结果.html'}
def safe(p,base=OUT):
    p=p.resolve(strict=True)
    if p==base.resolve() or not p.is_relative_to(base.resolve()):raise ValueError('Outside intended output subtree: '+str(p))
    return p
def entries():
    selected=[]
    for p in sorted(OUT.iterdir()):
        if p.name in KEEP or p.name.startswith('dense_labels_'):continue
        selected.append(p)
    for name in ['github_release_final_20261027','github_release_update_20261028']:
        p=OUT/name/'verification'
        if p.is_dir():selected.append(p)
    return selected
def plan():
    RECORD.mkdir(exist_ok=True);chosen=entries();files=[];empty=[];delete=[]
    for top in chosen:
        safe(top)
        members=[top]+list(top.rglob('*')) if top.is_dir() else [top]
        for p in members:
            s=p.lstat()
            if p.is_symlink() or getattr(s,'st_file_attributes',0)&0x400:raise ValueError('Reparse point needs separate handling: '+str(p))
            if p.is_file():files.append(dict(path=p.relative_to(OUT).as_posix(),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
            elif p.is_dir() and not any(p.iterdir()):empty.append(p.relative_to(OUT).as_posix()+'/')
        if top.name=='verification' and top.parent.name in KEEP:
            delete.extend(p.relative_to(OUT).as_posix() for p in top.iterdir() if p.is_dir())
        else:delete.append(top.relative_to(OUT).as_posix())
    result=dict(output=str(OUT),archive=str(ARCHIVE),created=time.time(),files=files,empty_directories=empty,
        delete_after_verified=delete,selections=[p.relative_to(OUT).as_posix() for p in chosen],
        keep_top=sorted(KEEP),keep_dense_labels=True,bytes=sum(r['bytes'] for r in files),
        disk_free_before=shutil.disk_usage(ROOT).free)
    (RECORD/'plan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(files=len(files),GiB=result['bytes']/1024**3,delete_entries=len(delete),archive=str(ARCHIVE)),ensure_ascii=False),flush=True)
def archive():
    data=json.loads((RECORD/'plan.json').read_text('utf-8'));ARCHIVE.parent.mkdir(exist_ok=True)
    if ARCHIVE.exists():raise FileExistsError(ARCHIVE)
    temp=ARCHIVE.with_suffix('.zip.partial')
    if temp.exists():raise FileExistsError(temp)
    manifest=[];done=0;stamp=time.monotonic()
    stored={'.mp4','.webm','.zip','.npz','.png','.jpg','.jpeg','.gif','.deflate','.7z'}
    with zipfile.ZipFile(temp,'w',allowZip64=True,compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for row in data['files']:
            p=safe(OUT/row['path']);s=p.stat()
            assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns'],row['path']
            info=zipfile.ZipInfo.from_file(p,arcname=row['path'])
            info.compress_type=zipfile.ZIP_STORED if p.suffix.lower() in stored else zipfile.ZIP_DEFLATED
            info._compresslevel=1;digest=hashlib.sha256();length=0
            with p.open('rb') as source,z.open(info,'w',force_zip64=True) as dest:
                for chunk in iter(lambda:source.read(2**20),b''):
                    digest.update(chunk);dest.write(chunk);length+=len(chunk)
            after=p.stat();assert after.st_size==s.st_size and after.st_mtime_ns==s.st_mtime_ns and length==row['bytes']
            manifest.append(dict(path=row['path'],bytes=length,sha256=digest.hexdigest()));done+=length
            if time.monotonic()-stamp>=15:
                print('archive',len(manifest),'/',len(data['files']),'GiB',round(done/1024**3,2),flush=True);stamp=time.monotonic()
        for name in data['empty_directories']:z.writestr(name,b'')
        z.writestr('__ARCHIVE_MANIFEST__.json',json.dumps(dict(original_root='output/',files=manifest),ensure_ascii=False,indent=2))
    # Read every compressed member, check CRC automatically and compare SHA256.
    count=0;stamp=time.monotonic()
    with zipfile.ZipFile(temp) as z:
        for row in manifest:
            digest=hashlib.sha256();length=0
            with z.open(row['path']) as source:
                for chunk in iter(lambda:source.read(2**20),b''):digest.update(chunk);length+=len(chunk)
            assert digest.hexdigest()==row['sha256'] and length==row['bytes'],row['path'];count+=1
            if time.monotonic()-stamp>=15:
                print('verify',count,'/',len(manifest),flush=True);stamp=time.monotonic()
    # No original can be removed if any source changed after archive creation.
    for row in data['files']:
        p=safe(OUT/row['path']);s=p.stat()
        assert s.st_size==row['bytes'] and s.st_mtime_ns==row['mtime_ns'],row['path']
    temp.replace(ARCHIVE)
    result=dict(archive=str(ARCHIVE),archive_bytes=ARCHIVE.stat().st_size,original_bytes=data['bytes'],
        verified_file_SHA256_count=count,all_files_verified=True,originals_deleted=False,
        intended_delete_entries=data['delete_after_verified'],frozen_publish_ZIPs_unchanged=True)
    (RECORD/'archive_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(result|dict(intended_delete_entries=len(result['intended_delete_entries'])),ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','archive']);a=p.parse_args()
    plan() if a.action=='plan' else archive()
