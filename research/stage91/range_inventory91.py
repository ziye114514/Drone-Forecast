"""Read only ZIP directories and annotation members through HTTP byte ranges.

No images/RAW downloaded. Verify every206 Content-Range; abort on a server
which ignores ranges. Fixed metadata budget and prechosen sequence IDs.
"""
from pathlib import Path
import datetime, hashlib, io, json, sys, urllib.request, zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage88'))
from inventory88 import save,load,parse,unique_rows,sha
OUT=ROOT.parent/'output/stage91_FRED_download_audit_20261022'
IDS=[8,9,12,20,21,29,34,39,41,46]
BUDGET=32*1024*1024
consumed=0


def request(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'event-uav-FRED-metadata-audit'}),timeout=45) as r:
        return json.load(r)


class RangedZip(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.position=0;self.calls=[]
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if not 0<=self.position<=self.size:raise ValueError('Out of archive bounds')
        return self.position
    def read(self,n=-1):
        global consumed
        if n<0:n=self.size-self.position
        n=min(n,self.size-self.position)
        if not n:return b''
        if n>8*1024*1024 or consumed+n>BUDGET:raise RuntimeError('Metadata range budget exceeded')
        start=self.position;end=start+n-1
        req=urllib.request.Request(self.url+f'?metadata_range={start}-{end}',headers={'User-Agent':'event-uav-FRED-metadata-audit','Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=45) as response:
            assert response.status==206,'Range ignored: abort, never read whole dataset'
            assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}'
            data=response.read(n+1);assert len(data)==n
        self.position+=n;consumed+=n;self.calls.append(dict(start=start,end=end,bytes=n,sha256=hashlib.sha256(data).hexdigest()))
        return data


def main():
    assert not OUT.exists()
    meta=request('https://huggingface.co/api/datasets/GabrieleMagrini/FRED');commit=meta['sha']
    tree=request(f'https://huggingface.co/api/datasets/GabrieleMagrini/FRED/tree/{commit}/test')
    files={int(Path(x['path']).stem):x for x in tree if x['path'].endswith('.zip')}
    save(OUT/'protocol91.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),
        official_dataset='GabrieleMagrini/FRED',official_revision=commit,ids=IDS,budget_bytes=BUDGET,
        scope='ZIP central directories and original annotation text only. No media/RAW download.',
        selection='Fixed first8 missing canonical-test IDs plus existing8/9 provenance checks. Rank for future geometry data by supplied box size, never by forecasting error.',
        limitations='Large boxes do not guarantee four visible rotors; this is a download screen, not a test accuracy score.'))
    local={r['sequence']:r for r in load(ROOT.parent/'output/stage88_causal_event_controls_20261022/inventory88.json')['sequences']}
    result=[]
    for seq in IDS:
        row=dict(sequence=seq)
        try:
            x=files[seq];row.update(archive_bytes=x['size'],archive_path=x['path'])
            url=f'https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/{commit}/{x["path"]}'
            reader=RangedZip(url,x['size'])
            with zipfile.ZipFile(reader) as z:
                names=[n for n in z.namelist() if Path(n).name=='coordinates.txt'];assert len(names)==1
                info=z.getinfo(names[0]);assert info.file_size<4*1024*1024
                content=z.read(names[0]);rr,bad=parse(content.decode('utf-8-sig'));assert not bad
                rr=unique_rows(rr);assert rr
                side=np.array([min(r['bbox'][2]-r['bbox'][0],r['bbox'][3]-r['bbox'][1]) for r in rr])
                dest=OUT/'annotations'/f'{seq}.txt';dest.parent.mkdir(exist_ok=True);dest.write_bytes(content)
                raw_names=[n for n in z.namelist() if n.lower().endswith(('.raw','.hdf5','.h5'))]
                row.update(annotation_member=names[0],annotation_sha256=sha(dest),unique_valid_boxes=len(rr),
                    median_min_box_side_px=float(np.median(side)),p90_min_box_side_px=float(np.percentile(side,90)),
                    fraction_side24=float(np.mean(side>=24)),fraction_side48=float(np.mean(side>=48)),fraction_side96=float(np.mean(side>=96)),
                    event_members=[dict(name=n,bytes=z.getinfo(n).file_size) for n in raw_names],
                    annotation_matches_existing=seq in local and local[seq]['annotation_sha256']==sha(dest),
                    bytes_read=sum(c['bytes'] for c in reader.calls),range_requests=reader.calls,
                    download_url=f'https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/{commit}/{x["path"]}?download=true')
        except Exception as exc:
            row['error']=type(exc).__name__+': '+str(exc)[:220]
        result.append(row);save(OUT/'partial_inventory91.json',dict(records=result,bytes_read=consumed))
        print(json.dumps({k:v for k,v in row.items() if k in ['sequence','archive_bytes','median_min_box_side_px','fraction_side96','annotation_matches_existing','error']},ensure_ascii=False),flush=True)
    usable=[r for r in result if 'error' not in r and r['sequence'] not in [8,9]]
    ranked=sorted(usable,key=lambda r:(r['fraction_side96'],r['median_min_box_side_px']),reverse=True)
    save(OUT/'inventory91.json',dict(records=result,download_priority_ids=[r['sequence'] for r in ranked],bytes_read=consumed,
        media_downloaded=False,official_revision=commit,source_hash_unchanged=sha(__file__)==load(OUT/'protocol91.json')['source_sha256']))
    print('91 metadata-only download screening complete; bytes',consumed,flush=True)


if __name__=='__main__':main()
