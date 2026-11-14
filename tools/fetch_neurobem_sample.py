"""Fetch one preselected CSV using HTTP ranges; never download the627MB ZIP."""
from pathlib import Path
import hashlib,io,json,re,sys,urllib.request,zipfile
URL='https://download.ifi.uzh.ch/rpg/NeuroBEM/processed_data.zip'


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position=0;self.requests=[];self.transferred=0
        data,header=self.get('bytes=-65536')
        match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',header)
        if not match:raise RuntimeError('Missing valid Content-Range')
        start,end,self.size=map(int,match.groups());self.tail=(start,data)

    def get(self,range_header):
        req=urllib.request.Request(URL,headers={'Range':range_header,'User-Agent':'EventDroneForecast/0.3 research sample'})
        with urllib.request.urlopen(req,timeout=30) as r:
            if r.status!=206:raise RuntimeError('Server did not honor Range; full ZIP download refused')
            length=int(r.headers['Content-Length'])
            if self.transferred+length>12*1024*1024:raise RuntimeError('12MB transfer cap')
            data=r.read(length+1)
            if len(data)!=length:raise RuntimeError('Unexpected response size')
            header=r.headers['Content-Range']
        self.transferred+=len(data);self.requests.append(dict(range=range_header,content_range=header,bytes=len(data)))
        return data,header

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else (self.position+offset if whence==1 else self.size+offset)
        return self.position
    def read(self,n=-1):
        if n<0:n=self.size-self.position
        n=min(n,self.size-self.position)
        if n<=0:return b''
        start=self.position;end=start+n-1
        if start>=self.tail[0]:data=self.tail[1][start-self.tail[0]:start-self.tail[0]+n]
        else:
            data,header=self.get(f'bytes={start}-{end}')
            if not header.startswith(f'bytes {start}-{end}/'):raise RuntimeError('Wrong byte range returned')
        self.position+=len(data);return data


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    remote=RemoteZip()
    with zipfile.ZipFile(remote) as z:
        choices=sorted((i for i in z.infolist() if i.filename.endswith('.csv')),key=lambda i:i.filename)
        selected=choices[0] # Lexically first CSV, fixed before any score is seen.
        if selected.compress_size>8*1024*1024 or selected.file_size>32*1024*1024:raise RuntimeError('Selected file exceeds bounded sample size')
        (out/'selection.json').write_text(json.dumps(dict(source=URL,selection='lexically first CSV before scoring',
            filename=selected.filename,compressed_bytes=selected.compress_size,bytes=selected.file_size,
            all_csv_files=len(choices)),indent=2),'utf-8')
        data=z.read(selected)
    target=out/'sample.csv';target.write_bytes(data)
    result=dict(source=URL,member=selected.filename,sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),network_bytes=remote.transferred,requests=remote.requests,
        source_readme='https://rpg.ifi.uzh.ch/neuro_bem/Readme.html',
        scope='Public physical flight telemetry only; no external event-camera images. Local evaluation input, excluded from source release.')
    (out/'provenance.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main(sys.argv[1])
