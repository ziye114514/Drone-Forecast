"""Resumable native-member compressed blocks, full ZIP CRC before event use."""
from pathlib import Path
import binascii,datetime,hashlib,json,struct,sys,time,urllib.request,zipfile,zlib
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage91'),str(ROOT/'research/stage88')]
import range_inventory91 as ranged
from inventory88 import load,save,sha
OUT=ROOT.parent/'output/stage102_resumable_native_download_20261022'
DATA=ROOT.parent/'output/stage100_official195_longer_20261022'
BLOCK=32*1024*1024


def main():
    p=load(DATA/'protocol100.json');r=p['selected'];url=f'https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/{p["official_revision"]}/{r["archive_path"]}'
    rd=ranged.RangedZip(url,r['archive_bytes'])
    with zipfile.ZipFile(rd) as z:
        info=z.getinfo(r['event_members'][0]['name']);rd.seek(info.header_offset);header=rd.read(30);assert header[:4]==b'PK\x03\x04'
        n,e=struct.unpack_from('<HH',header,26);begin=info.header_offset+30+n+e;length=info.compress_size
    metadata=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),official_revision=p['official_revision'],
        member=info.filename,zip_bytes=r['archive_bytes'],compressed_begin=begin,compressed_length=length,native_bytes=info.file_size,native_crc32=info.CRC,
        block_bytes=BLOCK,per_block_attempts=8,preserve_first_failure=True,download_RGB=False)
    protocol=OUT/'protocol102.json'
    if protocol.exists():
        old=load(protocol)
        for k in ['source_sha256','official_revision','member','compressed_begin','compressed_length','native_bytes','native_crc32','block_bytes']:assert old[k]==metadata[k]
    else:save(protocol,metadata)
    blocks=OUT/'blocks';blocks.mkdir(exist_ok=True);inventory=[];beg=time.perf_counter()
    for index,start in enumerate(range(0,length,BLOCK)):
        size=min(BLOCK,length-start);part=blocks/f'{index:04d}.part';done=blocks/f'{index:04d}.deflate'
        if done.exists():assert done.stat().st_size==size
        else:
            for attempt in range(8):
                existing=part.stat().st_size if part.exists() else 0;assert 0<=existing<=size
                if existing==size:break
                first=begin+start+existing;last=begin+start+size-1
                request=urllib.request.Request(url+f'?native_block={index}-{existing}-{attempt}',headers={'User-Agent':'event-uav-resumable-native-download','Range':f'bytes={first}-{last}'})
                try:
                    with urllib.request.urlopen(request,timeout=60) as response:
                        assert response.status==206 and response.headers.get('Content-Range')==f'bytes {first}-{last}/{r["archive_bytes"]}'
                        with part.open('ab') as f:
                            remaining=size-existing
                            while remaining:
                                chunk=response.read(min(1024*1024,remaining))
                                if not chunk:raise IOError('truncated HTTP range')
                                f.write(chunk);remaining-=len(chunk)
                    if part.stat().st_size==size:break
                except Exception as exc:
                    save(OUT/f'block{index:04d}_attempt{attempt}.json',dict(error=type(exc).__name__+': '+str(exc)[:200],partial_bytes=part.stat().st_size if part.exists() else 0,expected_bytes=size))
                    print('102 range retry',index,attempt,type(exc).__name__,flush=True)
            assert part.exists() and part.stat().st_size==size,('Block incomplete; rerun to resume',index)
            part.rename(done)
        inventory.append(dict(block=index,bytes=size,sha256=sha(done)))
        save(OUT/'block_inventory102.json',dict(blocks=inventory,completed_bytes=sum(x['bytes'] for x in inventory),total_bytes=length))
        print('102 verified compressed block',index,sum(x['bytes'] for x in inventory),'/',length,flush=True)
    target=DATA/'input/events.hdf5';partial=DATA/'input/events.recovered.hdf5.partial';assert not target.exists() and not partial.exists()
    decoder=zlib.decompressobj(-zlib.MAX_WBITS) if info.compress_type==8 else None
    assert info.compress_type in [0,8]
    crc=0;digest=hashlib.sha256();decoded=0
    with partial.open('xb') as dst:
        for row in inventory:
            block=blocks/f'{row["block"]:04d}.deflate';assert sha(block)==row['sha256']
            with block.open('rb') as src:
                while True:
                    chunk=src.read(1024*1024)
                    if not chunk:break
                    value=decoder.decompress(chunk) if decoder else chunk
                    dst.write(value);digest.update(value);crc=binascii.crc32(value,crc);decoded+=len(value)
        if decoder:
            value=decoder.flush();dst.write(value);digest.update(value);crc=binascii.crc32(value,crc);decoded+=len(value);assert decoder.eof
    assert decoded==info.file_size and crc==info.CRC
    partial.rename(target)
    annotation=ROOT.parent/'output/stage91b_FRED_all_test_screen_20261022/annotations/195.txt';(target.parent/'coordinates.txt').write_bytes(annotation.read_bytes())
    result=dict(sequence=195,official_revision=p['official_revision'],member=info.filename,path=str(target),compressed_bytes=length,native_bytes=decoded,
        native_sha256=digest.hexdigest(),crc32=crc,crc_matches_zip=True,downloaded_RGB_or_PNG=False,resumable_blocks=True,
        first_partial_preserved=True,wall_s=time.perf_counter()-beg,source_hash_unchanged=sha(__file__)==load(protocol)['source_sha256'])
    save(DATA/'download94.json',result);save(OUT/'completion102.json',result)
    print('102 full native CRC/SHA verified',decoded,flush=True)


if __name__=='__main__':main()
