"""Extract only one publicly hosted native event member via verified byte range."""
from pathlib import Path
import argparse,binascii,datetime,hashlib,json,struct,sys,time,urllib.request,zipfile,zlib
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage91'),str(ROOT/'research/stage88')]
import range_inventory91 as ranged
from inventory88 import save,load,sha,parse,unique_rows
OUT=ROOT.parent/'output/stage94_official_large_target_20261022'
META=ROOT.parent/'output/stage91b_FRED_all_test_screen_20261022'


def freeze():
    assert not OUT.exists();m=load(META/'inventory91b.json')
    records=[r for r in m['records'] if 'error' not in r and r['sequence'] not in [8,9]]
    selected=max(records,key=lambda r:(r['fraction_side96'],r['median_min_box_side_px']))
    assert selected['sequence']==194
    commit=load(META/'protocol91b.json')['official_revision']
    annotation=META/'annotations/194.txt';rr,bad=parse(annotation.read_text(encoding='utf-8-sig'));assert not bad;rr=unique_rows(rr)
    # Midpoint interval selected from time support, not forecast error or motion.
    midpoint=int((rr[0]['t']+rr[-1]['t'])/2);start=max(0,midpoint-5)
    save(OUT/'protocol94.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),
        official_revision=commit,selected=selected,annotation_path=str(annotation),annotation_sha256=sha(annotation),
        start_s=start,duration_s=10.,selection='Largest supplied-box fraction>=96px among46 successful official-test metadata screens; no forecasting error used.',
        interpretation='Size-conditioned feasibility diagnostic, not unbiased benchmark. Public annotation metadata already inspected.',
        native_event_download_max_compressed_bytes=700*1024*1024,media_policy='Only native events.hdf5 and existing annotation; no RGB or event PNGs.',
        no_default_promotion=True))
    print('94 frozen large-target data selection',selected['sequence'],'window',start,start+10,flush=True)


def download():
    p=load(OUT/'protocol94.json');assert p['source_sha256']==sha(__file__)
    r=p['selected'];url=f'https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/{p["official_revision"]}/{r["archive_path"]}'
    reader=ranged.RangedZip(url,r['archive_bytes'])
    with zipfile.ZipFile(reader) as z:
        member=r['event_members'][0]['name'];info=z.getinfo(member)
        assert info.compress_type in [zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED]
        assert info.compress_size<=p['native_event_download_max_compressed_bytes']
        reader.seek(info.header_offset);header=reader.read(30)
        assert header[:4]==b'PK\x03\x04'
        name_len,extra_len=struct.unpack_from('<HH',header,26)
        begin=info.header_offset+30+name_len+extra_len;end=begin+info.compress_size-1
    dest=OUT/'input/events.hdf5';dest.parent.mkdir(exist_ok=True);partial=dest.with_suffix('.hdf5.partial')
    assert not partial.exists() and not dest.exists()
    expected=f'bytes {begin}-{end}/{r["archive_bytes"]}'
    req=urllib.request.Request(url+f'?native_event_range={begin}-{end}',headers={'User-Agent':'event-uav-native-data-research','Range':f'bytes={begin}-{end}'})
    decoded=0;received=0;crc=0;digest=hashlib.sha256();decompress=zlib.decompressobj(-zlib.MAX_WBITS) if info.compress_type==8 else None
    started=time.perf_counter();last=started
    with urllib.request.urlopen(req,timeout=90) as response,partial.open('xb') as f:
        assert response.status==206 and response.headers.get('Content-Range')==expected
        while True:
            b=response.read(1024*1024)
            if not b:break
            received+=len(b);assert received<=info.compress_size
            chunk=decompress.decompress(b) if decompress else b
            f.write(chunk);digest.update(chunk);crc=binascii.crc32(chunk,crc);decoded+=len(chunk)
            if time.perf_counter()-last>15:
                print(json.dumps(dict(compressed_received=received,compressed_total=info.compress_size,native_decoded=decoded,phase='native event member only')),flush=True);last=time.perf_counter()
        if decompress:
            chunk=decompress.flush();f.write(chunk);digest.update(chunk);crc=binascii.crc32(chunk,crc);decoded+=len(chunk)
    assert received==info.compress_size and decoded==info.file_size and crc==info.CRC
    if decompress:assert decompress.eof
    partial.rename(dest)
    (dest.parent/'coordinates.txt').write_bytes(Path(p['annotation_path']).read_bytes())
    save(OUT/'download94.json',dict(sequence=194,official_revision=p['official_revision'],member=member,path=str(dest),
        compressed_bytes=received,native_bytes=decoded,native_sha256=digest.hexdigest(),crc32=crc,crc_matches_zip=True,
        downloaded_RGB_or_PNG=False,wall_s=time.perf_counter()-started,source_hash_unchanged=p['source_sha256']==sha(__file__)))
    print('94 native event download verified',received,decoded,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','download']);a=ap.parse_args();globals()[a.action]()
