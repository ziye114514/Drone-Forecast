"""Screen remaining canonical test annotations with the same ranged ZIP reader."""
from pathlib import Path
import datetime,json,sys,zipfile
import numpy as np
import range_inventory91 as old
ROOT=old.ROOT;OUT=ROOT.parent/'output/stage91b_FRED_all_test_screen_20261022'


def main():
    assert not OUT.exists()
    previous=old.load(old.OUT/'inventory91.json');commit=previous['official_revision']
    split=ROOT.parent/'output/stage7_localization_budget_v3_20261017/upstream/FRED/dataset_splits/canonical/test_split.txt'
    ids={int(x.strip().strip('/')) for x in split.read_text().splitlines() if x.strip()}
    seen={r['sequence'] for r in previous['records']};remaining=sorted(ids-seen)
    tree=old.request(f'https://huggingface.co/api/datasets/GabrieleMagrini/FRED/tree/{commit}/test')
    files={int(Path(x['path']).stem):x for x in tree if x['path'].endswith('.zip')}
    old.BUDGET=48*1024*1024;old.consumed=previous['bytes_read']
    old.save(OUT/'protocol91b.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=old.sha(__file__),
        range_reader_sha256=old.sha(Path(old.__file__)),official_revision=commit,remaining=remaining,
        combined_budget_bytes=old.BUDGET,previous_metadata_bytes=old.consumed,
        scope='Read only annotation members and ZIP directories for remaining canonical test sequences. No image/RAW download or prediction score selection.'))
    result=[]
    for seq in remaining:
        row=dict(sequence=seq)
        try:
            x=files[seq];row.update(archive_bytes=x['size'],archive_path=x['path'])
            reader=old.RangedZip(f'https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/{commit}/{x["path"]}',x['size'])
            with zipfile.ZipFile(reader) as z:
                names=[n for n in z.namelist() if Path(n).name=='coordinates.txt'];assert len(names)==1
                assert z.getinfo(names[0]).file_size<4*1024*1024
                content=z.read(names[0]);rr,bad=old.parse(content.decode('utf-8-sig'));assert not bad
                rr=old.unique_rows(rr);assert rr
                side=np.array([min(r['bbox'][2]-r['bbox'][0],r['bbox'][3]-r['bbox'][1]) for r in rr])
                path=OUT/'annotations'/f'{seq}.txt';path.parent.mkdir(exist_ok=True);path.write_bytes(content)
                row.update(annotation_member=names[0],annotation_sha256=old.sha(path),unique_valid_boxes=len(rr),
                    median_min_box_side_px=float(np.median(side)),p90_min_box_side_px=float(np.percentile(side,90)),
                    fraction_side48=float(np.mean(side>=48)),fraction_side96=float(np.mean(side>=96)),
                    bytes_read=sum(c['bytes'] for c in reader.calls),range_requests=reader.calls,
                    event_members=[dict(name=n,bytes=z.getinfo(n).file_size) for n in z.namelist() if n.lower().endswith(('.raw','.hdf5','.h5'))],
                    download_url=f'https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/{commit}/{x["path"]}?download=true')
        except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)[:220]
        result.append(row);old.save(OUT/'partial_inventory91b.json',dict(records=result,combined_metadata_bytes=old.consumed))
        print(json.dumps({k:v for k,v in row.items() if k in ['sequence','median_min_box_side_px','fraction_side96','error']}),flush=True)
    all_rows=previous['records']+result;usable=[r for r in all_rows if 'error' not in r]
    ranked=sorted(usable,key=lambda r:(r['fraction_side96'],r['median_min_box_side_px']),reverse=True)
    old.save(OUT/'inventory91b.json',dict(records=all_rows,download_priority_ids=[r['sequence'] for r in ranked],
        canonical_test_total=len(ids),successes=len(usable),combined_metadata_bytes=old.consumed,media_downloaded=False,
        source_hash_unchanged=old.sha(__file__)==old.load(OUT/'protocol91b.json')['source_sha256']))
    print('91b complete',len(usable),'top',[(r['sequence'],r['median_min_box_side_px'],r['fraction_side96']) for r in ranked[:8]],flush=True)


if __name__=='__main__':main()
