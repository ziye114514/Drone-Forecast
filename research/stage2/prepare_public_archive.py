"""Exact archive names; labels extracted but never passed to inference."""
from pathlib import Path
import argparse,shutil,zipfile
from study import OUT,save,digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('sequence',choices=['2','3']);args=parser.parse_args()
    folder=OUT/f'activity_fallback_fixed/sequence{args.sequence}/input';folder.mkdir(parents=True,exist_ok=True)
    archive=Path(r'E:\FRED')/f'{args.sequence}.zip';records=[]
    with zipfile.ZipFile(archive) as z:
        for basename in ('events.raw','coordinates.txt'):
            matches=[n for n in z.namelist() if Path(n).name==basename];assert len(matches)==1,matches
            member=matches[0];target=folder/basename;info=z.getinfo(member)
            if not target.exists():
                with z.open(member) as source,target.open('wb') as f:shutil.copyfileobj(source,f)
            assert target.stat().st_size==info.file_size
            records.append(dict(member=member,target=basename,bytes=target.stat().st_size,sha256=digest(target),crc32=info.CRC))
    save(folder/'inventory.json',dict(archive=str(archive),members=records,labels_not_passed_to_estimator=True,
        no_interpolated_coordinate_file_selected=True))
    print('Exact original annotations prepared for sequence'+args.sequence,flush=True)

if __name__=='__main__':main()
