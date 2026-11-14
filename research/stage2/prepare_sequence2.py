"""Resolve nested archive names by exact basename, without changing estimator.

The frozen generic suffix lookup matched interpolated_coordinates.txt too and
stopped before sequence2 inference. Select only the original coordinates.txt.
"""
from pathlib import Path
import shutil,zipfile
from study import OUT,save,digest

folder=OUT/'activity_fallback/sequence2/input';folder.mkdir(parents=True,exist_ok=True)
archive=Path(r'E:\FRED\2.zip');records=[]
with zipfile.ZipFile(archive) as z:
    for basename in ('events.raw','coordinates.txt'):
        matches=[n for n in z.namelist() if Path(n).name==basename]
        assert len(matches)==1,matches
        member=matches[0];target=folder/basename;info=z.getinfo(member)
        if not target.exists():
            with z.open(member) as source,target.open('wb') as f:shutil.copyfileobj(source,f)
        assert target.stat().st_size==info.file_size
        records.append(dict(member=member,target=basename,bytes=target.stat().st_size,sha256=digest(target),crc32=info.CRC))
save(folder/'inventory.json',dict(archive=str(archive),members=records,preparation_repair='Exact basename excludes interpolated_coordinates.txt; estimator code and frozen settings unchanged.',
    repair_before_sequence2_inference=True,labels_not_passed_to_estimator=True))
print('Exact original annotations prepared; no interpolated labels selected',flush=True)
