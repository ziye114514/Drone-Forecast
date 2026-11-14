"""Local artifact checks and source snapshot, without reading other projects."""
import hashlib,json,shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from study import ROOT,OUT,verify,load,save


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.values=[]

    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href'] and value and not value.startswith(('http:','https:','#')):
                self.values.append(value)


if __name__=='__main__':
    verify()
    prior=[]
    for stage in ['stage6_body_axis_v2_20261017','stage7_localization_budget_v3_20261017']:
        hashes=load(ROOT.parent/'output'/stage/'protocol.json')['source_hashes']
        for rel,expected in hashes.items():
            assert digest(ROOT/rel)==expected,(stage,rel)
        prior.append(dict(stage=stage,unchanged_sources=len(hashes)))
    refs=load(OUT/'blind_reference/references.json')
    for frame in refs['frames']:
        path=OUT/'blind_reference'/f"{frame['case']}_{frame['t']:.2f}.png"
        assert digest(path)==frame['input_sha256']
    parser=Links();parser.feed((OUT/'第八阶段纯ONOFF桨盘提取报告.html').read_text(encoding='utf-8'))
    for value in parser.values:
        assert (OUT/unquote(value)).is_file(),value
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    files=[]
    for source in sorted(Path(__file__).parent.iterdir()):
        if source.is_file() and source.suffix in ['.py','.md']:
            target=snapshot/source.name;shutil.copyfile(source,target)
            files.append(dict(name=source.name,sha256=digest(target)))
    save(OUT/'final_validation.json',dict(frozen_stage8_unchanged=True,prior_sources=prior,
        raw_reference_images_unchanged=True,local_report_links_verified=len(parser.values),
        source_snapshot=files,no_other_project_access=True,
        limits='No recursive filesystem integrity scan of protected yolov10-main; this trial performed no access or writes there.'))
    print(json.dumps(dict(prior=prior,local_links=len(parser.values),snapshot_files=len(files)),ensure_ascii=False))
