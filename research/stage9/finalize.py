"""Check local evidence links and retain a source snapshot, without other projects."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote
import shutil
from study import OUT,ROOT,load,save,verify,sha


class Links(HTMLParser):
    def __init__(self):super().__init__();self.values=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href'] and value and not value.startswith(('http:','https:','#')):
                self.values.append(value)


if __name__=='__main__':
    verify();v=load(OUT/'validation.json');assert v['causal_prefix_checked'] and v['cached_baseline_equal']
    events=load(OUT/'side/event_audit.json');assert events['published_exact']==200
    parser=Links();parser.feed((OUT/'第九阶段整机状态保护报告.html').read_text(encoding='utf-8'))
    for value in parser.values:assert (OUT/unquote(value)).is_file(),value
    smoke=load(OUT/'cli_smoke/summary.json');assert smoke['processed_ticks']==53 and smoke['source_unchanged']
    for relative,expected in smoke['source_hashes'].items():assert sha(ROOT/relative)==expected,relative
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    paths=[p for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ['.py','.md']]
    paths.append(ROOT/'core/run_state_recovery.py');manifest=[]
    for source in paths:
        target=snapshot/source.name;shutil.copyfile(source,target)
        manifest.append(dict(path=str(source.relative_to(ROOT)),sha256=sha(target)))
    save(OUT/'final_validation.json',dict(frozen_inference_unchanged=True,
        local_links_verified=len(parser.values),side_all_tick_replay_exact=True,cli_smoke_ticks=53,
        source_snapshot=manifest,protected_project_written=False,
        limitations='No recursive protected-project integrity scan. Only local experiment evidence checked; no SOTA or trajectory accuracy claim.'))
    print('Stage9 links, source snapshots, CLI and audit checked')
