"""Pre-training bridge: clip YOLO labels to FOV, preserve original evaluation boxes."""
import datetime
from common40 import *

def main():
    verify();assert not (OUT/'inputs.json').exists() and not (OUT/'training').exists()
    supplement=OUT/'border_supplement.json'
    assert not supplement.exists()
    save(supplement,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),
        reason='Frozen preparation stopped on first train6 coarse box beyond native720px height. Six train6 validation boxes exceed FOV by2.62–15.21px. All other boxes within FOV and all clipped boxes retain positive area. No training or inference has occurred.',
        correction='Only YOLO train/validation label serialization clamps x to0..1280,y to0..720. targets.json retains original stored boxes exactly for final endpoint/box scoring. Full native input frames and sample selection unchanged. Original frozen prepare40.py/protocol remain intact; preserve partial dataset.'))
    partial=OUT/'failed_border_preparation_v1';partial.mkdir(exist_ok=False);(OUT/'dataset').rename(partial/'dataset')
    source=(Path(__file__).with_name('prepare40.py')).read_text(encoding='utf-8')
    a="b=np.array(label['bbox'],float);assert";b="b=np.clip(np.array(label['bbox'],float),[0,0,0,0],[w,h,w,h]);assert"
    assert source.count(a)==1;source=source.replace(a,b)
    a="target['bbox']=b.tolist();box=";b="target['bbox']=list(label['bbox']);box=";assert source.count(a)==1;source=source.replace(a,b)
    namespace={'__name__':'prepare40_bridge','__file__':str(Path(__file__).with_name('prepare40.py'))};exec(compile(source,namespace['__file__'],'exec'),namespace);namespace['main']()
    prep=load(OUT/'preparation.json');prep['border_supplement_sha256']=sha(supplement);prep['evaluation_boxes_not_clipped']=True;save(OUT/'preparation.json',prep)
if __name__=='__main__':main()
