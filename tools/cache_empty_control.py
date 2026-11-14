"""Cache2s from the project's pre-existing empty self-recording5 control."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches


def main(output):
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    out=Path(output)/'empty_control';out.mkdir(exist_ok=False)
    noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buf=noise.get_empty_output_buffer();image=np.empty((720,1280,3),np.uint8)
    raw=Path('E:/dataset/train/5/media/event.raw');before=raw.stat();samples=[];n=0
    for t,e in native_batches(raw,0.,2.,False,33333):
        noise.process_events(e,buf);BaseFrameGenerationAlgorithm.generate_frame(buf.numpy(),image)
        cv2.imwrite(str(out/f'{n:04d}.png'),image)
        if n in [0,29,59]:samples.append(cv2.resize(image,(640,360),interpolation=cv2.INTER_NEAREST))
        n+=1
    (out/'protocol.json').write_text(json.dumps(dict(raw=str(raw),start=0,duration=2.,frames=n,
        source='Existing project empty recording5, not newly mined negative data',
        raw_unchanged=raw.stat().st_size==before.st_size and raw.stat().st_mtime_ns==before.st_mtime_ns,
        future_or_annotation_input=False),indent=2),'utf-8')
    cv2.imwrite(str(out/'review.png'),np.vstack(samples));print(n,flush=True)


if __name__=='__main__':main(sys.argv[1])
