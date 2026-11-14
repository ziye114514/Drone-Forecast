"""Render completed full RAW pipeline output; no new detector or inference."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(Path(__file__).parent),str(ROOT)]
from reference_api import OUT
from live_decode import native_batches
from metavision_sdk_core import BaseFrameGenerationAlgorithm
import cv2

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',default='RAW_reanchor');args=parser.parse_args()
out=OUT/args.case;protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
rows=[json.loads(s) for s in (out/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
theta=np.linspace(0,2*np.pi,128);circle=np.c_[np.cos(theta),np.sin(theta),np.ones(len(theta))]
writer=cv2.VideoWriter(str(out/'phase_and_forecast.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50.,(1280,960));assert writer.isOpened()
tiles=[];sample_times=[23.2,25.6,26.66,29.0,protocol['start']+.3,protocol['start']+.6,protocol['start']+.9];lastcenter=np.array([640.,360.]);frames=0
try:
    for k,(t,e) in enumerate(native_batches(protocol['raw'],protocol['start'],protocol['duration'])):
        row=rows[k];assert abs(row['t']-t)<1e-8
        im=np.zeros((720,1280,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(e,im)
        original=im.copy();body=row['body'];obs=row['auxiliary_observation'];shift=np.asarray(obs.get('body_translation',[0.,0.]))
        if body['detected']:
            box=np.rint(body['bbox']).astype(int);cv2.rectangle(im,tuple(box[:2]),tuple(box[2:]),(190,190,190),1)
            lastcenter=np.asarray(body['center'])
        preferred=row['preferred_forecast']
        if row['valid'] and preferred in row['predictions']:
            c=row['coefficients'].get('optimized_geometry')
            if c:
                points=np.rint([c['position']]+[row['predictions'][preferred][str(h)] for h in [.1,.2,.4]]).astype(np.int32)
                cv2.polylines(im,[points],False,(40,190,40),2)
        periodic=obs.get('body_periodicity') or {}
        if periodic.get('axis_valid'):
            line=periodic['line'];angle=np.deg2rad(line['angle']);center=np.asarray(line['center'])
            direction=np.array([np.cos(angle),np.sin(angle)])*line['major']/2
            cv2.line(im,tuple(np.rint(center-direction).astype(int)),tuple(np.rint(center+direction).astype(int)),(255,200,0),2)
            cv2.putText(im,f'Periodic AXIS; {periodic["frequency_hz"]:.0f} flickerHz',tuple(np.rint(center+[0,-20]).astype(int)),0,.45,(255,200,0),1)
        for i,value in obs.get('rotors',{}).items():
            if not value['valid']:continue
            r=value['rectification_rotor'];H=np.asarray(value['selected']['H']);p=circle@H.T
            patch=p[:,:2]/p[:,2,None];a=np.deg2rad(r['angle']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            curve=((patch-128)*np.array([r['major']/2,r['minor']/2])/80)@R.T+[r['cx'],r['cy']]+shift
            cv2.polylines(im,[np.rint(curve).astype(np.int32)],True,(0,190,255),1)
            center=value.get('current_projection',{}).get('center') or value.get('conditional_conic_center_image_xy')
            if center:
                xy=np.rint(center).astype(int);cv2.putText(im,f'{i}:{value["selected"]["phase_hz"]:.0f}Hz',tuple(xy),0,.4,(0,190,255),1)
        canvas=np.zeros((960,1280,3),np.uint8);canvas[:720]=im
        center=np.minimum(np.maximum(np.rint(lastcenter).astype(int),[120,60]),[1160,660])
        zoom=im[center[1]-60:center[1]+60,center[0]-120:center[0]+120]
        canvas[720:960,800:1280]=cv2.resize(zoom,(480,240),interpolation=cv2.INTER_NEAREST)
        proxy=obs.get('conditional_rotation_proxy') or {};rho=proxy.get('relative_to_first_supported')
        messages=[f'RAW t={t:.2f}s; original motion forecast remains unchanged',
            f'Rotor support {obs.get("supported_candidates",0)}/4; body period={periodic.get("qualified",False)}, axis={periodic.get("axis_valid",False)}',
            'Yellow: phase-supported conic; cyan: unsigned periodic image axis',
            'Model phase Hz != true mechanical RPM; force is NOT calibrated',
            f'Conditional relative rotation activity: {rho:.3f}' if rho else 'Conditional relative rotation activity: unavailable',
            '50fps offline playback; NOT realtime performance evidence']
        for j,message in enumerate(messages):cv2.putText(canvas,message,(12,752+34*j),0,.49,(220,220,220),1)
        writer.write(canvas);frames+=1
        if any(abs(t-s)<1e-7 for s in sample_times):
            path=out/f'native_{t:.2f}.png';cv2.imwrite(str(path),canvas);tiles.append(path.name)
finally:writer.release()
assert frames==len(rows)
(out/'render_check.json').write_text(json.dumps(dict(frames=frames,native_timestamps_preserved=True,
    reference='Published estimate JSON over same native RAW slice; no new inference or labels',screenshots=tiles),indent=2),encoding='utf-8')
print('Rendered frames:',frames,flush=True)
