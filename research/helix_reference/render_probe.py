"""Native ON/OFF overlay and scientific figures for the completed author probe."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT)]
from reference_api import OUT
from live_decode import native_batches
from metavision_sdk_core import BaseFrameGenerationAlgorithm
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=OUT/'adaptive_signed/probe';d=json.loads((out/'results.json').read_text(encoding='utf-8'))
parent=json.loads((OUT/'real_probe.json').read_text(encoding='utf-8'))
t0=d['protocol']['anchor_t'];velocity=np.asarray(parent['protocol']['velocity_only_at_anchor'])
anchor=next(json.loads(s) for s in Path(parent['protocol']['current_geometry_source']).read_text(encoding='utf-8').splitlines() if abs(json.loads(s)['t']-t0)<1e-8)
rotors=anchor['geometry']['count']['rotors'];records=d['records']
xs=np.asarray([r['t']-t0 for r in records]);fixed=json.loads((OUT/'causal_extension/results.json').read_text(encoding='utf-8'))
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
for i in range(4):
    values=[r['rotors'][str(i)] for r in records]
    phase=[v['selected']['phase_hz'] if v['valid'] else np.nan for v in values]
    quality=[max(v['modes']['1']['weighted_phase_concentration'],v['modes']['-1']['weighted_phase_concentration']) for v in values]
    axes[0].plot(xs,phase,label=f'Rotor candidate{i}')
    axes[1].plot(xs,quality,label=f'Rotor candidate{i}')
axes[1].axhline(.5,color='black',linestyle='--',label='Frozen phase gate')
for rows,label in [(fixed['records'],'Fixed initial region'),(records,'Moving region')]:
    axes[2].step(xs,[sum(v['valid'] for v in r['rotors'].values()) for r in rows],where='post',label=label)
for ax in axes:
    ax.axvline(.4,color='gray',linestyle=':');ax.grid(alpha=.25);ax.legend(loc='best')
axes[0].set_ylabel('Supported model phase (Hz)');axes[1].set_ylabel('Predictive radial-weighted R')
axes[2].set_ylabel('Supported candidates');axes[2].set_xlabel(f'Seconds since initial conics at{t0}s')
axes[2].set_yticks(range(5));fig.suptitle('Real RAW observation probe; no mechanical RPM / force / flight accuracy truth')
fig.tight_layout();fig.savefig(out/'phase_tracking.png',dpi=150);plt.close(fig)

theta=np.linspace(0,2*np.pi,180);unit=np.c_[np.cos(theta),np.sin(theta),np.ones(len(theta))]
audit=[];tiles=[];writer=cv2.VideoWriter(str(out/'native_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50.,(1060,480))
assert writer.isOpened()
try:
    for k,(t,e) in enumerate(native_batches(d['protocol']['raw'],t0,1.2)):
        frame=np.zeros((720,1280,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(e,frame)
        row=records[k];assert abs(t-row['t'])<1e-8
        centers=[]
        for i,r in enumerate(rotors):
            value=row['rotors'][str(i)]
            if not value['valid']:continue
            H=np.asarray(value['selected']['H']);projected=unit@H.T
            patch=projected[:,:2]/projected[:,2,None]
            a=np.deg2rad(r['angle']);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            curve=((patch-128)*np.array([r['major']/2,r['minor']/2])/80)@rotation.T
            curve+=np.array([r['cx'],r['cy']])+velocity*(t-t0)
            cv2.polylines(frame,[np.rint(curve).astype(np.int32)],True,(0,220,0),1)
            c=np.asarray(value['conditional_axis_image_xy']);centers.append((i,c))
            cv2.drawMarker(frame,tuple(np.rint(c).astype(int)),(0,180,255),cv2.MARKER_CROSS,6,1)
            cv2.putText(frame,f'{i}: {value["selected"]["phase_hz"]:.0f}Hz',tuple(np.rint(c+[0,15]).astype(int)),0,.4,(0,230,0),1)
        minimum=None
        if len(centers)>1:minimum=min(float(np.linalg.norm(a-b)) for i,a in centers for j,b in centers if i<j)
        audit.append(dict(t=t,supported=len(centers),minimum_axis_spacing_px=minimum,axis_xy={str(i):c.tolist() for i,c in centers}))
        crop=cv2.resize(frame[:240,750:1280],(1060,480),interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop,f't={t:.2f}s; green: CURRENT phase-supported model, yellow: conditional axis',(15,390),0,.55,(0,230,255),1)
        cv2.putText(crop,'Native20ms ON/OFF; phase Hz != mechanical rotation Hz; no future geometry',(15,425),0,.55,(0,230,255),1)
        cv2.putText(crop,f'Supported {len(centers)}/4; NOT a flight forecast or real-time performance demo',(15,460),0,.55,(0,230,255),1)
        writer.write(crop)
        if k in [0,19,39,59]:tiles.append(crop)
finally:writer.release()
cv2.imwrite(str(out/'native_overlay_grid.png'),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
(out/'axis_spacing_diagnostic.json').write_text(json.dumps(dict(scope='Self-consistency only; no rotor identity or axis truth',records=audit),indent=2),encoding='utf-8')
print('Saved phase_tracking.png, native_overlay.mp4, native_overlay_grid.png',flush=True)
