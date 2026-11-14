from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'core/run_optimized_forecast.py'
s=p.read_text(encoding='utf8')
out=ROOT.parent/'output/stage149_inertia_scale_20261024'
out.mkdir(exist_ok=True)
backup=out/'run_optimized_before149.py'
if not backup.exists(): backup.write_text(s,encoding='utf8')
def change(before,after):
    global s
    assert s.count(before)==1,(before,s.count(before))
    s=s.replace(before,after)
change("    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}",
       "    paths += [ROOT/'research/stage149'/name for name in ['inertia149.py','dynamics149.py','observer149.py']]\n    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}")
change("rotor_temporal=False):","rotor_temporal=False,rotor_inertia=False,preview_window_ms=20.):")
change("    rotor_curvature = rotor_curvature or rotor_temporal","    rotor_temporal = rotor_temporal or rotor_inertia\n    if not 0 < preview_window_ms <= 20: raise ValueError('Preview window must be in (0,20] ms')\n    rotor_curvature = rotor_curvature or rotor_temporal")
change("        branch_observer=BranchObserver(w,h,rotor_threshold,rotor_adaptive)",
"""        if rotor_inertia:
            sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage149'))
            from observer149 import BranchObserver,PARAMS as INERTIA_PARAMS
            BRANCH_PARAMS=dict(BRANCH_PARAMS,inertia149=INERTIA_PARAMS)
        branch_observer=BranchObserver(w,h,rotor_threshold,rotor_adaptive)""")
change("stage=148 if rotor_temporal else","stage=149 if rotor_inertia else (148 if rotor_temporal else")
change("(147 if rotor_curvature else (146 if rotor_branches else 143)),rotor_temporal=",
       "(147 if rotor_curvature else (146 if rotor_branches else 143))),rotor_inertia=rotor_inertia,preview_window_ms=preview_window_ms,detector_window_ms=20.,rotor_temporal=")
change("                if video is not None:\n                    if branch_measurement",
"""                if video is not None:
                    display_image=image
                    if preview_window_ms<20:
                        display_image=np.zeros_like(image)
                        display_events=clean[clean['t']>=round(t*1e6-preview_window_ms*1000)]
                        BaseFrameGenerationAlgorithm.generate_frame(display_events,display_image)
                    if n%50==0:
                        (out/'native_frames').mkdir(exist_ok=True)
                        cv2.imwrite(str(out/'native_frames'/f'{n:06d}.png'),display_image)
                    if branch_measurement""")
change("render_panel(image,branch_observer","render_panel(display_image,branch_observer")
change("im[:,:w]=image","im[:,:w]=display_image")
change("                    lines=[f'RAW t={t:.2f}s'",
"""                    if rotor_inertia and branch_measurement:
                        for coast in branch_measurement['inertia149']['coasting_tracks']:
                            center=tuple(np.rint(coast['center']).astype(int))
                            cv2.circle(im,center,7,(150,150,150),1)
                            cv2.putText(im,'coast v'+str(coast['visual_track_id']),(center[0]+8,center[1]),0,.35,(150,150,150),1)
                    lines=[f'RAW t={t:.2f}s'""")
change("'50fps playback, not realtime']",
       "f'Preview {preview_window_ms:g}ms; detector20ms','50fps playback, not realtime']")
change("['Orange: fused1480.4s; cyan: visual IDs'",
       "['Orange: fused'+('149' if rotor_inertia else '148')+'0.4s; cyan: visual IDs'")
change("completion=dict(frames=n,raw_only=True","completion=dict(frames=n,rotor_inertia=rotor_inertia,preview_window_ms=preview_window_ms,raw_only=True")
change("p.add_argument('--rotor-temporal'",
       "p.add_argument('--rotor-inertia',action='store_true',help='149 inertial coast/relock and relative-scale projected dynamics; implies --rotor-temporal.');p.add_argument('--preview-window-ms',type=float,default=20.,help='Display-only trailing RAW window; detector/period estimation retain20ms.');p.add_argument('--rotor-temporal'")
change("a.rotor_curvature,a.rotor_temporal)","a.rotor_curvature,a.rotor_temporal,a.rotor_inertia,a.preview_window_ms)")
compile(s,str(p),'exec')
p.write_text(s,encoding='utf8')
print('149 main entry patched; default144 preserved')
