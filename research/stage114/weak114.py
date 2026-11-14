"""Small fixed native-count snapshot test of seeded weak-boundary recovery."""
from pathlib import Path
import argparse,copy,datetime,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage113'),str(ROOT/'research/stage112'),str(ROOT/'research/stage109')]
import support113 as geom
from capture112 import OUT as SUPPORT
from audit109 import OUT as AUDIT,save,sha,rows
OUT=ROOT.parent/'output/stage114_weak_boundary_pilot_20261022'
PARAMS=dict(times=[110.02,111.02,112.02,114.02,116.52,118.02,22.5,.5],
    weak_density_fraction=.5,maximum_ellipse_radius_factor=1.5,
    seed='Existing observed selected contour interiors, eroded3x3. No new seeds or forced four disks.',
    surface='Inverse clipped bipolar event density for standard cv2.watershed, not intensity/G reconstruction.',
    admission='Same existing contour ellipse shape checks and104 conic gates; no inferred missing rotor. Boundary recovery is diagnostic until support/geometry stability shown.')

def sources():return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'research/stage113/support113.py',ROOT/'src/event_drone_forecast/geometry.py']}
def freeze():
    assert not OUT.exists();save(OUT/'protocol114.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),parameters=PARAMS,
        scope='Eight pre-existing fixed native20 count snapshots and exact112 observed contours. No newRAW, future labels, forecast, neural segmentation or physical-edge truth. C and all other frames already reviewed development examples.',
        reason='113 robust raw threshold-contour fit worsened clear forecast and reduced qualifications; test whether strong activity cores truncate physical rotor support before replacing the measurement model.',
        design='Standard marker-controlled watershed from existing measured supports into half-threshold counts, bounded by1.5x original ellipse. Compare raw footprint/fit, retain all fixed failures; not new-method originality.',default_promoted=False))

def recover(density,threshold,origin,rotors,supports):
    h,w=density.shape;yy,xx=np.indices((h,w));xy=np.stack([xx,yy],axis=-1)+origin;regions=[]
    for r in rotors:
        delta=xy-[r['cx'],r['cy']];a=np.deg2rad(r['angle']);u=delta[...,0]*np.cos(a)+delta[...,1]*np.sin(a);v=-delta[...,0]*np.sin(a)+delta[...,1]*np.cos(a)
        regions.append((u/(r['major']/2*1.5))**2+(v/(r['minor']/2*1.5))**2<=1)
    if not regions:return [],dict(reason='no_measured_seeds'),np.ones(density.shape,np.int32)
    allowed=np.any(regions,axis=0);weak=(density>=threshold*.5)&allowed;markers=np.zeros((h,w),np.int32);markers[~weak]=1
    seeded=[]
    for i,(r,p) in enumerate(zip(rotors,supports)):
        if len(p)<6:continue
        seed=np.zeros((h,w),np.uint8);cv2.fillPoly(seed,[np.rint(np.asarray(p)-origin).astype(np.int32)],255)
        seed=cv2.erode(seed,np.ones((3,3),np.uint8));seed=(seed>0)&weak&regions[i]
        if seed.sum()<3:continue
        markers[seed]=i+2;seeded.append(i)
    surface=cv2.cvtColor(np.uint8(255-np.clip(density/threshold,0,1)*255),cv2.COLOR_GRAY2BGR);cv2.watershed(surface,markers)
    result=[];audit=[]
    for i in seeded:
        mask=np.uint8((markers==i+2)&weak)*255;cc,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not cc:continue
        contour=max(cc,key=cv2.contourArea).reshape(-1,2)+origin
        if len(contour)<6:continue
        r=rotors[i];ellipse=cv2.fitEllipseDirect(np.float32(contour));(cx,cy),(aa,bb),angle=ellipse
        major=max(aa,bb);minor=min(aa,bb);angle=angle+(90 if aa<bb else 0);fill=cv2.contourArea(np.float32(contour))/max(np.pi*major*minor/4,1)
        valid=bool(10<=major<=r['major']*1.7 and minor>=3 and minor/max(major,1)>=.12 and .4<fill<1.6
            and np.linalg.norm(np.array([cx,cy])-[r['cx'],r['cy']])<=max(18.,r['major']*.65))
        audit.append(dict(rotor=i,valid=valid,major=float(major),minor=float(minor),fill=float(fill),
            center_shift_px=float(np.linalg.norm(np.array([cx,cy])-[r['cx'],r['cy']])),points=len(contour)))
        if valid:
            q=dict(r,cx=float(cx),cy=float(cy),major=float(major),minor=float(minor),angle=float(angle),ratio=float(minor/major),
                geometry_quality=min(r['geometry_quality'],float(np.exp(-abs(fill-1)))))
            result.append((q,contour))
    return result,dict(reason='current_watershed_support',seeded=len(seeded),recovered=len(result),per_rotor=audit),markers

def run():
    p=json.loads((OUT/'protocol114.json').read_text('utf-8'));assert p['sources']==sources();results=[];panels=[]
    lookup={case:{round(r['t'],8):r for r in rows(SUPPORT/case/'supports.jsonl')} for case in ['continuous6','clear6','negative5']}
    for t in PARAMS['times']:
        case='continuous6' if t>100 else 'clear6' if t>20 else 'negative5';r=lookup[case][round(t,8)]
        a=np.load(AUDIT/case/f'{t:.2f}_counts.npz');rotors=r['geometry'].get('rotors',[])
        measured,audit,markers=recover(a['density'],float(a['threshold']),a['origin'],rotors,r['support_points'])
        output=dict(t=t,case=case,original_count=len(rotors),recovered_count=len(measured),recovery=audit)
        if len(rotors)==4 and all(len(z)>5 for z in r['support_points']) and r['body'].get('detected'):
            oldfit=geom.support_fit([geom.geometry58.from_rotor(z) for z in rotors],use_analytic=True)
            output['original_fit']=oldfit
        if len(measured)==4 and r['body'].get('detected'):
            rr=[q[0] for q in measured];support=geom.supports_for([q[1] for q in measured]);fit=geom.support_fit([geom.geometry58.from_rotor(z) for z in rr],support)
            output['recovered_measurement']=geom.measure_fitted(rr,r['body'],fit)
        on=a['on'];off=a['off'];base=np.full((*on.shape,3),255,np.uint8);base[(on>0)&(off==0)]=(40,80,220);base[(off>0)&(on==0)]=(220,170,20);base[(on>0)&(off>0)]=(15,15,15)
        for i,(rotor,points) in enumerate(measured):
            pc=np.rint(points-a['origin']).astype(np.int32);cv2.polylines(base,[pc],True,(30,170,30),1)
        tile=np.full((260,420,3),245,np.uint8);scale=min(410/base.shape[1],220/base.shape[0]);im=cv2.resize(base,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);tile[:im.shape[0],:im.shape[1]]=im
        cv2.putText(tile,f'{t:.2f}s seed={len(rotors)} recovered={len(measured)}',(4,242),0,.5,(20,35,50),1);panels.append(tile)
        results.append(output)
    grid=np.full((520,1680,3),245,np.uint8)
    for i,tile in enumerate(panels):grid[i//4*260:(i//4+1)*260,i%4*420:(i%4+1)*420]=tile
    cv2.imwrite(str(OUT/'weak_boundary_contact114.png'),grid);save(OUT/'results114.json',results)
    print(json.dumps([dict(t=r['t'],old=r['original_count'],recovered=r['recovered_count'],reason=r.get('recovered_measurement',{}).get('reason')) for r in results]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
