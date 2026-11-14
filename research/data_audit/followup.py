"""Extra visual suitability checks selected from existing time/label catalogues."""
import hashlib,time
import cv2,numpy as np
from audit import OUT,load,save,open_raw,counts,white

PLAN={'train3':[7.,19.65,35.25,49.88],
      'train4':[27.,50.,75.,83.],
      'train6':[28.,32.,36.,40.,104.,106.,112.,114.,116.,118.,119.5]}


if __name__=='__main__':
    base=OUT/'temporal';assert not (base/'complete.json').exists()
    save(base/'plan.json',dict(times_s=PLAN,window_us=10000,
        selection='Follow-up from existing time/annotation catalogue and known clear/side train6 intervals; no model scoring. Fixed before this follow-up decode.',
        warning='Sparse frames cannot establish continuous visibility or same-UAV identity between times.'))
    catalog={r['id']:r for r in load(OUT/'catalog.json')['records']};records=[]
    for key,times in PLAN.items():
        raw=catalog[key]['raw'];reader=open_raw(raw);h,w=reader.get_size();folder=base/key;folder.mkdir(parents=True,exist_ok=True)
        tiles=[];frames=[]
        for t in times:
            lo=round(t*1e6);reader.seek_time(lo);e=reader.load_delta_t(10000);e=e[(e['t']>=lo)&(e['t']<lo+10000)]
            q=counts(e,w,h);im=white(q['on'],q['off']);p=folder/f'raw_onoff_{t:.2f}.png'
            p.write_bytes(cv2.imencode('.png',im)[1].tobytes());np.savez_compressed(folder/f'counts_{t:.2f}.npz',on=q['on'],off=q['off'])
            frames.append(dict(id=key,raw=raw,begin_s=t,end_s=t+.01,events=len(e),
                image=p.relative_to(OUT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            tile=np.full((398,640,3),255,np.uint8);tile[38:]=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA);tile[:38]=25
            cv2.putText(tile,f'{key} t={t:.2f}-{t+.01:.2f}s | {len(e)} events',(8,25),0,.57,(240,240,240),1);tiles.append(tile)
        reader.__del__()
        while len(tiles)%2:tiles.append(np.full_like(tiles[0],255))
        sheet=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
        (folder/'contact.png').write_bytes(cv2.imencode('.png',sheet)[1].tobytes())
        records.append(dict(id=key,frames=frames,contact=(folder/'contact.png').relative_to(OUT).as_posix()))
        print(key+' follow-up complete',flush=True)
    save(base/'complete.json',dict(records=records,G_used=False,detector_used=False))
