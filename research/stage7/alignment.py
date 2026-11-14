"""Fixed-time read-only alignment audit. Never fit a time shift from errors."""
from pathlib import Path
import json,zipfile
import cv2,numpy as np
from study import OUT,load,save,read_labels,open_raw,raw_chunks

def main():
    all_panels=[];audit=[]
    for item in load(OUT/'local_inventory.json'):
        i=item['sequence'];labels=read_labels(i);label=min(labels,key=lambda r:abs(r['t']-18.))
        stamp=label['t'];reader=open_raw(item['raw']);h,w=reader.get_size()
        events=[e for t,e in raw_chunks(reader,max(0,stamp-.02),.02,10000)]
        e=np.concatenate(events);pix=e['y'].astype(int)*w+e['x'];density=np.bincount(pix,minlength=h*w).reshape(h,w)
        raw=cv2.cvtColor(np.uint8(np.clip(np.log1p(density)/4,0,1)*255),cv2.COLOR_GRAY2BGR)
        with zipfile.ZipFile(item['archive']) as z:
            names=[n for n in z.namelist() if '/Event/Frames/' in '/'+n and n.endswith('.png')]
            member=min(names,key=lambda n:abs(int(Path(n).stem.split('_')[-1])/1e6-stamp))
            ft=int(Path(member).stem.split('_')[-1])/1e6
            provided=cv2.imdecode(np.frombuffer(z.read(member),np.uint8),cv2.IMREAD_COLOR)
        panels=[]
        for name,im in [('Author event frame',provided),('Native RAW 20ms ending at label',raw)]:
            hh,ww=im.shape[:2];b=np.round(np.array(label['bbox'])*[ww/w,hh/h,ww/w,hh/h]).astype(int)
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(40,240,40),2)
            im=cv2.resize(im,(640,360));cv2.putText(im,f'Seq{i} {name}',(8,24),0,.6,(0,220,255),2)
            cv2.putText(im,f'label={stamp:.6f}s author={ft:.6f}s',(8,48),0,.5,(0,220,255),1);panels.append(im)
        panel=np.hstack(panels);(OUT/f'alignment_sequence{i}.png').write_bytes(cv2.imencode('.png',panel)[1].tobytes())
        all_panels.append(cv2.resize(panel,(1024,288)))
        x0,y0,x1,y1=np.int32(label['bbox']);x0=max(x0,0);y0=max(y0,0);x1=min(x1,w);y1=min(y1,h)
        audit.append(dict(sequence=i,label_t=stamp,author_t=ft,author_timestamp_difference_s=ft-stamp,
            member=member,raw_dimensions=[w,h],provided_dimensions=[provided.shape[1],provided.shape[0]],
            events_inside_label_box=int(density[y0:y1,x0:x1].sum()),total_events=len(e),time_shift_fitted=False,
            status='Fixed-time visual check; render differences and object-identity correctness require inspection, not certified by event counts.'))
        print('alignment',i,flush=True)
    (OUT/'alignment_audit.png').write_bytes(cv2.imencode('.png',np.vstack(all_panels))[1].tobytes())
    save(OUT/'alignment_audit.json',audit)

if __name__=='__main__':main()
