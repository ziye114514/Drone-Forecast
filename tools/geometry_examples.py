"""Render controlled geometry measurements; synthetic, never flight evidence."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.geometry import fit_rotor_geometry,ellipse_points


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    canvas=np.zeros((720,1200,3),np.uint8);canvas[:]=(18,24,32)
    titles=['Complete ellipse','Occluded ellipse: cut edge is not the rotor rim',
            'Near-edge-on footprint: axis only','Very short arc: full shape not established']
    records=[]
    for i,title in enumerate(titles):
        image=np.zeros((280,560),np.uint8)
        expected=None
        if i in (0,1):
            cv2.ellipse(image,(280,145),(105,45),18,0,360,255,-1);expected=[280,145]
            if i==1:image[155:]=0
            contours,_=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            points=contours[0].reshape(-1,2)
        elif i==2:
            cv2.line(image,(150,130),(380,160),255,2)
            contours,_=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            points=contours[0].reshape(-1,2)
        else:
            points=ellipse_points([280,145,105,45,.3],np.linspace(.5,.8,90))
        result=fit_rotor_geometry(points,520.)
        tile=np.zeros((360,600,3),np.uint8);tile[:]=(18,24,32)
        for point in points.astype(int):cv2.circle(tile,tuple(point+[20,40]),1,(240,240,240),-1)
        cv2.putText(tile,title,(15,25),0,.56,(240,240,240),1)
        record=dict(case=title,accepted=result is not None)
        if result:
            mode=result['mode'];record.update(mode=mode,center=result['center'].tolist(),
                    geometry_quality=result['geometry_quality'],center_sigma_px=result['center_sigma_px'])
            if expected is not None:record['center_error_px']=float(np.linalg.norm(result['center']-expected))
            color=(40,230,240) if mode=='line' else (240,140,220) if mode=='arc' else (60,220,70)
            center=result['center']+[20,40]
            if mode=='line':
                angle=np.deg2rad(result['angle']);d=np.array([np.cos(angle),np.sin(angle)])*result['major']/2
                cv2.line(tile,tuple((center-d).astype(int)),tuple((center+d).astype(int)),color,2)
            else:cv2.ellipse(tile,(tuple(center),(result['major'],result['minor']),result['angle']),color,2)
            text=f"{mode} | quality {result['geometry_quality']:.2f} | center sigma {result['center_sigma_px']:.1f}px"
        else:text='Rejected: insufficient support for a reliable shape'
        cv2.putText(tile,text,(15,330),0,.54,(230,230,230),1)
        y,x=divmod(i,2);canvas[y*360:(y+1)*360,x*600:(x+1)*600]=tile
        records.append(record)
    cv2.imwrite(str(args.output/'geometry_synthetic.png'),canvas)
    (args.output/'geometry_synthetic.json').write_text(json.dumps(dict(synthetic=True,cases=records),indent=2),encoding='utf-8')
    print(json.dumps(records))


if __name__=='__main__':main()
