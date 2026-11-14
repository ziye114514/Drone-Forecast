"""Fixed RAW polarity snapshots for optional human visibility review only."""
from pathlib import Path
import hashlib,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage107')]
from event_drone_forecast.raw_io import open_raw
from study107 import OUT,SCREEN,rows,save


def main():
    targets=[110.02,111.02,112.02,114.02,116.52,118.02]
    raw=Path('E:/dataset/train/6/media/event.raw');reader=open_raw(raw);h,w=reader.get_size()
    current={round(r['t'],8):r for r in rows(SCREEN/'raw/observations_forecasts.jsonl')}
    folder=OUT/'visibility_review';folder.mkdir(exist_ok=False);records=[];tiles=[]
    for label,t in zip('ABCDEF',targets):
        begin=round((t-.02)*1e6);end=round(t*1e6);reader.seek_time(begin);e=reader.load_delta_t(20000)
        e=e[(e['t']>=begin)&(e['t']<end)];pix=e['y'].astype(np.int32)*w+e['x']
        on=np.bincount(pix[e['p']>0],minlength=w*h).reshape(h,w);off=np.bincount(pix[e['p']<=0],minlength=w*h).reshape(h,w)
        image=np.full((h,w,3),255,np.uint8);image[(on>0)&(off==0)]=(40,80,220);image[(off>0)&(on==0)]=(220,170,20);image[(on>0)&(off>0)]=(15,15,15)
        r=current[round(t,8)];box=r['body_observation'].get('bbox')
        if box is not None:
            lo=np.maximum(np.int32(np.floor(np.asarray(box[:2])-50)),0);hi=np.minimum(np.int32(np.ceil(np.asarray(box[2:])+50)),[w,h])
        else:lo=np.array([0,0]);hi=np.array([w,h])
        crop=image[lo[1]:hi[1],lo[0]:hi[0]];tile=np.full((410,640,3),245,np.uint8)
        scale=min(640/crop.shape[1],350/crop.shape[0]);small=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        ch,cw=small.shape[:2];tile[:ch,:cw]=small
        cv2.putText(tile,f'{label}: t={t:.2f}s, native 20ms ON/OFF',(12,378),0,.65,(20,30,40),1)
        cv2.putText(tile,'No fitted ellipses or predicted motor points shown',(12,402),0,.47,(20,30,40),1)
        cv2.imwrite(str(folder/f'{label}_full.png'),image);cv2.imwrite(str(folder/f'{label}_review.png'),tile);tiles.append(tile)
        records.append(dict(id=label,t=t,event_count=len(e),event_timestamp_max=None if not len(e) else int(e['t'].max()),
            cutoff_us=end,crop=np.r_[lo,hi].tolist(),human_visibility=None,
            inferred_four_from_known_airframe_is_not_visibility=True))
    grid=np.full((820,1920,3),255,np.uint8)
    for j,tile in enumerate(tiles):grid[(j//3)*410:(j//3+1)*410,(j%3)*640:(j%3+1)*640]=tile
    cv2.imwrite(str(folder/'A_F_visibility_review.png'),grid)
    save(folder/'review_protocol107.json',dict(raw=str(raw),raw_bytes=raw.stat().st_size,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=records,render='Native current20ms polarity occupancy: ON red, OFF cyan, both black. No G/intensity reconstruction, predicted points, fitted overlays or future events.',
        request='Optional: A-F each choose four separate rotor disks / partially overlapping / nearly a line / unclear. Count actual visible boundaries, do not infer missing disks from known quadrotor structure.',
        scope='Human visibility sanity review only, not COM/RPM/pose truth, and not inference input. No response required for ongoing algorithm tests.'))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>六帧桨盘可见性核验</title><style>body{max-width:1500px;margin:25px auto;font:18px/1.7 sans-serif;padding:0 18px}img{width:100%}</style><h1>可选：帮忙判断这六帧是否真的看得清四个桨盘</h1><p>请按实际能看到的边界判断：①四桨可分；②部分重叠；③基本成线；④看不清。可以直接回复“A①、B②……”；不需要圈桨心或提供飞控真值，也不要根据知道它是四旋翼而补猜。</p><p>这些是固定时刻的原生20ms ON/OFF：红为ON、青为OFF、黑为两类事件均出现。只有当前检测框扩大后的裁剪，没有拟合椭圆、预测点或G重建。算法实验无需等待这项可选核验。</p><img src="A_F_visibility_review.png"><p>这只核查观测可见性，不是重心、姿态或转速真值。</p></html>'''
    (folder/'六帧桨盘可见性核验.html').write_text(page,encoding='utf-8');print(str(folder),flush=True)


if __name__=='__main__':main()
