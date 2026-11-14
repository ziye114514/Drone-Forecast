import ast
import cv2,numpy as np
from common37 import OUT
from metavision_sdk_core import BaseFrameGenerationAlgorithm

path=OUT/'upstream/yolo/render_events.py';tree=ast.parse(path.read_text(encoding='utf-8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_frame')
namespace={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),namespace)
upstream_get_frame=namespace['get_frame']


def past(events,cutoff,window):return events[(events['t']>=cutoff-window)&(events['t']<cutoff)]


def render(events,w,h,kind):
    if kind=='upstream_dark':return upstream_get_frame((events['x'],events['y'],events['p']>0),width=w,height=h)
    if kind=='native_count_white':
        pixels=events['y'].astype(np.int32)*w+events['x'];counts=np.bincount(pixels,minlength=w*h).reshape(h,w)
        grey=np.uint8(255-230*np.clip(np.log1p(counts)/np.log(8),0,1));return cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
    if kind=='sdk_default':
        frame=np.empty((h,w,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(events,frame);return frame
    raise ValueError(kind)
