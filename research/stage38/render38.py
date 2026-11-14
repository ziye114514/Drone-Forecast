import numpy as np
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from common38 import PARAMS,viewport


def activity_filter(events,w,h):
    algorithm=ActivityNoiseFilterAlgorithm(w,h,PARAMS['activity_threshold_us']);buffer=algorithm.get_empty_output_buffer();algorithm.process_events(events,buffer)
    return buffer.numpy().copy()


def project(events,w,h,d):
    view=viewport(w,h,d);result=events.copy();result['x']=result['x']//d+view[0];result['y']=result['y']//d+view[1]
    return result,view


def render(events,w,h,d):
    e,view=project(events,w,h,d);frame=np.empty((h,w,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(e,frame);return frame,view
