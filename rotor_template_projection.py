"""Five-circle geometric PRIOR for the camera's visible-envelope readout.

Four CURRENT fitted conics initialise a template; the fifth point is their
centre. Later qualified headless plane/visible rotor scale can transport it.
Hidden conics are projections, never new measurements or force inputs. In-plane
yaw/depth changes not observed in the current data remain conditional priors.
"""
import copy
import numpy as np
from conic_observability import usable
from axis_amplitude_bridge import current_axis_qualified
from native_fov_observation import EnvelopeMemory,project


def ellipse_bounds(e):
    x,y,A,B,C,D=e
    return np.array([x-np.hypot(A,B),y-np.hypot(C,D),x+np.hypot(A,B),y+np.hypot(C,D)])


def ellipse_visible_bounds(e,width=1280.,height=720.):
    x,y,A,B,C,D=map(float,e);M=np.array([[A,B],[C,D]]);points=[]
    def add(t):
        p=np.array([x+A*np.cos(t)+B*np.sin(t),y+C*np.cos(t)+D*np.sin(t)])
        if -.000001<=p[0]<=width+.000001 and -.000001<=p[1]<=height+.000001:points.append(np.clip(p,[0,0],[width,height]))
    for u,v in [(A,B),(C,D)]:
        phase=np.arctan2(v,u);add(phase);add(phase+np.pi)
    for boundary in (0.,width):
        radius=np.hypot(A,B)
        if radius and abs((boundary-x)/radius)<=1:
            phase=np.arctan2(B,A);arc=np.arccos(np.clip((boundary-x)/radius,-1,1));add(phase+arc);add(phase-arc)
    for boundary in (0.,height):
        radius=np.hypot(C,D)
        if radius and abs((boundary-y)/radius)<=1:
            phase=np.arctan2(D,C);arc=np.arccos(np.clip((boundary-y)/radius,-1,1));add(phase+arc);add(phase-arc)
    for p in [(0.,0.),(0.,height),(width,0.),(width,height)]:
        if np.linalg.norm(np.linalg.pinv(M)@(np.asarray(p)-[x,y]))<=1+1e-10:points.append(np.asarray(p))
    if not points:return None
    p=np.asarray(points);return np.r_[p.min(0),p.max(0)]


def shape_matrix(theta,ratio):
    c,s=np.cos(theta),np.sin(theta);R=np.array([[c,-s],[s,c]])
    return R@np.diag([1.,ratio])@R.T


class ConicTemplateMemory:
    def __init__(self):self.envelope=EnvelopeMemory();self.template=None;self.stamp=None;self.observations=None;self.last=None
    def step(self,row):
        info=self.envelope.step(row);obs=row.get('state',{}).get('observations');t=row['t']
        if obs is not None and self.observations is not None and obs<self.observations or self.last is not None and t-self.last>.5:self.template=None
        self.observations=obs;self.last=t
        body=row['body'];box=body.get('bbox');conics=[q for q in row['geometry'].get('rotors',[]) if usable(q)]
        complete=bool(box and body.get('detected') and not body.get('center_provisional') and min(box[:2])>0 and box[2]<1279 and box[3]<719)
        if complete and len(conics)==4 and row['structure'].get('mode')=='four':
            centers=np.array([[q['cx'],q['cy']] for q in conics]);hull=__import__('cv2').convexHull(centers.astype(np.float32))
            if len(hull)==4:
                theta=np.deg2rad([q['angle'] for q in conics]);angle=.5*np.arctan2(np.sin(2*theta).mean(),np.cos(2*theta).mean());ratio=float(np.median([q['minor']/q['major'] for q in conics]));ell=[]
                for q in conics:
                    z=np.deg2rad(q['angle']);c,s=np.cos(z),np.sin(z);a=q['major']/2;b=q['minor']/2
                    ell.append([q['cx'],q['cy'],a*c,-b*s,a*s,b*c])
                bounds=np.array([ellipse_bounds(e) for e in ell]);middle=(bounds[:,:2].min(0)+bounds[:,2:].max(0))/2
                for e in ell:e[0]-=middle[0];e[1]-=middle[1]
                self.template=dict(ellipses=ell,shape=shape_matrix(angle,ratio).tolist(),major=float(np.median([q['major'] for q in conics])),theta=angle,ratio=ratio)
                self.stamp=t
        if self.template is None:return info
        temp=copy.deepcopy(self.template);scale=1.;S=np.asarray(temp['shape'])
        if conics:
            scale=float(np.median([q['major'] for q in conics]))/temp['major']
        n=np.asarray(row['structure'].get('axis_xy') or [],float)
        if current_axis_qualified(row['structure']) and n.shape==(2,) and np.linalg.norm(n)>1e-6:
            theta=np.arctan2(n[1],n[0])+np.pi/2;ratio=np.sqrt(max(0.,1-min(float(n@n),1.)))
            S=shape_matrix(theta,ratio)
        normal=row.get('normal_turn_forecast') or {};trend=normal.get('trend') or {}
        temp.update(current_shape=S.tolist(),current_scale=scale,omega=float(trend.get('shrunk_angular_rate_rad_s',0.)) if normal.get('applied') else 0.)
        info.update(rotor_template=temp,template_source_t=self.stamp,five_circle_template_is_prior_not_new_detection=True,
            in_plane_yaw_unobserved=True,hidden_rotor_force_created=False)
        return info


class TemplateVisibleResponse:
    def __init__(self):self.memory=ConicTemplateMemory()
    def step(self,row):
        info=self.memory.step(row);r=copy.deepcopy(row)
        if r.get('coefficient') and info['full_extent_px'] is not None:
            r['native_fov_observation']=info
            from physical_publication import evaluate
            r['physical']=evaluate(r,[.1,.2,.4]);r['output_reference']='predicted_visible_native_rotor_envelope_center'
        return r


def projected(point,info,h):
    if not info.get('rotor_template'):return project(point,info['full_extent_px'])
    temp=info['rotor_template'];angle=temp['omega']*h;c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s],[s,c]])
    S=R@np.asarray(temp['current_shape'])@R.T;H=temp['current_scale']*S@np.linalg.inv(temp['shape']);ell=[]
    for e in temp['ellipses']:
        q=H@e[:2];M=H@np.array([[e[2],e[3]],[e[4],e[5]]]);u,sv,vt=np.linalg.svd(M)
        # Same2.5px minor-radius resolution as the current conic fitter.
        M=u@np.diag(np.maximum(sv,2.5))@vt;ell.append([*q,M[0,0],M[0,1],M[1,0],M[1,1]])
    full=np.array([ellipse_bounds(e) for e in ell]);center=(full[:,:2].min(0)+full[:,2:].max(0))/2;shift=np.asarray(point)-center
    visible=[]
    for e in ell:
        e[0]+=shift[0];e[1]+=shift[1];b=ellipse_visible_bounds(e)
        if b is not None:visible.append(b)
    if not visible:return np.asarray(point),dict(predicted_visible=False,unclipped_point_retained=True,rotor_template=True)
    b=np.array(visible);box=np.r_[b[:,:2].min(0),b[:,2:].max(0)]
    return (box[:2]+box[2:])/2,dict(predicted_visible=True,projected_visible_bbox=box.tolist(),rotor_template=True)
