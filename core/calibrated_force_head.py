"""Small frozen diagnostic response head, not a Newton thrust identifier."""
import numpy as np
from adaptive_force_memory import kernel


def features_at(rows,query,delay=0.):
    available=[r for r in rows if r['t']<=query+1e-9]
    if not available:return None
    current=available[-1];age=float(query-current['t']);c=current.get('coefficient')
    if not current['valid'] or c is None or age>.04:return None
    h=.4+age;K=float(kernel(-1/.35,h))
    base=np.asarray(c['position'])+h*np.asarray(c['velocity'])+K*np.asarray(c['acceleration'])
    past=[r for r in rows if r['t']<=current['t']-delay+1e-9]
    u=np.zeros(2)
    if current['current_input_available'] and past and past[-1]['u'] is not None and current['t']-delay-past[-1]['t']<=.04:
        u=np.asarray(past[-1]['u'],float)
    return dict(query_t=float(query),source_t=current['t'],h=h,K=K,baseline=base.tolist(),
        input_u=u.tolist(),input_available=current['current_input_available'],branch=current['evidence_branch'])


def fit_head(records,targets,zero=False):
    if len(records)<2:raise ValueError('At least two calibration targets required')
    X=np.array([r['K']*np.asarray(r['input_u']) for r in records]);X[:]=0 if zero else X
    variation=float(np.sqrt(np.mean(np.sum((X-X.mean(0))**2,axis=1))))
    scale=variation if variation>1e-12 else 1.
    A=np.zeros((len(records)*2,3));b=[]
    for i,(r,z) in enumerate(zip(records,targets)):
        A[2*i:2*i+2,0]=X[i]/scale
        A[2*i,1]=1000*r['K'];A[2*i+1,2]=1000*r['K']
        b.extend(np.asarray(z)-r['baseline'])
    if variation<=1e-12:A[:,0]=0.
    precision=A.T@A/36.+np.diag([1/36.,1.,1.]);rhs=A.T@np.asarray(b)/36.
    theta=np.linalg.solve(precision,rhs)
    if theta[0]<0:
        theta[0]=0.;theta[1:]=np.linalg.solve(precision[1:,1:],rhs[1:])
    return dict(gain=float(theta[0]/scale),unknown_bias_acceleration=(1000*theta[1:]).tolist(),
        training_examples=len(records),training_feature_scale_px=variation,prior_gain=0.,
        position_noise_proxy_px=6.,bias_prior_sigma_px_s2=1000.,test_labels_used=False,
        metric_force_identified=False,supervised_calibration_required=True)


def predict_head(record,model,zero=False):
    u=np.zeros(2) if zero else np.asarray(record['input_u'])
    return (np.asarray(record['baseline'])+record['K']*(model['gain']*u+model['unknown_bias_acceleration'])).tolist()
