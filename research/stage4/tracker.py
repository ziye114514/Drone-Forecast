"""Exploratory image-plane skeleton tracker; no claim of metric 3-D pose.

Canonical square -> a 2 x 3 affine projection. Three motor correspondences
update six image parameters, NOT a unique calibrated 6-DoF pose. Unseen motors
are predictions only. Genuine image support expires even if propagation works.
"""
import itertools
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

Q = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
X = np.c_[Q, np.ones(4)]
PARAMS = dict(init_frames=5, closure_fraction=.18, min_flatness=.12,
              diameter_ratio=2.5, association_fraction=.18,
              support_expiry_s=.30, rotor_expiry_s=.5, max_flow_s=.30)


def fit_affine(indices, points):
    return np.linalg.lstsq(X[indices], points, rcond=None)[0].T


def flatness(points):
    s = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
    return float(s[-1] / max(s[0], 1e-9))


def initialization(rotors):
    if len(rotors) != 4:
        return None, 'need_four'
    pts = np.array([[r['cx'], r['cy']] for r in rotors])
    diam = np.array([r['major'] for r in rotors])
    order = np.argsort(np.arctan2(*(pts - pts.mean(0))[:, ::-1].T))
    pts = pts[order]
    span = np.linalg.norm(np.ptp(pts, axis=0))
    closure = np.linalg.norm(pts[0]+pts[2]-pts[1]-pts[3]) / max(span, 1.)
    if diam.max()/max(diam.min(), 1) > PARAMS['diameter_ratio']:
        return None, 'unequal_blob_sizes'
    if flatness(pts) < PARAMS['min_flatness']:
        return None, 'near_collinear'
    if closure > PARAMS['closure_fraction']:
        return None, 'not_parallelogram'
    distance = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(4)*1e6
    if distance.min() < .3*np.median(diam) or span < 1.1*np.median(diam):
        return None, 'merged_or_too_small'
    return (pts, fit_affine(np.arange(4), pts), diam[order]), 'candidate'


class Skeleton:
    def __init__(self, body=False, partial=True):
        self.body, self.partial = body, partial
        self.A = None
        self.velocity = np.zeros(2)
        self.last_t = None
        self.last_support = self.last_rotor = None
        self.pending = None
        self.streak = 0
        self.anchor = np.zeros(2)
        self.features = None
        self.previous_image = None
        self.diam = np.ones(4)*20
        self.init_count = 0

    def body_transform(self, image, dt):
        if self.features is None or self.previous_image is None or len(self.features) < 6:
            return None, 0
        p = self.features.astype(np.float32).reshape(-1, 1, 2)
        q, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous_image, image, p, None,
                       winSize=(21,21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 25, .01))
        if q is None: return None, 0
        back, ok2, _ = cv2.calcOpticalFlowPyrLK(image, self.previous_image, q, None,
                                              winSize=(21,21), maxLevel=3)
        if back is None: return None, 0
        good = ok.ravel().astype(bool) & ok2.ravel().astype(bool)
        good &= np.linalg.norm(back.reshape(-1,2)-p.reshape(-1,2), axis=1) < 1.5
        good &= np.linalg.norm(q.reshape(-1,2)-p.reshape(-1,2), axis=1) < max(25, dt*1800)
        if good.sum() < 6: return None, int(good.sum())
        T, mask = cv2.estimateAffinePartial2D(p.reshape(-1,2)[good], q.reshape(-1,2)[good],
                      method=cv2.RANSAC, ransacReprojThreshold=2., maxIters=100,
                      confidence=.99, refineIters=5)
        if T is None or mask.sum() < 6: return None, 0
        scale = np.linalg.norm(T[:,0])
        if not .92 < scale < 1.08: return None, 0
        return T, int(mask.sum())

    def sample_body(self, image, rotors):
        mask = np.zeros_like(image)
        pts = X @ self.A.T
        low = pts.min(0)-[20,15]
        high = pts.max(0)+[20,max(35, .4*np.ptp(pts[:,0]))]
        low = np.maximum(low, 0).astype(int)
        high = np.minimum(high, [image.shape[1]-1,image.shape[0]-1]).astype(int)
        cv2.rectangle(mask, tuple(low), tuple(high), 255, -1)
        # Remove observed spinning disks; remaining support is body/arms/legs
        # plus possible background. This mask alone is not semantic recognition.
        for r in rotors:
            cv2.ellipse(mask, ((float(r['cx']),float(r['cy'])),
                        (float(r['major'])*1.1,float(r['minor'])*1.2),float(r['angle'])),0,-1)
        p = cv2.goodFeaturesToTrack(image, maxCorners=60, qualityLevel=.015,
                                   minDistance=4, mask=mask, blockSize=5)
        self.features = p.reshape(-1,2) if p is not None else None

    def step(self, t, rotors, image=None, original_center=None):
        dt = 0. if self.last_t is None else t-self.last_t
        self.last_t = t
        reason = 'tracking'
        if self.A is not None and t-self.last_support > PARAMS['support_expiry_s']:
            self.A = None; self.streak = 0; self.pending = None
            self.features = None; reason = 'support_expired'
        if self.A is None:
            init, why = initialization(rotors)
            if init is None:
                self.streak = 0; self.pending = None
                return dict(t=t, active=False, mode='uninitialized', reason=why)
            pts, A, diam = init
            center = pts.mean(0)
            if self.pending is not None and np.linalg.norm(center-self.pending) < 50:
                self.streak += 1
            else: self.streak = 1
            self.pending = center
            if self.streak < PARAMS['init_frames']:
                return dict(t=t, active=False, mode='initializing', reason=why, streak=self.streak)
            self.A, self.diam = A, diam
            # Match original image target anchor only at initialization. The
            # evaluation boxes refer to whole aircraft, not its rotor-plane center.
            self.anchor = np.zeros(2)
            if original_center is not None:
                self.anchor = np.clip(np.linalg.pinv(A[:,:2]) @ (original_center-A[:,2]), -.6, .6)
            self.last_support = self.last_rotor = t
            self.velocity[:] = 0
            self.init_count += 1
            mode = 'initialized'; matched = list(range(4)); accepted = 4; flow_count = 0
        else:
            old_center = self.A[:,2].copy()
            predicted = self.A.copy(); predicted[:,2] += self.velocity*dt
            predicted_pts = X @ predicted.T
            span = np.linalg.norm(np.ptp(predicted_pts,axis=0))
            gate = max(10., span*PARAMS['association_fraction'])
            points = np.array([[r['cx'],r['cy']] for r in rotors]).reshape(-1,2)
            ii = jj = np.array([],int)
            if len(points):
                cost = np.linalg.norm(predicted_pts[:,None]-points[None],axis=2)
                ii,jj = linear_sum_assignment(cost)
                valid = cost[ii,jj] < gate; ii,jj = ii[valid],jj[valid]
            accepted = len(ii); matched = ii.tolist(); mode = 'coast'; flow_count = 0
            if len(ii)>=3 and (self.partial or len(ii)==4):
                proposed = fit_affine(ii, points[jj])
                fit = X[ii] @ proposed.T
                error = np.mean(np.linalg.norm(fit-points[jj],axis=1))
                change = np.max(np.linalg.norm(X@proposed.T-predicted_pts,axis=1))
                # Flat projections support translation but cannot warrant a
                # reliable new plane shape. Hidden predictions are never fit data.
                if error < gate*.5 and change < gate*1.7 and flatness(points[jj]) > .08:
                    predicted = proposed; mode = 'rotor_affine'
                    self.last_support = self.last_rotor = t
                else: accepted = 0; reason = 'shape_or_collinearity_rejected'
            elif len(ii)>=2 and self.partial:
                residual = points[jj]-predicted_pts[ii]
                if np.max(np.linalg.norm(residual-np.median(residual,axis=0),axis=1)) < gate*.5:
                    predicted[:,2] += np.median(residual,axis=0)
                    self.last_support = self.last_rotor = t; mode = 'rotor_translation'
                else: accepted = 0
            else: accepted = 0
            if mode == 'coast' and self.body and image is not None and t-self.last_rotor < PARAMS['max_flow_s']:
                T, flow_count = self.body_transform(image, dt)
                if T is not None:
                    predicted = T[:,:2]@self.A; predicted[:,2] += T[:,2]
                    self.last_support = t; mode = 'body_flow'
            self.A = predicted
            if mode != 'coast':
                self.velocity = .7*self.velocity + .3*np.clip((self.A[:,2]-old_center)/max(dt,.001),-2000,2000)
        pts = X @ self.A.T
        if image is not None and self.body:
            self.sample_body(image, rotors); self.previous_image = image.copy()
        return dict(t=t,active=True,mode=mode,reason=reason,points=pts.tolist(),
                    center=(self.A[:,:2]@self.anchor+self.A[:,2]).tolist(),
                    geometric_center=self.A[:,2].tolist(),observed_slots=matched if accepted else [],
                    matched_count=accepted,body_inliers=flow_count,
                    flatness=flatness(pts),since_support_s=t-self.last_support,
                    since_rotor_s=t-self.last_rotor,initializations=self.init_count,
                    metric_pose=False,hidden_points_are_observations=False)
