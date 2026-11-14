"""Stage 14: 2D motion models for pixel-domain drone prediction.

Same input for every model: the published centre of one frozen run, tick by tick.
Every model is a causal recursive filter; predictions are recorded when they are
made and scored later against the observation that actually arrived, so no model
sees the future.

  CV    constant velocity Kalman filter (the stage7 style baseline)
  CA    constant acceleration Kalman filter (the "history acceleration" baseline)
  CTRV  constant turn rate and velocity EKF, 2D image-plane heading
        (state [x, y, v, psi, psi_dot], the Target_KF EKF_CTRV reduced to 2D)
  IMM   interacting multiple model over CV and CTRV with model probabilities
        (transition matrix and likelihood update as in Target_KF's IMM)

Scoring uses the dense observed centre as reference; that reference is the
detector's own output at a later time, not an independent ground truth, so this
measures extrapolation quality, not absolute accuracy. Sparse manual boxes are
handled separately.
"""
from pathlib import Path
import argparse
import json
import math

import numpy as np

HORIZONS = (0.1, 0.2, 0.4)
MATCH_WINDOW = 0.03           # a reference observation must be within this of t+h
SIGMA_ACCEL = 400.0           # px/s^2 process noise for velocity
SIGMA_TURN_ACCEL = 3.0        # rad/s^2 process noise for turn rate
SIGMA_MEAS = 6.0              # px measurement noise
MIN_HISTORY = 5               # observations required before an anchor is judged
SMALL_TURN = 0.05             # rad/s below which CTRV uses the straight-line branch
ACCEL_WINDOW_S = 0.35         # history window of the stage7-style acceleration baseline


def load_series(path, method):
    source = path / f'{method}.jsonl'
    if not source.exists():
        source = path / 'observations.jsonl'      # stage12 layout
    rows = [json.loads(line) for line in
            source.read_text(encoding='utf-8').splitlines() if line.strip()]
    series = []
    for row in rows:
        observation = row['observation']
        detected = bool(observation['detected'])
        center = observation.get('center') if detected else None
        series.append((float(row['t']), None if center is None else (float(center[0]), float(center[1]))))
    return series


class CVFilter:
    name = 'cv'

    def __init__(self):
        self.x = None
        self.P = np.eye(4) * 1e4

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        q = SIGMA_ACCEL ** 2
        Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                          [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                          [dt ** 3 / 2, 0, dt ** 2, 0],
                          [0, dt ** 3 / 2, 0, dt ** 2]], float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def initialize(self, z, velocity=None):
        self.x = np.array([z[0], z[1], 0., 0.]) if velocity is None else np.array(
            [z[0], z[1], velocity[0], velocity[1]])

    def update(self, z):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * SIGMA_MEAS ** 2
        y = np.array(z) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def predict_at(self, horizon):
        return np.array([self.x[0] + self.x[2] * horizon, self.x[1] + self.x[3] * horizon])

    def speed(self):
        return float(np.hypot(self.x[2], self.x[3]))


class CAFilter(CVFilter):
    name = 'ca'

    def __init__(self):
        self.x = None
        self.P = np.eye(6) * 1e4

    def predict(self, dt):
        F = np.eye(6)
        F[0, 2] = dt
        F[0, 4] = dt ** 2 / 2
        F[1, 3] = dt
        F[1, 5] = dt ** 2 / 2
        F[2, 4] = dt
        F[3, 5] = dt
        q = (SIGMA_ACCEL * 2.0) ** 2
        Q = np.zeros((6, 6))
        for i, j, power in ((0, 4, 4), (1, 5, 4), (2, 4, 2), (3, 5, 2), (4, 4, 2), (5, 5, 2)):
            Q[i, j] = q * dt ** power / (power * 2)
        Q = Q + Q.T - np.diag(np.diag(Q))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def initialize(self, z, velocity=None):
        self.x = np.array([z[0], z[1], 0., 0., 0., 0.]) if velocity is None else np.array(
            [z[0], z[1], velocity[0], velocity[1], 0., 0.])

    def update(self, z):
        H = np.zeros((2, 6))
        H[0, 0] = 1.
        H[1, 1] = 1.
        R = np.eye(2) * SIGMA_MEAS ** 2
        y = np.array(z) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def predict_at(self, horizon):
        return np.array([self.x[0] + self.x[2] * horizon + 0.5 * self.x[4] * horizon ** 2,
                         self.x[1] + self.x[3] * horizon + 0.5 * self.x[5] * horizon ** 2])

    def speed(self):
        return float(np.hypot(self.x[2], self.x[3]))


class CTRVFilter:
    """2D constant turn rate and velocity, Target_KF's EKF_CTRV without z."""

    name = 'ctrv'

    def __init__(self):
        self.x = None
        self.P = np.eye(5) * 1e4

    def initialize(self, z, velocity=None):
        speed = 0.0 if velocity is None else float(np.hypot(velocity[0], velocity[1]))
        heading = 0.0 if velocity is None or speed <= 1e-6 else math.atan2(velocity[1], velocity[0])
        self.x = np.array([z[0], z[1], speed, heading, 0.0])
        # A position-only measurement barely observes the heading, so start from a
        # finite prior instead of an uninformative one; a huge P made the EKF
        # linearise around a meaningless heading for seconds.
        self.P = np.diag([SIGMA_MEAS ** 2, SIGMA_MEAS ** 2, 1e4, 0.3 ** 2, 0.5 ** 2])

    def transition(self, x, dt):
        px, py, v, psi, omega = x
        out = np.array(x, float)
        if abs(omega) > SMALL_TURN:
            out[0] = px + v * (math.sin(psi + omega * dt) - math.sin(psi)) / omega
            out[1] = py + v * (-math.cos(psi + omega * dt) + math.cos(psi)) / omega
        else:
            out[0] = px + v * dt * math.cos(psi)
            out[1] = py + v * dt * math.sin(psi)
        out[3] = psi + omega * dt
        out[3] = (out[3] + math.pi) % (2 * math.pi) - math.pi
        return out

    def jacobian(self, x, dt):
        px, py, v, psi, omega = x
        J = np.eye(5)
        if abs(omega) > SMALL_TURN:
            J[0, 2] = (math.sin(psi + omega * dt) - math.sin(psi)) / omega
            J[0, 3] = v * (math.cos(psi + omega * dt) - math.cos(psi)) / omega
            J[0, 4] = (v * dt * math.cos(psi + omega * dt) / omega
                       - v * (math.sin(psi + omega * dt) - math.sin(psi)) / omega ** 2)
            J[1, 2] = (-math.cos(psi + omega * dt) + math.cos(psi)) / omega
            J[1, 3] = v * (math.sin(psi + omega * dt) - math.sin(psi)) / omega
            J[1, 4] = (v * dt * math.sin(psi + omega * dt) / omega
                       - v * (-math.cos(psi + omega * dt) + math.cos(psi)) / omega ** 2)
        else:
            J[0, 2] = dt * math.cos(psi)
            J[0, 3] = -v * dt * math.sin(psi)
            J[1, 2] = dt * math.sin(psi)
            J[1, 3] = v * dt * math.cos(psi)
        J[3, 4] = dt
        return J

    def predict(self, dt):
        F = self.jacobian(self.x, dt)
        q = np.diag([0., 0., (SIGMA_ACCEL * dt) ** 2, 0., (SIGMA_TURN_ACCEL * dt) ** 2])
        self.x = self.transition(self.x, dt)
        self.P = F @ self.P @ F.T + q

    def update(self, z):
        H = np.zeros((2, 5))
        H[0, 0] = 1.
        H[1, 1] = 1.
        R = np.eye(2) * SIGMA_MEAS ** 2
        y = np.array(z) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P
        self.innovation = float(y @ np.linalg.inv(S) @ y)
        self.innovation_cov = S

    def se2_covariance(self):
        """Position/velocity covariance projected to the 4-state SE2 layout used by IMM."""
        px, py, v, psi, omega = self.x
        J = np.zeros((4, 5))
        J[0, 0] = 1.
        J[1, 1] = 1.
        J[2, 2] = math.cos(psi)
        J[2, 3] = -v * math.sin(psi)
        J[3, 2] = math.sin(psi)
        J[3, 3] = v * math.cos(psi)
        return J @ self.P @ J.T, J

    def load_se2(self, mean4, cov4):
        px, py, vx, vy = mean4
        speed = float(np.hypot(vx, vy))
        psi = math.atan2(vy, vx) if speed > 1e-6 else 0.0
        self.x = np.array([px, py, speed, psi, self.x[4] if self.x is not None else 0.0])
        self.P = np.eye(5) * 1e4
        self.P[:2, :2] = cov4[:2, :2]
        self.P[2, 2] = max(cov4[2, 2] + cov4[3, 3], 1.0)
        self.P[3, 3] = max(cov4[2, 2] + cov4[3, 3], 1e-2) / max(speed ** 2, 1.0)

    def predict_at(self, horizon):
        state = self.transition(self.x, horizon)
        return np.array([state[0], state[1]])

    def speed(self):
        return float(abs(self.x[2]))


class IMMFilter:
    """CV + CTRV with model probabilities, following Target_KF's IMM structure."""

    name = 'imm'

    def __init__(self, stay=0.9):
        self.cv = CVFilter()
        self.ctrv = CTRVFilter()
        self.mu = np.array([0.5, 0.5])
        self.transition = np.array([[stay, 1 - stay], [1 - stay, stay]])
        self.mu_history = []

    def initialize(self, z, velocity=None):
        self.cv.initialize(z, velocity)
        self.ctrv.initialize(z, velocity)

    def predict(self, dt):
        self.cv.predict(dt)
        self.ctrv.predict(dt)

    def update(self, z):
        predicted = self.transition @ self.mu
        likelihoods = np.zeros(2)
        for index, model in enumerate((self.cv, self.ctrv)):
            H = np.zeros((2, 4 if index == 0 else 5))
            H[0, 0] = 1.
            H[1, 1] = 1.
            innovation = np.array(z) - H @ model.x
            S = H @ model.P @ H.T + np.eye(2) * SIGMA_MEAS ** 2
            likelihoods[index] = math.exp(-0.5 * innovation @ np.linalg.inv(S) @ innovation) / (
                math.sqrt((2 * math.pi) ** 2 * max(np.linalg.det(S), 1e-9)))
        total = float(predicted @ likelihoods)
        if total <= 0 or not math.isfinite(total):
            self.mu = predicted
        else:
            self.mu = predicted * likelihoods / total
        self.mu = np.clip(self.mu, 1e-3, 1 - 1e-3)
        self.mu = self.mu / self.mu.sum()
        self.cv.update(z)
        self.ctrv.update(z)
        self.mu_history.append((float(self.mu[0]), float(self.mu[1])))

    def predict_at(self, horizon):
        cv_point = self.cv.predict_at(horizon)
        ctrv_point = self.ctrv.predict_at(horizon)
        return self.mu[0] * cv_point + self.mu[1] * ctrv_point

    def speed(self):
        return self.cv.speed()


class HistoryAccelCV(CVFilter):
    """CV filter plus an acceleration term estimated from the recent past.

    This mirrors the stage7 "history acceleration" arm, which is the strong
    baseline there: the acceleration is a finite difference over a 0.35 s window
    of already-published positions, never a future value.
    """

    name = 'cv_hist_acc'

    def __init__(self):
        super().__init__()
        self.history = []
        self.now = 0.0

    def predict(self, dt):
        self.now += dt
        super().predict(dt)

    def update(self, z):
        super().update(z)
        self.history.append((self.now, np.array(z, float)))
        self.history = [item for item in self.history if item[0] >= self.now - ACCEL_WINDOW_S]

    def acceleration(self):
        if len(self.history) < 6:
            return np.zeros(2)
        times = np.array([item[0] for item in self.history])
        points = np.array([item[1] for item in self.history])
        times = times - times.mean()
        design = np.column_stack([np.ones_like(times), times, 0.5 * times ** 2])
        coefficients, *_ = np.linalg.lstsq(design, points, rcond=None)
        return coefficients[2]

    def predict_at(self, horizon):
        base = super().predict_at(horizon)
        return base + 0.5 * self.acceleration() * horizon ** 2


def build_filters():
    return [CVFilter(), CAFilter(), CTRVFilter(), IMMFilter(), HistoryAccelCV()]


def first_fix(model):
    return model.cv if isinstance(model, IMMFilter) else model


def run(series, method, label):
    filters = build_filters()
    pending = []          # (t_anchor, horizon, model_index, prediction)
    errors = {model.name: {h: [] for h in HORIZONS} for model in filters}
    anchors = {model.name: {h: 0 for h in HORIZONS} for model in filters}
    updates = 0
    observations = 0
    last_t = None
    previous_fix = None
    for t, center in series:
        dt = None if last_t is None else max(t - last_t, 1e-3)
        velocity = None
        if center is not None and previous_fix is not None and dt:
            velocity = ((center[0] - previous_fix[0]) / dt, (center[1] - previous_fix[1]) / dt)
        for model in filters:
            if first_fix(model).x is None and center is not None:
                model.initialize(center, velocity)
                continue
            if first_fix(model).x is None:
                continue
            if dt is not None:
                model.predict(dt)
            if center is not None:
                model.update(center)
        if center is not None:
            observations += 1
            previous_fix = center
        if first_fix(filters[0]).x is not None and dt is not None:
            updates += 1
            for model_index, model in enumerate(filters):
                for horizon in HORIZONS:
                    pending.append((t, horizon, model_index, model.predict_at(horizon)))
        if center is not None:
            keep = []
            for anchor_t, horizon, model_index, prediction in pending:
                target = anchor_t + horizon
                if t + MATCH_WINDOW < target:
                    keep.append((anchor_t, horizon, model_index, prediction))
                    continue
                if abs(t - target) <= MATCH_WINDOW and updates >= MIN_HISTORY:
                    error = float(np.hypot(prediction[0] - center[0], prediction[1] - center[1]))
                    name = filters[model_index].name
                    errors[name][horizon].append(error)
                    anchors[name][horizon] += 1
            pending = keep
        last_t = t
    summary = {}
    for model in filters:
        summary[model.name] = {}
        for horizon in HORIZONS:
            values = sorted(errors[model.name][horizon])
            if not values:
                summary[model.name][str(horizon)] = None
                continue
            summary[model.name][str(horizon)] = dict(
                n=len(values), median=float(np.median(values)),
                mean=float(np.mean(values)),
                p90=float(values[int(0.9 * (len(values) - 1))]))
    mu = None
    imm = next((model for model in filters if isinstance(model, IMMFilter)), None)
    if imm and imm.mu_history:
        history = np.array(imm.mu_history)
        mu = dict(mean_cv=float(history[:, 0].mean()), mean_ctrv=float(history[:, 1].mean()),
                  final_cv=float(history[-1, 0]), final_ctrv=float(history[-1, 1]))
    return dict(label=label, method=method, updates=updates, observations=observations,
                models=summary, imm_model_probabilities=mu)


def main():
    global HORIZONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--out', default=None)
    parser.add_argument('--horizons', default='0.1,0.2,0.4')
    args = parser.parse_args()
    HORIZONS = tuple(float(item) for item in args.horizons.split(',') if item.strip())
    run_dir = Path(args.run).resolve()
    series = load_series(run_dir, args.method)
    report = dict(stage=14, run=str(run_dir), method=args.method, label=args.label,
                  horizons=list(HORIZONS), match_window_s=MATCH_WINDOW,
                  noise=dict(sigma_accel=SIGMA_ACCEL, sigma_turn_accel=SIGMA_TURN_ACCEL,
                             sigma_meas=SIGMA_MEAS),
                  note='Reference is the detector output at t+h, not independent ground truth; '
                       'this scores extrapolation, not absolute accuracy.')
    report.update(run(series, args.method, args.label))
    destination = Path(args.out).resolve() if args.out else run_dir / f'motion_{args.method}.json'
    if destination.exists():
        raise SystemExit('refusing to overwrite an existing motion report')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding='utf-8')
    print(json.dumps(dict(label=args.label or run_dir.name, method=args.method,
                          observations=report['observations'],
                          medians={name: {h: (None if value is None else round(value['median'], 1))
                                          for h, value in horizons.items()}
                                   for name, horizons in report['models'].items()},
                          imm=report['imm_model_probabilities'],
                          written=str(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
