"""Stage 143 variants: line-direction correction and a reliability-veto baseline.

Two candidates, both bounded and both refusing to invent physics:

  tangent   the 141 line's own direction is the projection axis (the previous
            normal axis was shown to be anti-correlated with observed motion);
            the sign follows the recent observed acceleration along that axis;
            the physical term is added with a maturity-weighted, capped weight.
  veto      no physical term at all. The local-track confidence only scales the
            existing 140 dense acceleration down when the observation is poor.
            This tests the "when should physics participate" question without
            letting physics contribute a direction.
"""
from pathlib import Path
import copy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'research/stage142'), str(ROOT / 'src'), str(ROOT / 'research/stage135')]
from feature_model142 import AxisFeatureModel  # noqa: E402
from predictor135 import forecast  # noqa: E402
from event_drone_forecast.paper_config import PaperConfig  # noqa: E402

MIN_TRACKS = 2
MIN_DOMINANT_CONFIDENCE = 0.5
MIN_MEAN_CONFIDENCE = 0.4
MAX_WEIGHT = 0.5
ACCELERATION_CAP = 400.0
MATURITY_WINDOW = 1.2
MATURED_MARGIN = 0.02
VETO_FLOOR = 0.35


class _Matured:
    """Causal matured-error bookkeeping shared by the variants."""

    def __init__(self):
        self.pending = []
        self.matured = []
        self.centres = []

    def observe(self, t, centre):
        if centre is not None:
            self.centres.append((t, np.asarray(centre, float)))
        self.centres = [item for item in self.centres if item[0] >= t - 2.0]
        still = []
        for target, branch, base in self.pending:
            if target > t - MATURED_MARGIN:
                still.append((target, branch, base))
                continue
            reference = next((value for stamp, value in self.centres
                              if abs(stamp - target) <= .03), None)
            if reference is None:
                continue
            self.matured.append((target, float(np.linalg.norm(branch - reference)),
                                 float(np.linalg.norm(base - reference))))
        self.pending = still
        self.matured = [item for item in self.matured if item[0] >= t - MATURITY_WINDOW]

    def factor(self):
        if not self.matured:
            return .5
        branch = float(np.mean([item[1] for item in self.matured]))
        base = float(np.mean([item[2] for item in self.matured]))
        return float(np.clip(1.2 - branch / max(base, 1e-6), .15, 1.0))


def _gated(row, local, need_line):
    gate = row['predictor'].get('dense_motion_fallback', {})
    axis = row.get('periodic_axis') or {}
    line = axis.get('line') if axis.get('axis_valid') else None
    tracks = sorted((local.get('local_tracks') or []), key=lambda item: -item['hz'])
    ok = bool(local) and local.get('tracks', 0) >= MIN_TRACKS \
        and local.get('dominant_confidence', 0) >= MIN_DOMINANT_CONFIDENCE \
        and local.get('mean_confidence', 0) >= MIN_MEAN_CONFIDENCE \
        and gate.get('eligible') and gate.get('unclipped_fit_window') \
        and not row['predictor']['structure_supported']
    if need_line:
        ok = ok and line is not None
    return ok, line, tracks


class TangentModel:
    """Physical correction projected on the observed line direction."""

    name = 'tangent143'

    def __init__(self):
        self.model = AxisFeatureModel(PaperConfig(), adaptive=True)
        self.reference = None
        self.matured = _Matured()
        self.admitted = 0
        self.active = 0
        self.weights = []
        self.directions = 0

    def step(self, row, local):
        coefficient = row['coefficients'].get('optimized_geometry')
        centre = row['body'].get('center') if row['body'].get('detected') else None
        self.matured.observe(row['t'], centre)
        ok, line, tracks = _gated(row, local, need_line=True)
        direction = None
        s = None
        if ok and coefficient is not None:
            normal = np.asarray(line['normal'], float)
            direction = np.array([-normal[1], normal[0]])
            diagnostic = row['predictor'].get('acceleration') or {}
            recent = np.asarray((diagnostic.get('acceleration') if isinstance(diagnostic, dict)
                                 else diagnostic) or coefficient.get('acceleration', [0., 0.]), float)[:2]
            if float(recent @ direction) < 0:
                direction = -direction
            total = float(np.sum(np.square([track['hz'] for track in tracks[:4]])))
            if self.reference is None:
                self.reference = total
            s = total / self.reference
            ok = .05 <= s <= 50.0
            self.directions += 1
        if coefficient is not None:
            state = np.r_[coefficient['position'], coefficient['velocity']]
            self.model.update(row['t'], state, np.r_[direction, 0.] if ok else np.r_[0., -1., 0.],
                              s if s is not None else 1.0,
                              float(min(1.0, local.get('mean_confidence', 0.) / .8)) if ok else 0.0,
                              row['body'].get('detected', False), ok)
            self.admitted += bool(ok)
        active = bool(ok and coefficient is not None and self.model.updates >= 10
                      and self.model.last_update is not None
                      and row['t'] - self.model.last_update <= .26)
        corrected = copy.deepcopy(coefficient) if coefficient is not None else None
        weight = 0.0
        if corrected is not None and active:
            a_model = np.asarray(self.model.acceleration(np.r_[direction, 0.], s,
                                                         np.asarray(coefficient['velocity'])))[:2]
            a_base = np.asarray(coefficient.get('acceleration', [0., 0.]), float)
            delta = np.clip(a_model - a_base, -ACCELERATION_CAP, ACCELERATION_CAP)
            weight = MAX_WEIGHT * min(1.0, local.get('mean_confidence', 0.) / .8) * self.matured.factor()
            corrected['acceleration'] = (a_base + weight * delta).tolist()
            self.active += 1
            self.weights.append(weight)
        if corrected is not None:
            for horizon in (.1, .2, .4):
                self.matured.pending.append((row['t'] + horizon,
                                             np.asarray(forecast(corrected, horizon), float),
                                             np.asarray(forecast(coefficient, horizon), float)))
        return dict(t=row['t'], coefficient=corrected, active=active,
                    gated=ok, weight=weight, direction='line_tangent' if ok else 'none',
                    local_tracks=len(tracks), maturity=self.matured.factor(),
                    updates=self.model.updates, relative_rate_squared=s,
                    actual_mechanical_RPM_or_thrust=False)


class VetoModel:
    """No physical direction: poor local evidence only shrinks the 140 acceleration."""

    name = 'veto143'

    def __init__(self):
        self.applied = 0
        self.scales = []

    def step(self, row, local):
        coefficient = row['coefficients'].get('optimized_geometry')
        ok, _line, tracks = _gated(row, local, need_line=False)
        corrected = copy.deepcopy(coefficient) if coefficient is not None else None
        scale = 1.0
        if corrected is not None and ok:
            confidence = float(local.get('mean_confidence', 0.))
            scale = float(np.clip(VETO_FLOOR + (1.0 - VETO_FLOOR) * min(1.0, confidence / .8),
                                  VETO_FLOOR, 1.0))
            acceleration = np.asarray(coefficient.get('acceleration', [0., 0.]), float)
            corrected['acceleration'] = (acceleration * scale).tolist()
            self.applied += 1
            self.scales.append(scale)
        return dict(t=row['t'], coefficient=corrected, active=bool(ok), gated=ok,
                    weight=1.0 - scale, direction='none' if ok else 'unused',
                    local_tracks=len(tracks), maturity=None, updates=0,
                    relative_rate_squared=None, actual_mechanical_RPM_or_thrust=False)


class RobustVelocityModel:
    """Keep the 140 state but replace a jump-polluted velocity with a robust slope.

    The largest 0.4 s errors sit at high speed and at re-acquisition, where a single
    body-box jump can push the filtered velocity far from the recent trend. The
    slope is a median over pairwise differences inside a 0.3 s window of already
    observed body centres, so one bad centre cannot move it.
    """

    name = 'robustv143'

    def __init__(self, window=.30, disagreement=150.0, one_sided=False, ratio=1.2):
        self.centres = []
        self.window = window
        self.disagreement = disagreement
        self.one_sided = one_sided
        self.ratio = ratio
        self.replaced = 0
        self.jumps = 0

    def _slope(self, t):
        self.centres = [item for item in self.centres if item[0] >= t - self.window]
        if len(self.centres) < 4:
            return None
        times = np.array([item[0] for item in self.centres])
        points = np.array([item[1] for item in self.centres])
        slopes = []
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                if times[j] - times[i] > .05:
                    slopes.append((points[j] - points[i]) / (times[j] - times[i]))
        if len(slopes) < 3:
            return None
        return np.median(np.array(slopes), axis=0)

    def step(self, row, local):
        coefficient = row['coefficients'].get('optimized_geometry')
        centre = row['body'].get('center') if row['body'].get('detected') else None
        if centre is not None:
            self.centres.append((row['t'], np.asarray(centre, float)))
        corrected = copy.deepcopy(coefficient) if coefficient is not None else None
        active = False
        if corrected is not None and centre is not None:
            slope = self._slope(row['t'])
            velocity = np.asarray(corrected.get('velocity', [0., 0.]), float)
            if slope is not None:
                gap = float(np.linalg.norm(slope - velocity))
                speed_slope = float(np.linalg.norm(slope))
                speed_filtered = float(np.linalg.norm(velocity))
                # One-sided mode never slows the filter down: a lagging velocity is
                # the failure we can act on, a filter that already tracks a hard
                # acceleration must be left alone.
                admissible = (not self.one_sided) or (speed_slope > self.ratio * max(speed_filtered, 1.0))
                if gap > max(self.disagreement, 0.5 * speed_slope) and admissible:
                    corrected['velocity'] = slope.tolist()
                    active = True
                    self.replaced += 1
                self.jumps += int(gap > self.disagreement)
        return dict(t=row['t'], coefficient=corrected, active=active, gated=True,
                    weight=None, direction='robust_slope' if active else 'kept',
                    local_tracks=len(local.get('local_tracks') or []) if local else 0,
                    maturity=None, updates=0, relative_rate_squared=None,
                    actual_mechanical_RPM_or_thrust=False)
