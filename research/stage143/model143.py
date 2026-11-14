"""Stage 143 model v2: direction from the observed line, weight from matured errors.

Two changes over v1, both required by the plan rather than tuned on the endpoints:

  1. DIRECTION. v1 passed a constant image-up normal, so the learned term was just
     a vertical acceleration proportional to the summed squared rates. Now the
     normal comes from the current 141 line when that line is valid, and the
     correction is refused when no direction is observed. The sign stays unknown,
     so the term is applied with the sign that matches the recent observed
     acceleration, and its uncertainty is carried by the weight.

  2. WEIGHT FROM MATURED ERROR. The correction is multiplied by how well this
     branch has actually predicted over the last --maturity_window seconds, using
     only predictions whose horizon has already matured and the *observed* body
     centre as reference (no labels, no future truth). When the branch has been
     worse than the baseline, its weight collapses towards the baseline.

Everything else (admission, RLS arithmetic, 140 baseline) is unchanged.
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

PARAMS = dict(
    feature='Local stage143 tracks: sum of squared rates over gated tracks, relative to the first admitted value.',
    direction='141 current line normal when axis_valid (unsigned). No constant direction; no line means no correction.',
    admission='>=2 tracks, dominant confidence>=.5, mean confidence>=.4, 140 dense fallback eligible and '
              'unclipped, no structure support, valid 141 line.',
    correction='a = a_base + w*(a_model - a_base), |delta|<=400 px/s^2, w<=.5.',
    weight='w = .5*min(1, mean_conf/.8) * maturity, maturity = clip(1.2 - err_branch/err_base, .15, 1) '
           'over matured anchors in the last 1.2 s; .5 before any matured anchor.',
    sign='Chosen to agree with the recent observed acceleration component along the normal.',
    interpretation='Bounded image-space correction; parameters are not mass, wind or thrust coefficients.')

MIN_TRACKS = 2
MIN_DOMINANT_CONFIDENCE = 0.5
MIN_MEAN_CONFIDENCE = 0.4
MAX_WEIGHT = 0.5
ACCELERATION_CAP = 400.0
MATURITY_WINDOW = 1.2
MATURED_MARGIN = 0.02


class Model:
    def __init__(self):
        self.model = AxisFeatureModel(PaperConfig(), adaptive=True)
        self.reference = None
        self.admitted = 0
        self.active = 0
        self.pending = []          # (t_target, prediction_branch, prediction_base)
        self.matured = []          # (t_target, err_branch, err_base)
        self.history_centres = []  # (t, centre) for matured comparison
        self.weight_used = []

    # ---- matured-error bookkeeping -------------------------------------
    def _observe(self, t, centre):
        if centre is not None:
            self.history_centres.append((t, np.asarray(centre, float)))
        self.history_centres = [item for item in self.history_centres if item[0] >= t - 2.0]
        still = []
        for t_target, branch, base in self.pending:
            if t_target > t - MATURED_MARGIN:
                still.append((t_target, branch, base))
                continue
            reference = next((c for stamp, c in self.history_centres
                              if abs(stamp - t_target) <= .03), None)
            if reference is None:
                continue
            self.matured.append((t_target,
                                 float(np.linalg.norm(branch - reference)),
                                 float(np.linalg.norm(base - reference))))
        self.pending = still
        self.matured = [item for item in self.matured if item[0] >= t - MATURITY_WINDOW]

    def maturity(self):
        if not self.matured:
            return .5
        error_branch = float(np.mean([item[1] for item in self.matured]))
        error_base = float(np.mean([item[2] for item in self.matured]))
        return float(np.clip(1.2 - error_branch / max(error_base, 1e-6), .15, 1.0))

    # ---- main step ------------------------------------------------------
    def step(self, row, local):
        coefficient = row['coefficients'].get('optimized_geometry')
        gate = row['predictor'].get('dense_motion_fallback', {})
        axis = row.get('periodic_axis') or {}
        line = axis.get('line') if axis.get('axis_valid') else None
        normal = np.r_[np.asarray(line['normal'], float), 0.] if line else None
        body_centre = (row['body'].get('center') if row['body'].get('detected') else None)
        self._observe(row['t'], body_centre)

        tracks = sorted((local.get('local_tracks') or []), key=lambda item: -item['hz'])
        gated = bool(local) and local.get('tracks', 0) >= MIN_TRACKS \
            and local.get('dominant_confidence', 0) >= MIN_DOMINANT_CONFIDENCE \
            and local.get('mean_confidence', 0) >= MIN_MEAN_CONFIDENCE \
            and gate.get('eligible') and gate.get('unclipped_fit_window') \
            and not row['predictor']['structure_supported'] and normal is not None
        s = None
        quality = 0.0
        if gated:
            total = float(np.sum(np.square([track['hz'] for track in tracks[:4]])))
            if self.reference is None:
                self.reference = total
            s = total / self.reference
            quality = float(min(1.0, local['mean_confidence'] / 0.8))
            gated = .05 <= s <= 50.0

        if coefficient is not None:
            state = np.r_[coefficient['position'], coefficient['velocity']]
            self.model.update(row['t'], state, normal if normal is not None else np.r_[0., -1., 0.],
                              s if s is not None else 1.0, quality,
                              row['body'].get('detected', False), gated)
            self.admitted += bool(gated)

        active = bool(gated and coefficient is not None and self.model.updates >= 10
                      and self.model.last_update is not None
                      and row['t'] - self.model.last_update <= .26)
        weight = 0.0
        corrected = copy.deepcopy(coefficient) if coefficient is not None else None
        if corrected is not None and active:
            a_model = np.asarray(self.model.acceleration(normal, s, np.asarray(coefficient['velocity'])))[:2]
            a_base = np.asarray(coefficient.get('acceleration', [0., 0.]), float)
            # sign of the physical term is unobservable; align it with the recent
            # observed acceleration along the same normal instead of image-up.
            diagnostic = row['predictor'].get('acceleration') or {}
            recent = np.asarray((diagnostic.get('acceleration') if isinstance(diagnostic, dict)
                                 else diagnostic) or a_base, float)[:2]
            projection = float(np.dot(recent, normal[:2]))
            if projection < 0:
                a_model = -a_model
            delta = np.clip(a_model - a_base, -ACCELERATION_CAP, ACCELERATION_CAP)
            weight = MAX_WEIGHT * min(1.0, local['mean_confidence'] / .8) * self.maturity()
            corrected['acceleration'] = (a_base + weight * delta).tolist()
            self.active += 1
            self.weight_used.append(weight)

        if corrected is not None:
            for horizon in (.1, .2, .4):
                self.pending.append((row['t'] + horizon,
                                     np.asarray(forecast(corrected, horizon), float),
                                     np.asarray(forecast(coefficient, horizon), float)))
        return dict(t=row['t'], coefficient=corrected, active=active, gated=gated,
                    relative_rate_squared=s, local_tracks=len(tracks),
                    mean_confidence=local.get('mean_confidence', 0.),
                    maturity=self.maturity(), weight=weight,
                    direction_from='141_line' if normal is not None else 'none',
                    updates=self.model.updates, reference_sum_squared=self.reference,
                    actual_mechanical_RPM_or_thrust=False)
