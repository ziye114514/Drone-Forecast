"""Stage 11: rotor identity - corrected orientation audit and association lab.

Runs entirely on an existing stage10 JSONL run: no RAW decode, no re-inference,
and no change to any frozen stage8/stage9/src file.

Two things happen here.

1. CORRECTED SWAP CRITERION. Stage 10 flagged a "rotor swap" when the
   best-fitting correspondence between two consecutive four-rotor ticks was an
   odd permutation. That criterion is wrong: of the four cyclic rotations of a
   planar quadrilateral, two are even and two are odd, so permutation parity
   says nothing about rigid realisability. The invariant that a planar rigid
   motion cannot change is the ORIENTATION (sign of the polygon area) of the
   points taken in a fixed label order. A label assignment that flips that sign
   on a non-degenerate quad is impossible for a rigid body and is therefore a
   genuine exchange of two rotor identities.

2. ASSOCIATION LAB. Three labelers are compared on identical tick sequences:
   serialized order (arbitrary, lower bound), per-rotor constant-velocity
   Hungarian matching (what the v0.3 RotorSensing branch does), and a rigid
   tracker that picks the orientation-preserving correspondence which best fits
   one planar similarity per tick.
   Evaluation is label-free: consecutive four-rotor sets are chained into a
   common body frame using the geometry only, so a labeler is good when each
   identity keeps a stable position inside that frame.
"""
from pathlib import Path
import argparse
import itertools
import json
import math

import numpy as np

PERMS4 = list(itertools.permutations(range(4)))
PERMS3 = list(itertools.permutations(range(3)))
PERMS2 = [(0, 1), (1, 0)]
PERMS = {2: PERMS2, 3: PERMS3, 4: PERMS4}

RIGID_TOL_FLOOR = 6.0
RIGID_TOL_FRACTION = 0.06
CV_GATE_FLOOR = 20.0
CV_GATE_FRACTION = 0.75
TRACK_EXPIRY_S = 0.30
DEGENERACY_RATIO = 0.02
RUPTURE_FRACTION = 0.35


def similarity_residual(a, b):
    """Residual of the best rotation + uniform scale + translation mapping a to b."""
    za = a[:, 0] + 1j * a[:, 1]
    zb = b[:, 0] + 1j * b[:, 1]
    design = np.column_stack([za, np.ones_like(za)])
    solution, *_ = np.linalg.lstsq(design, zb, rcond=None)
    predicted = design @ solution
    return float(np.sqrt(np.mean(np.abs(predicted - zb) ** 2)))


def signed_area(points):
    """Orientation of a polygon taken in the given vertex order."""
    center = points.mean(axis=0)
    delta = points - center
    rolled = np.roll(delta, -1, axis=0)
    return float(np.sum(delta[:, 0] * rolled[:, 1] - delta[:, 1] * rolled[:, 0]))


def anisotropy(points):
    """Smallest/largest covariance eigenvalue: 0 means collinear."""
    if len(points) < 2:
        return 0.0
    values = np.linalg.eigvalsh(np.cov(points.T))
    return float(max(values[0], 0.0) / max(values[-1], 1e-9))


def best_assignment(previous, current, orientation_only=False):
    """Best correspondence for equal-size sets; optionally orientation preserving."""
    count = len(previous)
    permutations = PERMS[count]
    reference_sign = math.copysign(1.0, signed_area(np.asarray(previous)))
    best = None
    for order in permutations:
        ordered = np.asarray(current)[list(order)]
        if orientation_only:
            sign = math.copysign(1.0, signed_area(ordered))
            if sign != reference_sign:
                continue
        residual = similarity_residual(np.asarray(previous), ordered)
        if best is None or residual < best[0]:
            best = (residual, list(order))
    return best


class Track:
    __slots__ = ('id', 'position', 'velocity', 'last_t', 'birth_t', 'seen', 'major', 'local')

    def __init__(self, identifier, position, t, major):
        self.id = identifier
        self.position = np.asarray(position, float)
        self.velocity = np.zeros(2)
        self.last_t = t
        self.birth_t = t
        self.seen = t
        self.major = float(major)
        self.local = []


class BaseTracker:
    """Per-rotor constant-velocity Hungarian matching (the incumbent scheme)."""

    name = 'cv_hungarian'

    def __init__(self, expiry=TRACK_EXPIRY_S):
        self.tracks = []
        self.next_id = 0
        self.fallback_ticks = 0
        self.expiry = expiry
        self.revivals = 0

    def _expire(self, t):
        self.tracks = [track for track in self.tracks if t - track.seen <= self.expiry]

    def _birth(self, candidates, t, used):
        created = {}
        for index, rotor in enumerate(candidates):
            if index in used or len(self.tracks) >= 4:
                continue
            track = Track(self.next_id, rotor['center'], t, rotor['major'])
            self.tracks.append(track)
            self.next_id += 1
            created[index] = track.id
        return created

    def labels(self, previous_rotors, t):
        candidates = [dict(center=np.asarray(r, float), major=0.0) for r in previous_rotors]
        return self._associate(candidates, t)

    def _associate(self, candidates, t):
        raise NotImplementedError


class CVTracker(BaseTracker):
    def _associate(self, candidates, t):
        # Labels are returned per candidate index: None means this measurement was
        # not attached to any identity.
        assigned = [None] * len(candidates)
        if self.tracks and candidates:
            cost = np.zeros((len(self.tracks), len(candidates)))
            for i, track in enumerate(self.tracks):
                predicted = track.position + track.velocity * (t - track.last_t)
                for j, candidate in enumerate(candidates):
                    cost[i, j] = np.linalg.norm(predicted - candidate['center'])
            from scipy.optimize import linear_sum_assignment
            rows, columns = linear_sum_assignment(cost)
            used = set()
            for i, j in zip(rows, columns):
                if cost[i, j] < max(CV_GATE_FLOOR, CV_GATE_FRACTION * max(self.tracks[i].major, 8.0)):
                    track = self.tracks[i]
                    dt = max(t - track.last_t, 1e-3)
                    track.velocity = .8 * track.velocity + .2 * np.clip(
                        (candidates[j]['center'] - track.position) / dt, -2000, 2000)
                    track.position = candidates[j]['center']
                    track.last_t = t
                    track.seen = t
                    assigned[j] = track.id
                    used.add(j)
        else:
            used = set()
        self._expire(t)
        for index, identifier in self._birth(candidates, t, used).items():
            assigned[index] = identifier
        return assigned


class RigidTracker(BaseTracker):
    """One planar similarity carried across ticks; orientation-preserving matching.

    The motion model is a single similarity (rotation + uniform scale +
    translation) shared by the whole quad, not four independent constant-velocity
    rotors. When exactly four rotors are seen on both ticks the correspondence is
    chosen by exhaustive search over the orientation-preserving permutations; in
    mixed-count ticks the last similarity predicts the tracks and a Hungarian
    match assigns them.
    """

    name = 'rigid_orientation'

    def __init__(self):
        super().__init__()
        self.transform = np.eye(3)
        self.rejected_flips = 0
        self.degenerate_accepts = 0
        self.transform_updates = 0

    def _predict(self, track, t):
        point = self.transform @ np.array([track.position[0], track.position[1], 1.0])
        return point[:2] / max(point[2], 1e-9)

    def _fit(self, source, target):
        za = source[:, 0] + 1j * source[:, 1]
        zb = target[:, 0] + 1j * target[:, 1]
        design = np.column_stack([za, np.ones_like(za)])
        solution, *_ = np.linalg.lstsq(design, zb, rcond=None)
        alpha, beta = solution
        return np.array([[alpha.real, -alpha.imag, beta.real],
                         [alpha.imag, alpha.real, beta.imag],
                         [0.0, 0.0, 1.0]])

    def _associate(self, candidates, t):
        live = [track for track in self.tracks if t - track.seen <= TRACK_EXPIRY_S]
        spacing = 30.0
        if len(live) >= 2:
            points = np.array([track.position for track in live])
            spacing = float(np.median([np.linalg.norm(points[i] - points[j])
                                       for i, j in itertools.combinations(range(len(points)), 2)]))
        tolerance = max(RIGID_TOL_FLOOR, RIGID_TOL_FRACTION * spacing)
        # The looser gate only decides whether a measurement may update an existing
        # track; it is deliberately the same size as the incumbent's gate so the
        # comparison is about the motion model, not about gate tuning.
        gate = max(CV_GATE_FLOOR, CV_GATE_FRACTION * max(
            [track.major for track in live] or [8.0]))
        assigned = [None] * len(candidates)
        used = set()
        if live and candidates:
            four_to_four = len(live) == 4 and len(candidates) == 4
            assignment = None
            if four_to_four:
                previous_points = np.array([track.position for track in live])
                current_points = np.array([candidate['center'] for candidate in candidates])
                admissible = best_assignment(previous_points, current_points, orientation_only=True)
                unrestricted = best_assignment(previous_points, current_points, orientation_only=False)
                if admissible is not None and unrestricted is not None:
                    if unrestricted[1] != admissible[1]:
                        if anisotropy(previous_points) > DEGENERACY_RATIO:
                            self.rejected_flips += 1
                        else:
                            self.degenerate_accepts += 1
                if admissible is not None and admissible[0] <= tolerance:
                    assignment = list(zip(range(4), admissible[1]))
            if assignment is None:
                from scipy.optimize import linear_sum_assignment
                predicted = np.array([self._predict(track, t) for track in live])
                cost = np.linalg.norm(predicted[:, None, :] - np.array(
                    [candidate['center'] for candidate in candidates])[None, :, :], axis=2)
                rows, columns = linear_sum_assignment(cost)
                assignment = [(int(i), int(j)) for i, j in zip(rows, columns)
                              if cost[i, j] <= gate]
            matched_source = []
            matched_target = []
            for track_index, candidate_index in assignment:
                track = live[track_index]
                candidate = candidates[candidate_index]
                matched_source.append(track.position)
                matched_target.append(candidate['center'])
                dt = max(t - track.last_t, 1e-3)
                track.velocity = .8 * track.velocity + .2 * np.clip(
                    (candidate['center'] - track.position) / dt, -2000, 2000)
                track.position = candidate['center']
                track.last_t = t
                track.seen = t
                track.major = candidate['major']
                assigned[candidate_index] = track.id
                used.add(candidate_index)
            if len(matched_source) >= 3:
                self.transform = self._fit(np.array(matched_source), np.array(matched_target))
                self.transform_updates += 1
            elif len(matched_source) == 1:
                delta = np.array(matched_target[0]) - np.array(matched_source[0])
                self.transform = np.array([[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]], [0.0, 0.0, 1.0]])
        self._expire(t)
        for index, identifier in self._birth(candidates, t, used).items():
            assigned[index] = identifier
        return assigned


class SerialTracker(BaseTracker):
    """Arbitrary serialized order: the lower bound, not a real method."""

    name = 'serialized_order'

    def _associate(self, candidates, t):
        # Serialized order: label = candidate index, so labels are aligned by
        # construction and only the geometric order itself is being judged.
        if not self.tracks or not candidates:
            self._expire(t)
            self._birth(candidates, t, set())
        return list(range(len(candidates)))


class ShapeMemoryTracker(CVTracker):
    """Incumbent association plus identity memory across detection gaps.

    While at least three rotors are tracked, their configuration is remembered in
    a body frame (relative geometry only). When the tracker comes back after a gap
    the new candidate set is matched to that remembered configuration by shape
    under a planar similarity, preserving chirality, so the same four identities
    are revived instead of four fresh ones being minted. This is the single-target
    assumption of this project: the recording contains one quadrotor.
    """

    name = 'cv_shape_memory'

    def __init__(self, expiry=1.5, memory_s=6.0, stale_gap_s=0.5):
        super().__init__(expiry=expiry)
        self.memory = None
        self.memory_t = None
        self.memory_s = memory_s
        self.stale_gap_s = stale_gap_s
        self.revival_log = []

    def _remember(self, tracks, t):
        live = [track for track in tracks if abs(track.seen - t) < 1e-9]
        if len(live) < 3:
            return
        points = np.array([track.position for track in live])
        center = points.mean(axis=0)
        local = points - center
        spacing = float(np.median([np.linalg.norm(points[i] - points[j])
                                   for i, j in itertools.combinations(range(len(points)), 2)]))
        self.memory = dict(points=local, spacing=max(spacing, 1e-6),
                           ids=[track.id for track in live],
                           major=[track.major for track in live])
        self.memory_t = t

    def _revive(self, candidates, t):
        if self.memory is None or self.memory_t is None or t - self.memory_t > self.memory_s:
            return None
        model = self.memory['points']
        count = len(model)
        if count < 3 or len(candidates) != count:
            return None
        image = np.array([candidate['center'] for candidate in candidates])
        tolerance = max(RIGID_TOL_FLOOR, RIGID_TOL_FRACTION * self.memory['spacing'])
        model_area = math.copysign(1.0, signed_area(model))
        best = None
        for order in PERMS[count]:
            ordered = image[list(order)]
            if math.copysign(1.0, signed_area(ordered)) != model_area:
                continue
            residual = similarity_residual(model, ordered)
            if best is None or residual < best[0]:
                best = (residual, list(order))
        if best is None or best[0] > tolerance:
            return None
        created = []
        for slot, candidate_index in enumerate(best[1]):
            track = Track(self.memory['ids'][slot], candidates[candidate_index]['center'], t,
                          self.memory['major'][slot])
            track.birth_t = t
            self.tracks.append(track)
            created.append((candidate_index, track.id))
        self.revivals += 1
        self.revival_log.append(dict(
            t=t, gap_s=None if self.memory_t is None else t - self.memory_t,
            residual=best[0], tolerance=tolerance,
            spacing=self.memory['spacing'], count=count,
            ratio=best[0] / max(tolerance, 1e-6)))
        self._remember(self.tracks, t)
        return created

    def _associate(self, candidates, t):
        self._expire(t)
        fresh = [track for track in self.tracks if abs(track.seen - t) < 1e-9]
        stale = all(t - track.seen > self.stale_gap_s for track in self.tracks) if self.tracks else True
        if not fresh and stale and len(candidates) >= 3:
            # After a real gap the old positions are stale, so identity is decided
            # by the remembered relative geometry instead of by dead reckoning.
            self.tracks = []
            revived = self._revive(candidates, t)
            self._expire(t)
            if revived:
                assigned = [None] * len(candidates)
                for index, identifier in revived:
                    assigned[index] = identifier
                return assigned
        assigned = super()._associate(candidates, t)
        self._remember(self.tracks, t)
        return assigned


def body_frame_chain(ticks):
    """Chain label-free similarity fits so every tick can be mapped to its segment start.

    A time gap longer than one publication interval starts a new reference frame;
    local coordinates from different segments are never mixed.
    """
    chain = {}
    transform = np.eye(3)
    previous_points = None
    previous_t = None
    segment = 0
    for tick in ticks:
        points = np.asarray(tick['centers'], float)
        gap = None if previous_t is None else tick['t'] - previous_t
        if gap is not None and gap > 0.05:
            transform = np.eye(3)
            previous_points = None
            segment += 1
        if previous_points is not None and len(previous_points) == len(points):
            best = best_assignment(previous_points, points, orientation_only=True)
            if best is not None:
                residual, order = best
                ordered = points[list(order)]
                za = previous_points[:, 0] + 1j * previous_points[:, 1]
                zb = ordered[:, 0] + 1j * ordered[:, 1]
                design = np.column_stack([za, np.ones_like(za)])
                solution, *_ = np.linalg.lstsq(design, zb, rcond=None)
                alpha, beta = solution
                step = np.array([[alpha.real, -alpha.imag, beta.real],
                                 [alpha.imag, alpha.real, beta.imag],
                                 [0.0, 0.0, 1.0]])
                transform = step @ transform
                chain[tick['t']] = (transform.copy(), residual, segment)
                previous_points = points
                previous_t = tick['t']
                continue
        chain[tick['t']] = (transform.copy(), 0.0, segment)
        previous_points = points
        previous_t = tick['t']
    return chain


def evaluate_labels(ticks, labels_by_tick, chain):
    """Label-free body-frame consistency per identity, within each segment.

    Two numbers per labeler: how far an identity wanders from its own median
    position in the chained body frame, and how often it jumps by more than 0.6
    rotor spacings, which is what a swapped identity looks like (the corner it
    occupies in the body frame changes).
    """
    samples = {}
    scale = tick_scale(ticks)
    for tick in ticks:
        key = tick['t']
        if key not in chain or key not in labels_by_tick:
            continue
        transform, residual, segment = chain[key]
        if residual > max(RIGID_TOL_FLOOR, RIGID_TOL_FRACTION * tick['spacing'] * 2):
            continue
        inverse = np.linalg.inv(transform)
        for label, point in zip(labels_by_tick[key], tick['centers']):
            if label is None:
                continue
            homogeneous = np.array([point[0], point[1], 1.0])
            local = inverse @ homogeneous
            samples.setdefault((segment, label), []).append(
                (tick['t'], local[:2] / max(local[2], 1e-9)))
    seen = 0
    spreads = []
    jumps = []
    jump_events = 0
    lifetimes = []
    for points in samples.values():
        if len(points) < 5:
            continue
        lifetimes.append(len(points))
        times = [item[0] for item in points]
        array = np.asarray([item[1] for item in points])
        median = np.median(array, axis=0)
        spreads.append(float(np.median(np.linalg.norm(array - median, axis=1))))
        seen += len(points)
        for index in range(1, len(array)):
            if times[index] - times[index - 1] > 0.05:
                continue
            step = float(np.linalg.norm(array[index] - array[index - 1]))
            jumps.append(step)
            if step > 0.6 * scale:
                jump_events += 1
    jumps.sort()
    return dict(judged_ticks=seen, identities=len(samples), identity_jumps=jump_events,
                identity_jump_fraction=jump_events / max(len(jumps), 1),
                median_local_spread_px=float(np.median(spreads)) if spreads else None,
                median_step_px=float(jumps[len(jumps) // 2]) if jumps else None,
                p90_step_px=float(jumps[int(0.9 * (len(jumps) - 1))]) if jumps else None,
                median_identity_lifetime_ticks=float(np.median(lifetimes)) if lifetimes else None,
                max_identity_lifetime_ticks=int(max(lifetimes)) if lifetimes else None)


def tick_scale(ticks):
    spacings = [tick['spacing'] for tick in ticks if tick['spacing'] > 0]
    return float(np.median(spacings)) if spacings else 30.0


def four_rotor_ticks(rows):
    ticks = []
    for row in rows:
        centers = row['state']['previous_rotors']
        if len(centers) != 4:
            continue
        points = np.asarray(centers, float)
        spacing = float(np.median([np.linalg.norm(points[i] - points[j])
                                   for i, j in itertools.combinations(range(4), 2)]))
        ticks.append(dict(t=row['t'], centers=[list(map(float, p)) for p in points],
                          spacing=spacing, anisotropy=anisotropy(points)))
    return ticks


def orientation_audit(ticks):
    """Corrected swap criterion: orientation flip between consecutive ticks."""
    flips = []
    for previous, current in zip(ticks, ticks[1:]):
        if current['t'] - previous['t'] > 0.05:
            continue
        a = np.asarray(previous['centers'], float)
        b = np.asarray(current['centers'], float)
        identity_area = signed_area(b)
        previous_area = signed_area(a)
        scale = max(previous['spacing'], 1e-6)
        degenerate = (anisotropy(b) <= DEGENERACY_RATIO
                      or abs(identity_area) / (scale * scale) < 0.05)
        if degenerate:
            continue
        if math.copysign(1.0, identity_area) != math.copysign(1.0, previous_area):
            residual, order = best_assignment(a, b, orientation_only=True)
            flips.append(dict(t=current['t'], residual_preserving=residual,
                              order=list(order), anisotropy=anisotropy(b)))
    return flips


def run_labels(tracker, stream, wanted_times):
    """Every tick is fed to the tracker so identities survive partial observations.

    `stream` carries the measured rotor centres of every tick (including empty
    ticks, which let tracks expire naturally); labels are kept only for the
    four-rotor ticks the audit scores.
    """
    labels = {}
    for stamp, centers in stream:
        assigned = tracker.labels(centers, stamp)
        if stamp in wanted_times:
            labels[stamp] = assigned
    return labels


def analyze(run, method, report):
    source = run / f'{method}.jsonl'
    if not source.exists():
        return None
    rows = [json.loads(line) for line in
            source.read_text(encoding='utf-8').splitlines() if line.strip()]
    ticks = four_rotor_ticks(rows)
    wanted = {tick['t'] for tick in ticks}
    stream = [(row['t'], [[float(v) for v in center] for center in row['state']['previous_rotors']])
              for row in rows]
    chain = body_frame_chain(ticks)
    flips = orientation_audit(ticks)
    result = dict(method=method, four_rotor_ticks=len(ticks), ticks_in_stream=len(stream),
                  corrected_orientation_flips=len(flips),
                  orientation_flip_events=flips[:40],
                  labelers={})
    for tracker in (SerialTracker(), CVTracker(), RigidTracker(), ShapeMemoryTracker()):
        labels = run_labels(tracker, stream, wanted)
        metrics = evaluate_labels(ticks, labels, chain)
        lifetimes = sorted(track.seen - track.birth_t for track in tracker.tracks)
        metrics.update(tracker=tracker.name, identities_created=tracker.next_id,
                       revivals=tracker.revivals,
                       median_track_life_s=(float(np.median(lifetimes)) if lifetimes else None),
                       p90_track_life_s=(float(np.percentile(lifetimes, 90)) if lifetimes else None),
                       max_track_life_s=(float(lifetimes[-1]) if lifetimes else None),
                       tracks_over_1s=int(sum(1 for value in lifetimes if value > 1.0)),
                       fallback_ticks=getattr(tracker, 'fallback_ticks', 0),
                       rejected_orientation_flips=getattr(tracker, 'rejected_flips', 0),
                       degenerate_accepts=getattr(tracker, 'degenerate_accepts', 0),
                       transform_updates=getattr(tracker, 'transform_updates', 0),
                       revival_events=getattr(tracker, 'revival_log', [])[:20])
        result['labelers'][tracker.name] = metrics
    report.setdefault('methods', {})[method] = result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    report = dict(stage=11, run=str(run), raw=protocol['raw'],
                  interval_s=[protocol['start_s'], protocol['start_s'] + protocol['duration_s']],
                  definitions=dict(
                      orientation_flip='sign of the signed area of the four centres taken in a fixed '
                                       'label order changes between consecutive ticks while the quad is '
                                       'not degenerate; a planar rigid motion cannot do that',
                      rupture=f'a tracked identity moves more than {RUPTURE_FRACTION} x median rotor '
                              f'spacing from its own median position in the chained body frame',
                      rigid_tolerance=f'max({RIGID_TOL_FLOOR}, {RIGID_TOL_FRACTION} x median spacing)',
                      degeneracy=f'anisotropy <= {DEGENERACY_RATIO} or area/spacing^2 < 0.05'))
    for method in protocol['methods']:
        result = analyze(run, method, report)
        if result is None:
            print(json.dumps(dict(method=method, skipped='no observations.jsonl in this directory'),
                             ensure_ascii=False))
            continue
        print(json.dumps(dict(method=method, four_rotor_ticks=result['four_rotor_ticks'],
                              corrected_flips=result['corrected_orientation_flips'],
                              labelers={name: dict(
                                  jumps=value['identity_jumps'],
                                  jump_fraction=round(value['identity_jump_fraction'], 4),
                                  steps_p90=None if value['p90_step_px'] is None else round(value['p90_step_px'], 1),
                                  ids=value['identities_created'],
                                  revivals=value['revivals'],
                                  life_s=None if value['median_track_life_s'] is None else round(value['median_track_life_s'], 2),
                                  max_life_s=None if value['max_track_life_s'] is None else round(value['max_track_life_s'], 2),
                                  over_1s=value['tracks_over_1s'],
                                  rejected_flips=value['rejected_orientation_flips'])
                                  for name, value in result['labelers'].items()}),
                         ensure_ascii=False))
    destination = Path(args.out).resolve() if args.out else run / 'identity.json'
    if destination.exists():
        raise SystemExit('refusing to overwrite an existing identity.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                           encoding='utf-8')
    print(json.dumps(dict(written=str(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
