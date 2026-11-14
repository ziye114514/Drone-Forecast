"""Shared current-conic eligibility; independent of optional author phase code."""


def usable(r):
    return (r.get('current_observation',True) and not r.get('fov_clipped')
        and r.get('minor',0)/max(r.get('major',1),1e-12)>.16
        and r.get('center_sigma_px',99)<=3 and r.get('axis_angle_sigma_deg',99)<=15
        and (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated')))
