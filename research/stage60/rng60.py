"""Controlled RNG intervention on the exact saved real collinear contour."""
from pathlib import Path
import cv2,numpy as np
import study60 as current

def run():
    current.verify();out=current.OUT/'rng_cause60.json';assert not out.exists()
    probe=current.load(current.OUT/'degenerate_contour_probe.json');row=next(r for r in probe['contours'] if r['bbox']==[443,293,1,12]);q=np.array(row['points'],np.float32).reshape(-1,1,2)
    def fit():
        e=cv2.fitEllipse(q);return [list(e[0]),list(e[1]),e[2]]
    fixed=[];consumed=[];unseeded=[]
    for repeat in range(6):
        cv2.setRNGSeed(123);fixed.append(fit())
        cv2.setRNGSeed(123);temp=np.zeros((32,32),np.float32);cv2.randu(temp,0.,1.);consumed.append(fit())
    cv2.setRNGSeed(123)
    for repeat in range(6):unseeded.append(fit())
    result=dict(helper_sha256=current.sha(Path(__file__)),opencv_version=cv2.__version__,contour_source='degenerate_contour_probe.json bbox443,293,1,12',
        exact_point_array_held_constant=True,fixed_seed=fixed,random_numbers_consumed_before_fit=consumed,sequential_same_array_fits=unseeded,
        fixed_seed_all_equal=all(x==fixed[0] for x in fixed),consumption_intervention_changed_fit=consumed[0]!=fixed[0],sequential_fit_changes=len(set(map(str,unseeded)))>1,
        inference='Only if all three observed checks pass: OpenCV RNG stream consumption changes fitEllipse on this identical degenerate real contour. This explains context-dependent candidate length and potential grouping, without identifying rotor identity or every earlier missing failure record. No global seed changes made in production.')
    current.save(out,result);print({k:result[k] for k in ['opencv_version','fixed_seed_all_equal','consumption_intervention_changed_fit','sequential_fit_changes']},flush=True)

if __name__=='__main__':run()
