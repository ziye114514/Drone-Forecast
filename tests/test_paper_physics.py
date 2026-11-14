from pathlib import Path
import sys,unittest
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.paper_physics import metric_acceleration,metric_forward,circle_normal_candidates,ThrustAxisFilter,ProjectedThrustModel
from event_drone_forecast.rotor_sensing import periodic_measurement,RotorSensing
from event_drone_forecast.motion import extrapolate
from synthetic_rotor import rotating_edges


class PaperPhysicsTests(unittest.TestCase):
    def test_hover_thrust_balances_gravity_in_metric_equation(self):
        omega=np.ones(4)*900;mass=1.4;cf=mass*9.81/(4*900**2)
        np.testing.assert_allclose(metric_acceleration(omega,[0,0,1],mass,cf),0,atol=1e-10)
        p,v,n=metric_forward([1,2,3],[0,0,0],[0,0,1],[0,0,0],omega,mass,cf,.4)
        np.testing.assert_allclose(p,[1,2,3],atol=1e-8)

    def test_tilted_thrust_predicts_next_position(self):
        normal=np.array([.3,0,np.sqrt(.91)]);omega=np.ones(4)*900;cf=9.81/(4*900**2)
        acceleration=metric_acceleration(omega,normal,1.,cf)
        p,_,_=metric_forward([0,0,0],[2,0,0],normal,[0,0,0],omega,1.,cf,.4)
        np.testing.assert_allclose(p,np.array([2,0,0])*.4+.5*acceleration*.4**2,atol=1e-8)

    def test_calibrated_circle_backprojection_has_true_normal_solution(self):
        normal=np.array([.3,-.7,.64807406984]);normal/=np.linalg.norm(normal)
        axis=np.cross(normal,[1,0,0]);axis/=np.linalg.norm(axis);second=np.cross(normal,axis)
        phases=np.linspace(0,2*np.pi,500,endpoint=False)
        points=np.array([.25,.1,3.])+ .15*(np.cos(phases)[:,None]*axis+np.sin(phases)[:,None]*second)
        K=np.array([[900,0,640],[0,930,360],[0,0,1.]])
        projected=points@K.T;pixels=projected[:,:2]/projected[:,2:]
        (cx,cy),(d1,d2),angle=cv2.fitEllipse(pixels.astype('float32'))
        candidates=circle_normal_candidates((cx,cy,d1/2,d2/2,np.deg2rad(angle)),K)
        self.assertGreater(max(abs(n@normal) for n in candidates),.99999)

    def test_known_rotation_recovers_blade_pass_and_provisional_rpm(self):
        events,rotor=rotating_edges(rps=120.,ratio=.4)
        result=periodic_measurement(events,rotor,[0,0],.16,PaperConfig())
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['blade_pass_hz'],240.,delta=5.)
        self.assertAlmostEqual(result['mechanical_rpm_assuming_blade_count'],7200.,delta=150.)

    def test_partial_and_translating_rotation(self):
        events,rotor=rotating_edges(rps=140.,ratio=.3,occlusion=True,velocity=(100,-20))
        result=periodic_measurement(events,rotor,[100,-20],.16,PaperConfig())
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['blade_pass_hz'],280.,delta=7.)

    def test_stable_rotor_ids_survive_detection_order_reversal(self):
        sensing=RotorSensing(PaperConfig());rotors=[]
        for x,y in [(300,200),(450,200),(310,280),(460,280)]:
            rotors.append(dict(cx=x,cy=y,major=80.,minor=20.,angle=0.,ratio=.25,geometry_quality=.8,activity=1000))
        first=sensing.step(dict(rotors=rotors),[],.01)
        second=sensing.step(dict(rotors=[dict(r,cx=r['cx']+3) for r in reversed(rotors)]),[],.02)
        before={r['rotor_id']:r['cx'] for r in first['rotors']}
        for r in second['rotors']:self.assertAlmostEqual(r['cx']-before[r['rotor_id']],3.)

    def test_rollout_used_for_forecast_not_constant_acceleration_shortcut(self):
        state=dict(position=[0,0],velocity=[0,0],acceleration=[0,0],
                   rollout_times=[0,.2,.4],rollout_positions=[[0,0],[10,3],[30,8]])
        np.testing.assert_allclose(extrapolate(state,.3),[20,5.5])

    def test_bad_measurement_freezes_effective_physical_parameters(self):
        model=ProjectedThrustModel(PaperConfig(),True)
        before=model.parameters.copy()
        for i in range(100):model.update(i*.01,np.array([i,0,100,0]),[.2,-.9,.3],1.2,0.,True,True)
        np.testing.assert_array_equal(model.parameters,before);self.assertEqual(model.updates,0)

    def test_vectorized_drag_rollout_matches_closed_form(self):
        model=ProjectedThrustModel(PaperConfig(),False);model.parameters=np.array([300.,200.,.4,.7])
        normal=np.array([.3,-.9,np.sqrt(.1)]);normal/=np.linalg.norm(normal)
        p=np.array([10.,20.]);v=np.array([40.,-10.]);h=.4
        times,points=model.rollout(p,v,normal,[0,0,0],1.2)
        force=300*1.2*normal[:2]+[0,200]
        drag=np.array([.4,.7]);B=(1-np.exp(-drag*h))/drag
        expected=p+B*v+(h-B)/drag*force
        np.testing.assert_allclose(points[np.argmin(abs(times-h))],expected,atol=1e-8)


if __name__=='__main__':unittest.main()
