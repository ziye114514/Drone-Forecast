from pathlib import Path
import sys,unittest
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.geometry import fit_rotor_geometry,ellipse_points
from event_drone_forecast.config import Config
from event_drone_forecast.motion import RecursiveDynamics,ForecastBank
from event_drone_forecast.perception import RotorPerception


class GeometryTests(unittest.TestCase):
    def test_circle_keeps_position_when_axis_is_unobservable(self):
        points=ellipse_points([200,150,50,50,0],np.linspace(0,2*np.pi,240,endpoint=False))
        measurement=fit_rotor_geometry(points,500.)
        self.assertIsNotNone(measurement)
        self.assertEqual(measurement['mode'],'ellipse')
        self.assertLess(np.linalg.norm(measurement['center']-[200,150]),1.)
        self.assertLess(measurement['geometry_quality'],.1)

    def test_complete_ellipse_known_center_and_axis(self):
        parameters=np.array([180.,140.,65.,24.,.4])
        points=ellipse_points(parameters,np.linspace(0,2*np.pi,240,endpoint=False))
        measured=fit_rotor_geometry(points,500.)
        self.assertEqual(measured['mode'],'ellipse')
        self.assertLess(np.linalg.norm(measured['center']-parameters[:2]),1.)
        self.assertAlmostEqual(measured['major'],130.,delta=1.)

    def test_missing_half_contour_does_not_use_occluder_chord_as_boundary(self):
        image=np.zeros((400,500),np.uint8)
        cv2.ellipse(image,(240,190),(90,40),20,0,360,255,-1)
        image[200:]=0
        contours,_=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        measurement=fit_rotor_geometry(contours[0],500.)
        self.assertIsNotNone(measurement)
        self.assertEqual(measurement['mode'],'arc')
        self.assertLess(np.linalg.norm(measurement['center']-[240,190]),15.)
        self.assertAlmostEqual(measurement['major'],180.,delta=25.)

    def test_short_arc_cannot_be_high_confidence_full_shape(self):
        points=ellipse_points([200,150,80,30,.3],np.linspace(.5,.8,100))
        rng=np.random.default_rng(4);points+=rng.normal(0,.2,points.shape)
        measurement=fit_rotor_geometry(points,500.)
        if measurement is not None:
            self.assertTrue(measurement['axis_only'] or measurement['geometry_quality']<.5)

    def test_edge_on_line_retains_axis_without_tilt_claim(self):
        image=np.zeros((300,500),np.uint8)
        cv2.line(image,(100,100),(240,140),255,2)
        contours,_=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        measurement=fit_rotor_geometry(contours[0],500.)
        self.assertEqual(measurement['mode'],'line')
        self.assertTrue(measurement['axis_only'])
        self.assertAlmostEqual(measurement['angle'],np.rad2deg(np.arctan2(40,140)),delta=2.)

    def test_director_features_ignore_normal_sign(self):
        learner=RecursiveDynamics(Config(),True)
        cv=np.array([100,200,10,20]);ca=np.r_[cv,5,3]
        obs=dict(normal_xy=[.3,.8],frequency_quality=.8,geometry_quality=.9,
                 confidence=.9,frequency_squared_relative=1.3)
        first=learner.features(cv,ca,obs)
        obs['normal_xy']=[-.3,-.8]
        np.testing.assert_array_equal(first,learner.features(cv,ca,obs))

    def test_bad_geometry_freezes_rotor_learning_but_history_continues(self):
        bank=ForecastBank(Config())
        for i in range(100):
            obs=dict(t=i*.02,detected=True,center=[100+i,200],confidence=.9,
                     normal_xy=[.3,.8],geometry_quality=0.,frequency_quality=.9,
                     frequency_squared_relative=1.)
            result=bank.step(obs)
        self.assertEqual(result['online_updates']['adaptive_rotor'],0)
        self.assertGreater(result['online_updates']['adaptive_history'],20)
        self.assertFalse(result['rotor_learning_allowed'])
        self.assertEqual(result['states']['adaptive_rotor']['acceleration'],result['states']['adaptive_history']['acceleration'])

    def test_four_thin_rotors_do_not_require_prior_or_boxes(self):
        dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);chunks=[]
        for cx,cy in [(300,220),(470,220),(320,290),(490,290)]:
            image=np.zeros((720,1280),np.uint8)
            cv2.line(image,(cx-45,cy-1),(cx+45,cy+1),255,1)
            yy,xx=np.where(image>0)
            e=np.zeros(len(xx)*12,dtype=dtype);e['x']=np.repeat(xx,12);e['y']=np.repeat(yy,12)
            e['p']=np.tile(np.arange(12)%2,len(xx));e['t']=np.arange(len(e))%20000
            chunks.append(e)
        events=np.concatenate(chunks);events.sort(order='t')
        result,_=RotorPerception(1280,720,Config()).detect(events,.02)
        self.assertTrue(result['detected'])
        self.assertGreaterEqual(result['observation_modes']['line'],2)
        self.assertIsNone(result['tilt_proxy_deg'])

if __name__=='__main__':unittest.main()
