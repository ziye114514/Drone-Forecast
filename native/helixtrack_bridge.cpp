// Local research C ABI over the separately supplied, unmodified author core.
// No RAW reader, SDK, GT files, GUI, HDF5, YAML, or author source is bundled.
#include <cmath>
#include <cstdint>
#include <limits>
#include "helixtrack_tracker.h"

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

struct Handle {
    helixtrack::HelixTrackTracker tracker;
    Handle() : tracker(1280,720,2) {}
};

API void* ht_create(double minor_radius, double ratio, double major_angle,
                    double phase_hz, double phi, int direction) {
    try {
        if (!(minor_radius>0 && ratio>.16 && ratio<=1 && phase_hz>0 &&
              std::isfinite(phase_hz) && (direction==1 || direction==-1))) return nullptr;
        auto *h=new Handle();
        // Work relative to the observed ellipse centre. For c=0, the author's
        // H=[sR,t;p^T,1] has ellipse axes(s,s/sqrt(1-|p|^2)) when t=sRp.
        // This maps the actual observed conic, avoiding the ambiguity between
        // author's origin/tip aspect parameter and a measured ellipse ratio.
        const double psi=major_angle-helixtrack::kPiD/2;
        const double rho=std::sqrt(std::max(0.,1-ratio*ratio));
        Eigen::Matrix<float,6,1> target;
        target << std::log(minor_radius),psi,minor_radius*rho*std::cos(psi),
            minor_radius*rho*std::sin(psi),rho,0.;
        h->tracker.apply_pose_increment_ext(target-h->tracker.pose().q);
        h->tracker.set_rotation_sign(direction);
        h->tracker.enable_auto_direction(false);
        h->tracker.set_initial_phase(phase_hz*60.,phi); // author omega in rad/s
        h->tracker.set_kappa(22.f);h->tracker.set_q_jerk(400.f);
        h->tracker.set_rho(.5f);
        h->tracker.enable_radial_roi(true);h->tracker.set_radial_roi(.2f,1.5f);
        h->tracker.set_phase_term_lambda(1.f);h->tracker.set_radial_term_lambda(1.f);
        h->tracker.enable_polarity_phase_term(true);
        h->tracker.set_polarity_phase_params(.25f,.25f,-.25f,.25f);
        h->tracker.set_radial_centering(true);h->tracker.set_band_edge_barrier(.05f,5.f);
        h->tracker.enable_tip_balance(true);
        h->tracker.set_tip_balance_params(1.f,.06f,.03f,.02f,.02f,.05f,.5f,.5f);
        h->tracker.enable_weights_visualization(false);
        return h;
    } catch (...) { return nullptr; }
}

API int ht_process(void *ptr,const int32_t *x,const int32_t *y,const int64_t *t,
                   const int32_t *p,int64_t n,int solve_pose) {
    if (!ptr || n<0) return -1;
    try {
        auto &tr=static_cast<Handle*>(ptr)->tracker;
        for (int64_t i=0;i<n;i++) {
            tr.process_event(x[i],y[i],t[i],p[i]);
            if (solve_pose && tr.num_events_in_H_batch()>=2000) tr.solve_pose_and_reset(.001f,2000);
        }
        return 0;
    } catch (...) { return -2; }
}

API int ht_state(void *ptr,double *out) {
    if (!ptr) return -1;
    auto &tr=static_cast<Handle*>(ptr)->tracker;const auto s=tr.state();const auto p=tr.pose();
    out[0]=s.omega/helixtrack::kTwoPiD;out[1]=s.alpha/helixtrack::kTwoPiD;
    out[2]=s.phi0;out[3]=tr.last_time_s();
    for (int i=0;i<6;i++) out[4+i]=p.q[i];
    for (int i=0;i<3;i++) for(int j=0;j<3;j++) out[10+3*i+j]=p.H(i,j);
    return 0;
}

API int ht_ring(void *ptr,double *out,int n) {
    if (!ptr || n<8) return -1;
    const auto p=static_cast<Handle*>(ptr)->tracker.pose();
    for(int i=0;i<n;i++) {
        const double theta=helixtrack::kTwoPiD*i/n;
        const auto v=p.x_from_u(Eigen::Vector2f(std::cos(theta),std::sin(theta)));
        out[2*i]=v[0];out[2*i+1]=v[1];
    }
    return 0;
}

API void ht_destroy(void *ptr) { delete static_cast<Handle*>(ptr); }
