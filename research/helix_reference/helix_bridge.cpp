// Local ABI adapter. HelixTrack headers remain in the user's author checkout;
// no upstream source is copied, patched or redistributed by this adapter.
#include <cstdint>
#include <cmath>
#include <limits>
#include "helixtrack.h"

struct Handle {
    helixtrack::HelixTrackTracker tracker;
    std::int64_t first=-1;
    int blades=2;
    Handle(int b,double rpm,int width=256,int height=256,float cx=128.f,float cy=128.f,float radius=80.f):tracker(width,height,b),blades(b){
        cv::setNumThreads(1);tracker.enable_gpu_accum(false);
        tracker.set_initial_pose(cx,cy,radius,0.f);
        tracker.set_initial_phase(rpm,0.);
        tracker.set_kappa(22.f);tracker.set_q_jerk(400.f);tracker.set_rho(.5f);
        tracker.enable_radial_roi(true);tracker.set_radial_roi(.2f,1.5f);
        tracker.enable_polarity_phase_term(true);
        tracker.set_polarity_phase_params(.25f,.25f,-.25f,.25f);
        tracker.set_radial_centering(true);tracker.set_band_edge_barrier(.05f,5.f);
        tracker.set_phase_term_lambda(1.f);tracker.set_radial_term_lambda(1.f);
        tracker.enable_tip_balance(true);
        tracker.set_tip_balance_params(1.f,.06f,.03f,.02f,.02f,.05f,.5f,.5f);
        tracker.set_gn_gammas(1.f,1.f,1.f,6.f);
        tracker.set_rotation_sign(1);tracker.enable_auto_direction(true);
        tracker.enable_weights_visualization(false);
    }
};
extern "C" {
__declspec(dllexport) void* ht_create(int blades,double phase_equivalent_rpm){
    try{return new Handle(blades,phase_equivalent_rpm);}catch(...){return nullptr;}
}
__declspec(dllexport) void* ht_create_canvas(int blades,double phase_equivalent_rpm,
 int width,int height,double cx,double cy,double radius){
    try{return new Handle(blades,phase_equivalent_rpm,width,height,float(cx),float(cy),float(radius));}
    catch(...){return nullptr;}
}
__declspec(dllexport) void ht_destroy(void* pointer){delete static_cast<Handle*>(pointer);}
__declspec(dllexport) void ht_initialize_phase(void* pointer,double phase_equivalent_rpm,double phase_rad){
    static_cast<Handle*>(pointer)->tracker.set_initial_phase(phase_equivalent_rpm,phase_rad);
}
__declspec(dllexport) void ht_fixed_direction(void* pointer,int rotation_sign){
    auto& h=*static_cast<Handle*>(pointer);
    h.tracker.set_rotation_sign(rotation_sign);h.tracker.enable_auto_direction(false);
}
}
template<typename Coordinate> int feed_impl(void* pointer,const Coordinate* x,const Coordinate* y,
 const std::int64_t* t,const std::int16_t* p,int n,int solve_pose,double* output){
    try{
        auto& h=*static_cast<Handle*>(pointer);double cc=0.,ss=0.;int accepted=0;unsigned sectors=0,ring_sectors=0;
        double wc[2]={0.,0.},ws[2]={0.,0.},ww[2]={0.,0.};int pn[2]={0,0};
        for(int i=0;i<n;++i){
            if(h.first<0)h.first=t[i];
            auto pose=h.tracker.pose();auto state=h.tracker.state();
            auto u=pose.u_from_x(Eigen::Vector2f(float(x[i]),float(y[i])));auto radius=u.norm();
            if(radius>=.2f && radius<=1.5f){
                double a=std::atan2(double(h.tracker.rotation_sign())*u.y(),u.x());
                auto residual=helixtrack::wrap_pi(float(state.phase_at(1e-6*double(t[i]-h.first))-h.blades*pose.q(1)-h.blades*a));
                cc+=std::cos(residual);ss+=std::sin(residual);++accepted;
                int bin=int((a+helixtrack::kPiD)/(helixtrack::kTwoPiD)*12.);bin=std::max(0,std::min(11,bin));sectors|=(1u<<bin);
                // The author's phase measurement variance uses this SAME
                // radial Gaussian. Score predictive evidence at its precision.
                const double z=(double(radius)-1.)/.15,w=std::exp(-.5*z*z);
                const int pol=p[i]>0?1:0;
                wc[pol]+=w*std::cos(residual);ws[pol]+=w*std::sin(residual);ww[pol]+=w;++pn[pol];
                if(w>=.1)ring_sectors|=(1u<<bin);
            }
            h.tracker.process_event(x[i],y[i],t[i],p[i]>0?1:-1);
        }
        if(solve_pose)h.tracker.solve_pose_and_reset(.001f,2000);
        auto state=h.tracker.state();auto pose=h.tracker.pose();
        output[0]=state.omega;output[1]=state.alpha;output[2]=state.phi0;
        output[3]=accepted?std::hypot(cc,ss)/accepted:0.;output[4]=accepted;
        output[5]=0.;for(int k=0;k<12;++k)output[5]+=((sectors>>k)&1u);
        output[6]=h.tracker.rotation_sign();
        for(int k=0;k<6;++k)output[7+k]=pose.q(k);
        for(int k=0;k<9;++k)output[13+k]=pose.H.data()[k];
        double total=ww[0]+ww[1];
        output[22]=total>0?std::hypot(wc[0]+wc[1],ws[0]+ws[1])/total:0.;
        output[23]=total;output[24]=0.;for(int k=0;k<12;++k)output[24]+=((ring_sectors>>k)&1u);
        for(int k=0;k<2;++k){output[25+k]=ww[k]>0?std::hypot(wc[k],ws[k])/ww[k]:0.;output[27+k]=pn[k];}
        output[29]=helixtrack::wrap_pi(float(std::atan2(ws[1],wc[1])-std::atan2(ws[0],wc[0])));
        output[30]=total>0?(std::hypot(wc[0],ws[0])+std::hypot(wc[1],ws[1]))/total:0.;
        output[31]=total;
        return 0;
    }catch(...){return -1;}
}
extern "C" {
__declspec(dllexport) int ht_feed_v2(void* pointer,const std::uint16_t* x,const std::uint16_t* y,
 const std::int64_t* t,const std::int16_t* p,int n,int solve_pose,double* output){
    return feed_impl(pointer,x,y,t,p,n,solve_pose,output);
}
__declspec(dllexport) int ht_feed_signed_v2(void* pointer,const std::int32_t* x,const std::int32_t* y,
 const std::int64_t* t,const std::int16_t* p,int n,int solve_pose,double* output){
    return feed_impl(pointer,x,y,t,p,n,solve_pose,output);
}
__declspec(dllexport) int ht_feed(void* pointer,const std::uint16_t* x,const std::uint16_t* y,
 const std::int64_t* t,const std::int16_t* p,int n,int solve_pose,double* output){
    double all[32]={0.};int status=ht_feed_v2(pointer,x,y,t,p,n,solve_pose,all);
    for(int k=0;k<22;++k)output[k]=all[k];return status;
}
}
