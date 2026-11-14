// Project-owned constant-acceleration circular phase filter. No upstream code.
// State: native microsecond cutoff, instantaneous phase/rate/acceleration, P3x3.
#include <cmath>
#include <cstdint>
#include <algorithm>
extern "C" __declspec(dllexport) int cp_feed(double* s,const std::int64_t* times,
 const double* phase,const double* radial,int count,int mode){
    try {
        for(int e=0;e<count;++e){
            const double dt=(double(times[e])-s[0])*1e-6;
            if(dt < -1e-10)return -2;
            const double d=std::max(0.,dt),d2=d*d,d3=d2*d,d4=d3*d,d5=d4*d;
            s[1]+=s[2]*d+.5*s[3]*d2;s[2]+=s[3]*d;
            const double F[9]={1,d,.5*d2,0,1,d,0,0,1};
            const double Q[9]={d5/20,d4/8,d3/6,d4/8,d3/3,d2/2,d3/6,d2/2,d};
            double temp[9]={0},P[9]={0};
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)temp[3*i+j]+=F[3*i+k]*s[4+3*k+j];
            for(int i=0;i<3;++i)for(int j=0;j<3;++j){
                for(int k=0;k<3;++k)P[3*i+j]+=temp[3*i+k]*F[3*j+k];
                P[3*i+j]+=400.*Q[3*i+j];
            }
            if(mode&2){
                // CoHelix S2: small relative rate random walk and circular
                // phase uncertainty after an unobserved stream gap. Retain
                // rate/acceleration instead of restarting the whole tracker.
                const double qr=std::pow(.0025*std::abs(s[2]),2);
                P[0]+=qr*d3/3;P[1]+=qr*d2/2;P[3]+=qr*d2/2;P[4]+=qr*d;
                const double limit=3.14159265358979323846*3.14159265358979323846/3;
                if(d>.020)P[0]=std::max(P[0],limit);
                if(P[0]>limit){
                    const double f=std::sqrt(limit/P[0]);
                    for(int i=0;i<3;++i)for(int j=0;j<3;++j)P[3*i+j]*=(i==0?f:1)*(j==0?f:1);
                }
            }
            const double residual=std::remainder(phase[e]-s[1],2*3.14159265358979323846);
            double R=1./std::max(1e-3,22.*radial[e]);
            if(mode&1){
                // Wrapped-normal component plus uniform circular outliers;
                // the fixed half/half prior mirrors the established rho=.5.
                double v=P[0]+R,likelihood=std::exp(-.5*residual*residual/v)/std::sqrt(2*3.14159265358979323846*v);
                double credit=likelihood/(likelihood+1/(2*3.14159265358979323846));R/=std::max(credit,1e-6);
            }
            const double S=P[0]+R,K[3]={P[0]/S,P[3]/S,P[6]/S};
            s[1]+=K[0]*residual;s[2]+=K[1]*residual;s[3]+=K[2]*residual;
            // Scalar Joseph covariance update, stable for dense event bursts.
            double A[9]={1-K[0],0,0,-K[1],1,0,-K[2],0,1},U[9]={0},V[9]={0};
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)U[3*i+j]+=A[3*i+k]*P[3*k+j];
            for(int i=0;i<3;++i)for(int j=0;j<3;++j){
                for(int k=0;k<3;++k)V[3*i+j]+=U[3*i+k]*A[3*j+k];
                V[3*i+j]+=R*K[i]*K[j];
            }
            s[0]=double(times[e]);s[1]=std::remainder(s[1],2*3.14159265358979323846);
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)s[4+3*i+j]=.5*(V[3*i+j]+V[3*j+i]);
        }
        for(int k=0;k<13;++k)if(!std::isfinite(s[k]))return -3;
        return 0;
    } catch(...){return -1;}
}
extern "C" __declspec(dllexport) int cp_version(){return 2;}
