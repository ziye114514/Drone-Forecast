#include "ellipse_hypotheses.cpp"
// Same first-order conic distance, no fast-math / approximate trig / sampling.
extern "C" __declspec(dllexport) int ellipse_distance_batch(
 const double* points,int count,const double* parameters,int hypotheses,double* out){
 if(!points||!parameters||!out||count<0||hypotheses<0)return -1;
 for(int j=0;j<hypotheses;++j){
  const double* p=parameters+5*j;double co=std::cos(p[4]),si=std::sin(p[4]);
  double a2=p[2]*p[2],b2=p[3]*p[3];
  for(int i=0;i<count;++i){
   double dx=points[2*i]-p[0],dy=points[2*i+1]-p[1];
   double x=dx*co+dy*si,y=-dx*si+dy*co;
   double gx=x/a2,gy=y/b2,gradient=2*std::sqrt(gx*gx+gy*gy);
   double xa=x/p[2],yb=y/p[3];
   out[j*count+i]=(xa*xa+yb*yb-1)/(gradient>1e-6?gradient:1e-6);
  }
 }
 return 0;
}
