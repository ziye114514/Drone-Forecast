#define NATIVE_AMS_WORKERS 1
#include "ellipse_hypotheses.cpp"
#include <immintrin.h>
extern "C" __declspec(dllexport) int ellipse_distance_batch(
 const double* points,int count,const double* parameters,int hypotheses,double* out){
 if(!points||!parameters||!out||count<0||hypotheses<0)return -1;
 const __m256d one=_mm256_set1_pd(1.),two=_mm256_set1_pd(2.),floor=_mm256_set1_pd(1e-6);
 for(int j=0;j<hypotheses;++j){
  const double* p=parameters+5*j;double co=std::cos(p[4]),si=std::sin(p[4]);
  double a2=p[2]*p[2],b2=p[3]*p[3];
  __m256d cx=_mm256_set1_pd(p[0]),cy=_mm256_set1_pd(p[1]),c=_mm256_set1_pd(co),s=_mm256_set1_pd(si);
  __m256d a=_mm256_set1_pd(p[2]),b=_mm256_set1_pd(p[3]),aa=_mm256_set1_pd(a2),bb=_mm256_set1_pd(b2);
  int i=0;
  for(;i+4<=count;i+=4){
   __m256d v0=_mm256_loadu_pd(points+2*i),v1=_mm256_loadu_pd(points+2*i+4);
   __m256d dx=_mm256_sub_pd(_mm256_permute4x64_pd(_mm256_unpacklo_pd(v0,v1),0xd8),cx);
   __m256d dy=_mm256_sub_pd(_mm256_permute4x64_pd(_mm256_unpackhi_pd(v0,v1),0xd8),cy);
   __m256d x=_mm256_add_pd(_mm256_mul_pd(dx,c),_mm256_mul_pd(dy,s));
   __m256d y=_mm256_add_pd(_mm256_sub_pd(_mm256_setzero_pd(),_mm256_mul_pd(dx,s)),_mm256_mul_pd(dy,c));
   __m256d gx=_mm256_div_pd(x,aa),gy=_mm256_div_pd(y,bb);
   __m256d gradient=_mm256_mul_pd(two,_mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(gx,gx),_mm256_mul_pd(gy,gy))));
   __m256d xa=_mm256_div_pd(x,a),yb=_mm256_div_pd(y,b);
   __m256d numerator=_mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(xa,xa),_mm256_mul_pd(yb,yb)),one);
   _mm256_storeu_pd(out+j*count+i,_mm256_div_pd(numerator,_mm256_max_pd(gradient,floor)));
  }
  for(;i<count;++i){
   double dx=points[2*i]-p[0],dy=points[2*i+1]-p[1],x=dx*co+dy*si,y=-dx*si+dy*co;
   double gx=x/a2,gy=y/b2,gradient=2*std::sqrt(gx*gx+gy*gy),xa=x/p[2],yb=y/p[3];
   out[j*count+i]=(xa*xa+yb*yb-1)/(gradient>1e-6?gradient:1e-6);
  }
 }
 return 0;
}
