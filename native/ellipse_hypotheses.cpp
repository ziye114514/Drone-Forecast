// Batch only the AMS seed calls. Python retains every original refinement,
// residual, coverage, uncertainty gate and candidate-order decision.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <limits>
#ifndef NATIVE_AMS_WORKERS
#define NATIVE_AMS_WORKERS 4
#endif
extern "C" __declspec(dllexport) int batch_ellipse_ams(
 const float* points,int count,int per,double* output,int* accepted){
 if(!points||!output||!accepted||count<0||per<6)return -1;
 // SDK4 and the separately loaded Python OpenCV are not mixed at the ABI.
 static bool initialized=[](){cv::setNumThreads(1);return true;}();
 (void)initialized;
 #pragma omp parallel for num_threads(NATIVE_AMS_WORKERS) schedule(static)
 for(int i=0;i<count;++i){
  accepted[i]=0;double* p=output+5*i;
  for(int j=0;j<5;++j)p[j]=std::numeric_limits<double>::quiet_NaN();
  try{
   cv::Mat view(per,1,CV_32FC2,const_cast<float*>(points+2*per*i));
   cv::RotatedRect r=cv::fitEllipseAMS(view);
   double a=r.size.width,b=r.size.height,angle=r.angle;
   if(a<b){double t=a;a=b;b=t;angle+=90.;}
   p[0]=r.center.x;p[1]=r.center.y;p[2]=a*.5;p[3]=b*.5;
   p[4]=std::fmod(angle*0.017453292519943295,3.14159265358979323846);
   bool good=b>.1;for(int j=0;j<5;++j)good=good&&std::isfinite(p[j]);
   accepted[i]=good?1:0;
  }catch(const cv::Exception&){}
 }
 return 0;
}
