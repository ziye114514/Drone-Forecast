// Original deterministic arc search, same inputs/order/gates. No temporal
// skipping, point subsampling, uncertainty approximation or cached poses.
#include "geometry_kernels206.cpp"
#include <algorithm>
#include <vector>
#include <array>
namespace {
bool fit215(const std::vector<cv::Point2f>& pts,std::array<double,5>& p){
 if(pts.size()<6)return false;
 try{
  auto r=cv::fitEllipseAMS(pts);double a=r.size.width,b=r.size.height,angle=r.angle;
  if(a<b){std::swap(a,b);angle+=90.;}
  p={r.center.x,r.center.y,a*.5,b*.5,std::fmod(angle*0.017453292519943295,3.14159265358979323846)};
  if(b<=.1)return false;
  for(auto v:p)if(!std::isfinite(v))return false;
  return true;
 }catch(const cv::Exception&){return false;}
}
double coverage215(const double* pts,int n,const std::array<double,5>& p,const std::vector<unsigned char>& keep){
 std::vector<double> phase;phase.reserve(n);double co=std::cos(p[4]),si=std::sin(p[4]),pi2=6.28318530717958647692;
 for(int i=0;i<n;++i)if(keep[i]){
  double dx=pts[2*i]-p[0],dy=pts[2*i+1]-p[1];
  double angle=std::atan2((-dx*si+dy*co)/p[3],(dx*co+dy*si)/p[2]);
  if(angle<0)angle+=pi2;phase.push_back(angle);
 }
 if(phase.size()<3)return 0.;
 std::sort(phase.begin(),phase.end());double gap=phase[0]+pi2-phase.back();
 for(size_t j=1;j<phase.size();++j)gap=std::max(gap,phase[j]-phase[j-1]);
 return 1-gap/pi2;
}
}
extern "C" __declspec(dllexport) int arc_search215(const double* points,int n,double extent,
 const int* indices,int hypotheses,double min_coverage,double max_rms,double* output,unsigned char* best_keep){
 if(!points||!indices||!output||!best_keep||n<22||hypotheses<0)return -1;
 static bool initialized=[](){cv::setNumThreads(1);return true;}();(void)initialized;
 std::vector<cv::Point2f> all;all.reserve(n);
 for(int i=0;i<n;++i)all.emplace_back((float)points[2*i],(float)points[2*i+1]);
 double minimum=std::max(22.,.60*n),best_score=-std::numeric_limits<double>::infinity();bool found=false;
 std::vector<double> dist(n);std::vector<unsigned char> keep(n);
 auto shape=[&](const std::array<double,5>& p){return 10<=2*p[2]&&2*p[2]<=extent*1.6&&p[3]>=2.5&&p[3]/p[2]>=.12;};
 auto consider=[&](const std::array<double,5>& initial){
  if(!shape(initial))return false;
  ellipse_distance_batch(points,n,initial.data(),1,dist.data());
  std::vector<cv::Point2f> selected;selected.reserve(n);
  for(int i=0;i<n;++i)if(std::abs(dist[i])<=1.8)selected.push_back(all[i]);
  if(selected.size()<minimum)return false;
  std::array<double,5> p;if(!fit215(selected,p))return false;
  ellipse_distance_batch(points,n,p.data(),1,dist.data());int count=0;double sum=0.;
  for(int i=0;i<n;++i){keep[i]=std::abs(dist[i])<=1.8;if(keep[i]){++count;sum+=dist[i]*dist[i];}}
  if(count<minimum||!shape(p))return false;
  double rms=std::sqrt(sum/count),coverage=coverage215(points,n,p,keep);
  if(coverage<min_coverage||rms>max_rms)return false;
  double fraction=(double)count/n,score=fraction+.2*coverage-.1*rms;
  if(score>best_score){
   found=true;best_score=score;std::copy(p.begin(),p.end(),output);
   output[5]=rms;output[6]=coverage;output[7]=fraction;
   std::copy(keep.begin(),keep.end(),best_keep);
  }
  return fraction>=.90&&coverage>=.72&&rms<.8;
 };
 std::array<double,5> p;bool stop=fit215(all,p)&&consider(p);
 if(!stop)for(int j=0;j<hypotheses;++j){
  std::vector<cv::Point2f> sample;sample.reserve(7);
  for(int k=0;k<7;++k){int i=indices[j*7+k];if(i<0||i>=n)return -2;sample.push_back(all[i]);}
  if(fit215(sample,p)&&consider(p))break;
 }
 return found?1:0;
}
