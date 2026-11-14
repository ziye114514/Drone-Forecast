#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
struct TimingHandle{
 int w,h,cell,bins,block,cols,groups;uint32_t frame=0;
 std::vector<uint32_t> stamp;std::vector<uint16_t> last,tick;
 std::vector<uint64_t> rank;
 TimingHandle(int W,int H,int C,int N,int B):w(W),h(H),cell(C),bins(N),block(B),cols((W+C-1)/C),groups(cols*((H+C-1)/C)*2),stamp(2*W*H,0),last(2*W*H),tick(2*W*H),rank(2*W*H){}
};
extern "C" __declspec(dllexport) void* create_timing_histogram(int w,int h,int cell,int bins,int block){
 if(w<=0||h<=0||cell<=0||bins<=0||block<=0||2*bins/block>65535)return nullptr;
 try{return new TimingHandle(w,h,cell,bins,block);}catch(...){return nullptr;}
}
extern "C" __declspec(dllexport) void destroy_timing_histogram(void* p){delete static_cast<TimingHandle*>(p);}
static uint64_t priority(uint64_t i){
 uint64_t z=i+UINT64_C(0x9e3779b97f4a7c15)+194;
 z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);
}
extern "C" __declspec(dllexport) int timing_histogram(void* opaque,const unsigned char* data,int64_t count,int64_t stride,
 int xo,int yo,int po,int to,int64_t begin,int64_t end,float* hist,int64_t* spatial){
 if(!opaque||count<0||end<=begin)return 1;auto& H=*static_cast<TimingHandle*>(opaque);
 if(++H.frame==0){std::fill(H.stamp.begin(),H.stamp.end(),0);H.frame=1;}
 std::fill(hist,hist+H.groups*2*H.bins,0.f);std::fill(spatial,spatial+H.groups,0);
 int64_t previous=std::numeric_limits<int64_t>::min();uint64_t index=0;
 for(int64_t i=0;i<count;++i){
  const unsigned char* p=data+i*stride;int64_t time;std::memcpy(&time,p+to,8);
  if(time<begin||time>=end)continue;if(time<previous)return 2;previous=time;
  uint16_t x,y;int16_t polarity;std::memcpy(&x,p+xo,2);std::memcpy(&y,p+yo,2);std::memcpy(&polarity,p+po,2);
  if(x>=H.w||y>=H.h)return 3;
  int pixel=(y*H.w+x)*2+(polarity>0),group=((y/H.cell)*H.cols+x/H.cell)*2+(polarity>0);
  uint16_t t=static_cast<uint16_t>((time-begin)*(2*H.bins)/(end-begin)),b=t/H.block;
  uint64_t r=priority(index++);
  if(H.stamp[pixel]!=H.frame){H.stamp[pixel]=H.frame;H.last[pixel]=b;H.rank[pixel]=r;H.tick[pixel]=t;++spatial[group];++hist[group*2*H.bins+t];}
  else if(H.last[pixel]!=b){H.last[pixel]=b;H.rank[pixel]=r;H.tick[pixel]=t;++hist[group*2*H.bins+t];}
  else if(r<H.rank[pixel]){--hist[group*2*H.bins+H.tick[pixel]];++hist[group*2*H.bins+t];H.rank[pixel]=r;H.tick[pixel]=t;}
 }
 return 0;
}
