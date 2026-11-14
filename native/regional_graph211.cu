// Exact data-independent minimum-priority votes from corrected208.
// Raw ON/OFF counts retain every in-window native event. Regional timing
// votes are explicitly capped once per pixel/polarity/nuisance block.
typedef unsigned long long U64;
struct Event { unsigned short x,y; short p,pad; long long t; };
__device__ U64 priority211(U64 i) {
    U64 z=i+0x9e3779b97f4a7c15ULL+194ULL;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}
extern "C" __global__ void regional_rank(const Event* e,const long long* header,
    U64* ranks,int* counts,int* seen,int* spatial) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=header[0])return;
    Event v=e[i];int w=(int)header[3],h=(int)header[4];
    if(v.x>=w||v.y>=h||v.t<header[1]||v.t>=header[2])return;
    int p=v.p>0,pixel=v.y*w+v.x,pp=pixel*2+p,size=w*h;
    int tick=(int)((v.t-header[1])*512/(header[2]-header[1]));
    atomicAdd(counts+(1-p)*size+pixel,1);
    if(v.t>=header[2]-20000)atomicAdd(counts+(3-p)*size+pixel,1);
    atomicMin(ranks+(U64)pp*32+tick/16,priority211(i));
    if(atomicCAS(seen+pp,0,1)==0)atomicAdd(spatial+((v.y/16)*((w+15)/16)+v.x/16)*2+p,1);
}
extern "C" __global__ void regional_histogram(const Event* e,const long long* header,
    const U64* ranks,float* hist) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=header[0])return;
    Event v=e[i];int w=(int)header[3],h=(int)header[4];
    if(v.x>=w||v.y>=h||v.t<header[1]||v.t>=header[2])return;
    int p=v.p>0,pixel=v.y*w+v.x;
    int tick=(int)((v.t-header[1])*512/(header[2]-header[1]));
    if(ranks[(U64)(pixel*2+p)*32+tick/16]==priority211(i)) {
        int group=((v.y/16)*((w+15)/16)+v.x/16)*2+p;
        atomicAdd(hist+(U64)group*512+tick,1.f);
    }
}
