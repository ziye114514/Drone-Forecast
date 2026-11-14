// Original fused implementation of the existing causal two-half NDFT test.
// No event binning, temporal subsampling, neural detector, or force calibration.
extern "C" __global__ void rotor_phase_coefficients(const long long* times,const short* polarity,
 const long long* lengths,const long long* offsets,const long long* ntrain,
 const float* period,int count,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float* out=output+4*i;for(int k=0;k<4;++k)out[k]=0.f;
 float f=period[12*i+1];int n=(int)lengths[i],nt=(int)ntrain[i],nv=n-nt;
 if(f<=0.f||nt<1||nv<1)return;
 long long start=offsets[i];double mean0=0.,mean1=0.;
 for(int j=0;j<n;++j){double p=2.*polarity[start+j]-1.;if(j<nt)mean0+=p;else mean1+=p;}
 mean0/=nt;mean1/=nv;double real0=0.,imag0=0.,real1=0.,imag1=0.;
 for(int j=0;j<n;++j){
   float phase=6.283185307179586f*f*(float)times[start+j]/1000000.f;float s,c;sincosf(phase,&s,&c);
   double w=2.*polarity[start+j]-1.-(j<nt?mean0:mean1);
   if(j<nt){real0+=w*c;imag0-=w*s;}else{real1+=w*c;imag1-=w*s;}
 }
 out[0]=(float)(real0/nt);out[1]=(float)(imag0/nt);out[2]=(float)(real1/nv);out[3]=(float)(imag1/nv);
}

extern "C" __global__ void rotor_harmonics(
 const long long* times,const short* polarity,const long long* pixels,
 const long long* lengths,const long long* offsets,const long long* ntrain,
 const float* frequencies,int nf,int window_us,float* output){
 int pixel=blockIdx.x,k=threadIdx.x,nt=(int)ntrain[pixel],n=(int)lengths[pixel],nv=n-nt;
 long long start=offsets[pixel];
 __shared__ float zr[256],zi[256],power[64],mean;
 __shared__ int choice[2],covered;
#ifdef ROTOR_WARP_REDUCTION
 __shared__ double reduction[16][8];
 double stats[16]={0.};
#else
 __shared__ double reduction[16][256];
#endif
 if(k==0){
   float sum=0;int bins0=0,bins1=0;
   long long min0=window_us,max0=0,min1=window_us,max1=0;
   for(int j=0;j<n;++j){
     int bin=(int)(times[start+j]*8/window_us);bin=bin>7?7:bin;
     long long t=times[start+j];
     if(j<nt){sum+=2.f*(float)polarity[start+j]-1.f;bins0|=1<<bin;min0=t<min0?t:min0;max0=t>max0?t:max0;}
     else {bins1|=1<<bin;min1=t<min1?t:min1;max1=t>max1?t:max1;}
   }
   mean=sum/nt;
   covered=nt>0&&nv>0&&
     max0-min0>=window_us/4.f&&max1-min1>=window_us/4.f&&
     __popc((unsigned)bins0)>=2&&__popc((unsigned)bins1)>=2;
 }
 __syncthreads();
 if(k<nf*4){
   int f=k/4,h=k%4+1;float real=0,imag=0;
   for(int j=0;j<nt;++j){
     float phase=((float)times[start+j]/1000000.f)*((6.283185307179586f*frequencies[f])*h);
     float s,c;sincosf(phase,&s,&c);float w=2.f*polarity[start+j]-1.f-mean;
     real+=w*c;imag-=w*s;
   }
   zr[k]=real/nt;zi[k]=imag/nt;
 }
 __syncthreads();
 if(k<nf){
   float p=0;for(int h=0;h<4;++h)p+=zr[4*k+h]*zr[4*k+h]+zi[4*k+h]*zi[4*k+h];power[k]=p;
 }
 __syncthreads();
 if(k==0){
   int a=0;for(int f=1;f<nf;++f)if(power[f]>power[a])a=f;
   int b=0;float best=-1.f;
   for(int f=0;f<nf;++f){
     float p=fabsf(logf(frequencies[f]/frequencies[a]))>.2f?power[f]:-1.f;
     if(p>best){best=p;b=f;}
   }
   choice[0]=a;choice[1]=b;
 }
 __syncthreads();
 for(int candidate=0;candidate<2;++candidate){
   double pred_sum=0,pred_sq=0,pol_sum=0,projection=0,first_sum=0,first_sq=0,first_projection=0;int f=choice[candidate];
   for(int j=k;j<nv;j+=blockDim.x){
     float pred=0,first=0;
     for(int h=1;h<=4;++h){
       float phase=(6.283185307179586f*(float)times[start+nt+j]/1000000.f*frequencies[f])*h;
       float s,c;sincosf(phase,&s,&c);float term=zr[4*f+h-1]*c-zi[4*f+h-1]*s;
       pred+=term;if(h==1)first=term;
     }
     double pol=2.f*polarity[start+nt+j]-1.f;
     pred_sum+=pred;pred_sq+=(double)pred*pred;pol_sum+=pol;projection+=pol*pred;
     first_sum+=first;first_sq+=(double)first*first;first_projection+=pol*first;
   }
#ifdef ROTOR_WARP_REDUCTION
   stats[candidate*5]=pred_sum;stats[candidate*5+1]=pred_sq;
   stats[candidate*5+2]=pol_sum;stats[candidate*5+3]=projection;
   stats[10+candidate*3]=first_sum;stats[11+candidate*3]=first_sq;stats[12+candidate*3]=first_projection;
#else
   reduction[candidate*5][k]=pred_sum;reduction[candidate*5+1][k]=pred_sq;
   reduction[candidate*5+2][k]=pol_sum;reduction[candidate*5+3][k]=projection;
   reduction[10+candidate*3][k]=first_sum;reduction[11+candidate*3][k]=first_sq;reduction[12+candidate*3][k]=first_projection;
#endif
 }
#ifdef ROTOR_WARP_REDUCTION
 for(int j=0;j<16;++j){
   if(j==4||j==9)continue;
   double value=stats[j];
   for(int offset=16;offset;offset>>=1)value+=__shfl_down_sync(0xffffffff,value,offset);
   if((k&31)==0)reduction[j][k>>5]=value;
 }
 __syncthreads();
 if(k<32){
   for(int j=0;j<16;++j){
     if(j==4||j==9)continue;
     double value=k<8?reduction[j][k]:0.;
     for(int offset=16;offset;offset>>=1)value+=__shfl_down_sync(0xffffffff,value,offset);
     if(k==0)reduction[j][0]=value;
   }
 }
 __syncthreads();
#else
 __syncthreads();
 for(int step=128;step;step>>=1){
   if(k<step){
     for(int c=0;c<2;++c)for(int j=0;j<4;++j)reduction[c*5+j][k]+=reduction[c*5+j][k+step];
     for(int j=10;j<16;++j)reduction[j][k]+=reduction[j][k+step];
   }
   __syncthreads();
 }
#endif
 if(k==0){
   float score[2];for(int c=0;c<2;++c){
     double a=reduction[c*5][0],energy=reduction[c*5+1][0]-a*a/nv;
     double p=reduction[c*5+3][0]-reduction[c*5+2][0]*a/nv;
     score[c]=p>0?(float)(p*p/(2*fmax(energy,1e-12))):0.f;
   }
   int best=score[1]>score[0]?1:0;float value=score[best];
   int accepted=covered&&(2.f*expf(-value)<=.01f);
   float* dst=output+pixel*12;dst[0]=(float)pixels[pixel];
   dst[1]=accepted?frequencies[choice[best]]:0.f;dst[2]=value;
   dst[3]=frequencies[choice[0]];dst[4]=frequencies[choice[1]];dst[5]=(float)accepted;
   for(int c=0;c<2;++c){
     double sum=reduction[10+c*3][0];
     dst[6+c*2]=covered?(float)(reduction[12+c*3][0]-reduction[c*5+2][0]*sum/nv):0.f;
     dst[7+c*2]=covered?(float)fmax(reduction[11+c*3][0]-sum*sum/nv,0.):0.f;
   }
   dst[10]=score[0];dst[11]=score[1];
 }
}

extern "C" __global__ void event_counts(const unsigned long long* events,int n,
 long long begin,long long end,int width,int height,int* counts,int* halves,int* counts20){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 unsigned long long word=events[2*i];long long time=(long long)events[2*i+1];
 int x=word&65535,y=(word>>16)&65535,p=((word>>32)&65535)>0;
 if(time<begin||time>=end||x>=width||y>=height)return;
 int pixel=y*width+x,size=width*height,half=(time-begin)*2>=end-begin;
 atomicAdd(counts+p*size+pixel,1);atomicAdd(halves+half*size+pixel,1);
 if(time>=end-20000)atomicAdd(counts20+p*size+pixel,1);
}
extern "C" __global__ void density_horizontal(const int* counts,int width,int height,int* sums){
 int i=blockIdx.x*blockDim.x+threadIdx.x,size=width*height;if(i>=size)return;
 int x=i%width,y=i/width,sum=0;
 for(int d=-15;d<=15;++d){int k=x+d;k=k<0?0:k>=width?width-1:k;sum+=counts[y*width+k]+counts[size+y*width+k];}
 sums[i]=sum;
}
extern "C" __global__ void density_vertical(const int* horizontal,const int* counts,const int* halves,
 int width,int height,int* sums,unsigned char* dense,unsigned char* eligible,int minimum_half){
 int i=blockIdx.x*blockDim.x+threadIdx.x,size=width*height;if(i>=size)return;
 int x=i%width,y=i/width,sum=0;
 for(int d=-15;d<=15;++d){int k=y+d;k=k<0?0:k>=height?height-1:k;sum+=horizontal[k*width+x];}
 int total=counts[i]+counts[size+i];bool d=total>=6&&(long long)total*961>=2LL*sum;
 sums[i]=sum;dense[i]=d;eligible[i]=d&&halves[i]>=minimum_half&&halves[size+i]>=minimum_half;
}
extern "C" __global__ void gather_pixel_events(const unsigned long long* events,int n,
 long long begin,long long end,int width,int height,const unsigned char* eligible,
 const long long* offsets,const int* halves,int* cursors,long long* times,short* polarity){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 unsigned long long word=events[2*i];long long time=(long long)events[2*i+1];
 int x=word&65535,y=(word>>16)&65535,p=((word>>32)&65535)>0;
 if(time<begin||time>=end||x>=width||y>=height)return;
 int pixel=y*width+x,size=width*height;if(!eligible[pixel])return;
 int half=(time-begin)*2>=end-begin;
 long long pos=offsets[pixel]+(half?halves[pixel]:0)+atomicAdd(cursors+half*size+pixel,1);
 times[pos]=time-begin;polarity[pos]=(short)p;
}

__device__ int reflect101(int i,int n){return i<0?-i:i>=n?2*n-2-i:i;}
extern "C" __global__ void native_blur_horizontal(const int* counts,float* out,const float* g,int width,int height){
 int i=blockIdx.x*blockDim.x+threadIdx.x,size=width*height;if(i>=2*size)return;
 int p=i%size,x=p%width,base=i-p+(p/width)*width;float sum=0;
 for(int d=-1;d<=1;++d)sum+=(float)counts[base+reflect101(x+d,width)]*g[d+1];
 out[i]=sum;
}
extern "C" __global__ void native_feature_vertical(const float* in,const int* counts,float* out,const float* g,int width,int height){
 int i=blockIdx.x*blockDim.x+threadIdx.x,size=width*height;if(i>=size)return;
 int x=i%width,y=i/width;float a=0,b=0;
 for(int d=-1;d<=1;++d){int p=reflect101(y+d,height)*width+x;a+=in[p]*g[d+1];b+=in[size+p]*g[d+1];}
 out[i]=counts[i]+counts[size+i]>=2?fminf(a,b):0.f;
}
extern "C" __global__ void native_feature_horizontal(const float* in,float* out,const float* g,int width,int height){
 int i=blockIdx.x*blockDim.x+threadIdx.x,size=width*height;if(i>=size)return;
 int x=i%width,base=(i/width)*width;float sum=0;
 for(int d=-1;d<=1;++d)sum+=in[base+reflect101(x+d,width)]*g[d+1];
 out[i]=sum;
}
extern "C" __global__ void native_support_vertical(const float* in,unsigned char* out,const float* g,int width,int height){
 int i=blockIdx.x*blockDim.x+threadIdx.x,size=width*height;if(i>=size)return;
 int x=i%width,y=i/width;float sum=0;
 for(int d=-1;d<=1;++d)sum+=in[reflect101(y+d,height)*width+x]*g[d+1];
 out[i]=sum>=4.f;
}
extern "C" __global__ void native_binary_morph(const unsigned char* in,unsigned char* out,int width,int height,int k,int erode){
 int i=blockIdx.x*blockDim.x+threadIdx.x,size=width*height;if(i>=size)return;
 int x=i%width,y=i/width,left=k/2;bool value=erode;
 for(int dy=-left;dy<k-left;++dy)for(int dx=-left;dx<k-left;++dx){
   int X=x+dx,Y=y+dy;bool v=(X<0||X>=width||Y<0||Y>=height)?(bool)erode:(bool)in[Y*width+X];
   value=erode?(value&&v):(value||v);
 }
 out[i]=value;
}
extern "C" __global__ void native_period_seeds(const unsigned char* strict,const unsigned char* support,unsigned char* out,int width,int height){
 int i=blockIdx.x*blockDim.x+threadIdx.x,size=width*height;if(i>=size)return;
 if(!support[i]){out[i]=0;return;}int x=i%width,y=i/width;bool value=false;
 for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx){
   if(abs(dx)+abs(dy)>2)continue;int X=x+dx,Y=y+dy;
   if(X>=0&&X<width&&Y>=0&&Y<height)value=value||(bool)strict[Y*width+X];
 }
 out[i]=value;
}
