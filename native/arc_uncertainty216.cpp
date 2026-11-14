// Same ten central differences, point subset, residual MAD and SVD cutoff.
#include "arc_search215.cpp"
namespace {
double median216(std::vector<double> a){
 size_t k=a.size()/2;std::nth_element(a.begin(),a.begin()+k,a.end());double upper=a[k];
 if(a.size()%2)return upper;
 return (*std::max_element(a.begin(),a.begin()+k)+upper)*.5;
}
}
extern "C" __declspec(dllexport) int arc_uncertainty216(const double* points,int n,const double* p,double* out){
 if(!points||!p||!out||n<3)return -1;
 int m=std::min(n,96);std::vector<double> q(m*2);
 for(int i=0;i<m;++i){int index=n>96?(int)(i*((n-1)/95.)):i;if(i==m-1)index=n-1;
  q[2*i]=points[2*index];q[2*i+1]=points[2*index+1];}
 std::vector<double> residual(m);ellipse_distance_batch(q.data(),m,p,1,residual.data());
 double center=median216(residual);for(double& v:residual)v=std::abs(v-center);
 double sigma=std::max(.8,1.4826*median216(residual));double steps[5]={.05,.05,.05,.05,.001};
 double parameters[50];for(int i=0;i<10;++i)std::copy(p,p+5,parameters+5*i);
 for(int j=0;j<5;++j){parameters[10*j+j]+=steps[j];parameters[10*j+5+j]-=steps[j];}
 std::vector<double> distances(10*m);ellipse_distance_batch(q.data(),m,parameters,10,distances.data());
 cv::Mat jac(m,5,CV_64F);
 for(int i=0;i<m;++i)for(int j=0;j<5;++j)jac.at<double>(i,j)=(distances[2*j*m+i]-distances[(2*j+1)*m+i])/(2*steps[j]);
 cv::Mat gram=jac.t()*jac;cv::SVD decomp(gram,cv::SVD::FULL_UV);cv::Mat inv=cv::Mat::zeros(5,5,CV_64F);
 double cutoff=decomp.w.at<double>(0)*1e-10;
 for(int j=0;j<5;++j)if(decomp.w.at<double>(j)>cutoff)inv.at<double>(j,j)=1/decomp.w.at<double>(j);
 cv::Mat covariance=(decomp.vt.t()*inv*decomp.u.t())*(sigma*sigma*4);
 cv::Mat scaled=jac.clone();for(int j=0;j<4;++j)scaled.col(j)*=p[2];
 cv::Mat singular;cv::SVD::compute(scaled,singular,cv::SVD::NO_UV);
 double a=covariance.at<double>(0,0),d=covariance.at<double>(1,1),b=covariance.at<double>(1,0);
 double eigen=(a+d)*.5+std::sqrt((a-d)*(a-d)*.25+b*b);
 out[0]=std::min(500.,std::sqrt(std::max(eigen,.25)));
 out[1]=std::min(90.,std::sqrt(std::max(covariance.at<double>(4,4),0.))*57.29577951308232);
 out[2]=singular.at<double>(0)/std::max(singular.at<double>(4),1e-9);
 return 0;
}
