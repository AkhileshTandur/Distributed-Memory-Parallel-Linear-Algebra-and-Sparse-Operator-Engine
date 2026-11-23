#include <mpi.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>
struct CSR{std::vector<int> row,col;std::vector<double> val;};
// Tridiagonal matrix: remote dependencies are only adjacent rank boundaries.
CSR matrix(int begin,int end,int n){CSR a;a.row.push_back(0);for(int i=begin;i<end;++i){for(int j=std::max(0,i-1);j<=std::min(n-1,i+1);++j){a.col.push_back(j);a.val.push_back(j==i?2:-1);}a.row.push_back(a.col.size());}return a;}
void spmv(const CSR&a,const std::vector<double>&x,std::vector<double>&y,int start,int rank,int size,MPI_Comm comm){
 int count=x.size();double left=0,right=0;MPI_Request req[4];int q=0;
 if(rank>0){MPI_Irecv(&left,1,MPI_DOUBLE,rank-1,10,comm,&req[q++]);MPI_Isend(x.data(),1,MPI_DOUBLE,rank-1,11,comm,&req[q++]);}
 if(rank+1<size){MPI_Irecv(&right,1,MPI_DOUBLE,rank+1,11,comm,&req[q++]);MPI_Isend(x.data()+count-1,1,MPI_DOUBLE,rank+1,10,comm,&req[q++]);}
 auto row=[&](int r){double s=0;for(int z=a.row[r];z<a.row[r+1];++z){int j=a.col[z]-start;s+=a.val[z]*(j<0?left:(j>=count?right:x[j]));}y[r]=s;};
 #pragma omp parallel for
 for(int r=1;r<count-1;++r)row(r);
 MPI_Waitall(q,req,MPI_STATUSES_IGNORE);row(0);if(count>1)row(count-1);
}
int main(int argc,char**argv){int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 if(provided<MPI_THREAD_FUNNELED)MPI_Abort(MPI_COMM_WORLD,3);
 int n=argc>1?std::atoi(argv[1]):10003,reps=argc>2?std::atoi(argv[2]):100;
 if(n<size||reps<1){if(rank==0)std::cerr<<"usage: spmv [N >= ranks] [positive repetitions]\n";MPI_Finalize();return 2;}
 int begin=(long long)n*rank/size,end=(long long)n*(rank+1)/size;auto a=matrix(begin,end,n);std::vector<double>x(end-begin),y(end-begin);
 for(int i=begin;i<end;++i)x[i-begin]=std::sin(i*.01);
 spmv(a,x,y,begin,rank,size,MPI_COMM_WORLD);double err=0;
 for(int i=begin;i<end;++i){double ref=2*std::sin(i*.01)-(i>0?std::sin((i-1)*.01):0)-(i+1<n?std::sin((i+1)*.01):0);err=std::max(err,std::abs(y[i-begin]-ref));}
 double maxerr;MPI_Allreduce(&err,&maxerr,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
 MPI_Barrier(MPI_COMM_WORLD);double start=MPI_Wtime();for(int i=0;i<reps;++i)spmv(a,x,y,begin,rank,size,MPI_COMM_WORLD);double elapsed=(MPI_Wtime()-start)/reps,worst;
 MPI_Reduce(&elapsed,&worst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
 if(rank==0)std::cout<<"{\"n\":"<<n<<",\"ranks\":"<<size<<",\"threads_per_rank\":"<<omp_get_max_threads()<<",\"seconds\":"<<worst<<",\"max_abs_error\":"<<maxerr<<"}\n";
 MPI_Finalize();return maxerr>1e-12?1:0;}
