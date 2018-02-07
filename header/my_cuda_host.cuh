#ifndef _MY_CUDA_HOST
#define _MY_CUDA_HOST

#include <iostream>
#include <string>
#include "my_gettime.hpp"
//+++++++++++++++++++++++++++++++++++
//  cuda event class
//   time -> "ms"(default) or "s"
//+++++++++++++++++++++++++++++++++++

void printCudaLastError(){
    cudaError_t err=cudaGetLastError();
    printf("cudaGetLastError::%s(code:%d)\n",cudaGetErrorString(err),err);
}

void checkCudaStatus(){
    cudaError_t err=cudaGetLastError();
    if(err){
        printf("checkCudaStatus::%s(code:%d)\n",cudaGetErrorString(err),err);
        exit(0);
    }
}

int get_sm_count(){
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	return props.multiProcessorCount;
}

__global__ void gpuBoostKernel(int* GIN, int *GOUT){
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int val = GIN[tx];
    __shared__ int s_buf[32];
    for(int k=0;k<128;k++){
        for(int i=1;i<32;i=i*2){
            val += s_buf[(threadIdx.x-i)&0x1F];
            s_buf[threadIdx.x] = val;
        }
    }
    GOUT[tx] = val;
}
void GPUBoost(size_t loopMax){
    if(loopMax==0) return;
    double start = get_time_sec();
    int numthreads=32; int numblocks=1024*1024; int num_items = numthreads*numblocks;
    int *d_in, *d_out;
	cudaMalloc((void **)&d_in,sizeof(int)*num_items);
    cudaMalloc((void **)&d_out,sizeof(int)*num_items);
    int Interval=(loopMax>100)?loopMax/100:1;
    for(size_t i=1;i<=loopMax;i++){
        if(i%Interval==0){
            fprintf(stderr,"\rBoost GPU ... [%3d %] %6.2f,sec",i*100/loopMax,get_time_sec()-start); fflush(stdout);fflush(stderr);
        }
        cudaThreadSynchronize();
        gpuBoostKernel  <<< numblocks , numthreads >>> (d_in,d_out);
        cudaThreadSynchronize();
    }
    cudaFree(d_in); cudaFree(d_out);
    fprintf(stderr,"\rBoost GPU ... [%3d %] %6.2f,sec\n",100,get_time_sec()-start); fflush(stdout);fflush(stderr);
    //fprintf(stderr,"\nGPUBoost end,%f,s\n",end-start); fflush(stdout);fflush(stderr);
}


template<class T>
float occupancy(T func,size_t threadNum){
	int numBlocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocks, func , threadNum , 0);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	return ((float)numBlocks*threadNum/props.maxThreadsPerMultiProcessor);
}

template<class T>
float occupancy(T func,size_t threadNum, size_t dynamicSharedAmount){
	int numBlocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocks, func , threadNum , dynamicSharedAmount);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	return ((float)numBlocks*threadNum/props.maxThreadsPerMultiProcessor);
}

template<class T>
size_t get_activeblock_per_device(T func,size_t threadNum){
    int numBlocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocks, func , threadNum , 0);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	return numBlocks * props.multiProcessorCount;
}

template<class T>
size_t get_activeblock_per_device(T func,size_t threadNum, size_t dynamicSharedAmount){
    int numBlocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocks, func , threadNum , dynamicSharedAmount);
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	return numBlocks * props.multiProcessorCount;
}

#define __DEFAULT_CUDA_TIME_UNIT "ms"
#define __DEFAULT_CUDA_MAX_EVENT_NUM 10
class cudatimeStamp{
public:
    cudatimeStamp();
    cudatimeStamp(int i);
    cudatimeStamp(int i,std::string time_unit);
    ~cudatimeStamp();
    void operator()();//{ sec_vec[timeCount++]=get_time_sec();};
    void stamp();
    void stamp(int i);
    void sync();
    void print();
    void print_hori();
    void setunit(std::string time_unit);
    std::string getunit();
    float getxrate();
    float interval(int i,int j);
    float interval(int i);
    int getindex();
private:
    void initialize(int i,std::string time_unit);
	int limit;
    int index;
    int syncflag;
    cudaEvent_t *start;
    float       *elapsedTime;
    cudaEvent_t *s;
    float *e;
    std::string unit;//"ms"(default) or "s"
    float xrate;
};
cudatimeStamp::cudatimeStamp(){
    initialize(__DEFAULT_CUDA_MAX_EVENT_NUM,__DEFAULT_CUDA_TIME_UNIT);
}
cudatimeStamp::cudatimeStamp(int i){
    initialize(i,__DEFAULT_CUDA_TIME_UNIT);
}
cudatimeStamp::cudatimeStamp(int i,std::string time_unit){
    initialize(i,time_unit);
}
cudatimeStamp::~cudatimeStamp(){
    delete start;
    delete elapsedTime;
}
void cudatimeStamp::initialize(int i,std::string time_unit){
    limit   =i;
    index   =0;
    syncflag=0;
    start = new cudaEvent_t [limit];
    setunit(time_unit);
    if(start==NULL){
        fprintf(stderr,"cudatimeStamp::allocation error\n");
        exit(1);
    }
    elapsedTime = new float [limit];

    if(elapsedTime==NULL){
        fprintf(stderr,"cudatimeStamp::allocation error\n");
        exit(1);
    }
	for(int i=0;i<limit;i++){
        cudaEventCreate(&start[i]);
        elapsedTime[i]=0.0;
	}
}
void cudatimeStamp::setunit(std::string time_unit){
    std::cout.precision(6);
    std::cout.setf(std::ios::fixed,std::ios::floatfield);
    if(time_unit=="s"){
        unit="s";
        xrate=0.0001;
    }else{
        unit="ms";
        xrate=1.0;
    }
}
inline void cudatimeStamp::stamp(){
    cudaEventRecord(start[index++],0);
}
inline void cudatimeStamp::stamp(int i){
    if(i==1){
        cudaError_t err=cudaGetLastError();
        if(err){
            printf("Error::cudatimeStamp::sync::%s(code:%d)\n",cudaGetErrorString(err),err);
            exit(0);
        }
    }
    cudaEventRecord(start[index++],0);
}
void cudatimeStamp::operator()(){
    cudaEventRecord(start[index++],0);
}
void cudatimeStamp::sync(){
    cudaThreadSynchronize();
    if(syncflag) return;

	for(int i=0;i<index;i++)
	 	cudaEventSynchronize(start[i]);
	for(int i=0;i<index-1;i++)
	 	cudaEventElapsedTime(&elapsedTime[i], start[i], start[i+1]);
    syncflag=1;
    cudaError_t err=cudaGetLastError();
    if(err) printf("warning::cudatimeStamp::sync::%s(code:%d)\n",cudaGetErrorString(err),err);
}
float cudatimeStamp::interval(int i){
    if(syncflag==0) this->sync();
    return elapsedTime[i]*xrate;
}
float cudatimeStamp::interval(int i,int j){
    if(syncflag==0) this->sync();
    float v;
    cudaEventElapsedTime(&v,start[i],start[j]);
    return v*xrate;
}
int cudatimeStamp::getindex(){
    return index;
}
float cudatimeStamp::getxrate(){
    return xrate;
}
std::string cudatimeStamp::getunit(){
    return unit;
}
void cudatimeStamp::print(){
    if(syncflag==0) this->sync();
	for(int i=0;i<index-1;i++)
        std::cout<<interval(i)<<","<<unit<<"\n";
}
void cudatimeStamp::print_hori(){
    if(syncflag==0) this->sync();
	for(int i=0;i<index-2;i++)
        std::cout<<interval(i)<<","<<unit<<",";
    std::cout<<interval(index-2)<<","<<unit<<",";
}
std::ostream& operator<<(std::ostream& os, cudatimeStamp &cuts){
	for(int i=0;i<cuts.getindex()-1;i++)
       std::cout<<cuts.interval(i)<<" , "<<cuts.getunit()<<" \n";
	return os;
}

#endif
