//
// cooperative_group
//   -> nvcc ... -rdc = true  and require TCC mode
//
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <typeinfo>
#include "header/my_gettime.hpp"
#include "header/my_cuda_host.cuh"

#define LOOPCOUNT (64)

typedef unsigned long long int uint64;
typedef long long int int64;

#define NUMCOPYWARPS (NUMTHREADS/TILEWIDTH)

// ZIGZAG COPY KERNEL
template <typename T ,unsigned int TILEWIDTH  ,unsigned int NUMTHREADS>
__global__ void zigzagCopy(T* G_IN, T *G_OUT, uint64 *XYPOS, unsigned int INPUTWIDTH){
	
	unsigned int copylaneId = threadIdx.x % TILEWIDTH;
	unsigned int copywarpId = threadIdx.x / TILEWIDTH;

	unsigned int tileId = blockIdx.x;
	
	extern __shared__ T s_buffer[];
	int X;
	int Y;
	asm volatile("mov.b64 {%0, %1}, %2;" : "=r"(Y), "=r"(X) : "l"( XYPOS[tileId] ));

	G_IN  = G_IN  + X*TILEWIDTH + Y*TILEWIDTH*INPUTWIDTH;
	G_OUT = G_OUT + X*TILEWIDTH + Y*TILEWIDTH*INPUTWIDTH;
	
	//if(threadIdx.x==0) printf("%d %d %d\n",blockIdx.x,X,Y);
	
	for(int y=0;y<TILEWIDTH;y=y+NUMCOPYWARPS){
		s_buffer[(y+copywarpId)*TILEWIDTH+copylaneId]  = G_IN[(y+copywarpId)*INPUTWIDTH +copylaneId];
	}
	__syncthreads();
	for(int y=0;y<TILEWIDTH;y=y+NUMCOPYWARPS){
		G_OUT[(y+copywarpId)*INPUTWIDTH +copylaneId] = s_buffer[(y+copywarpId)*TILEWIDTH+copylaneId];
	}

}

template <typename T ,unsigned int TILEWIDTH  ,unsigned int NUMTHREADS>
void execGPUkernel(size_t input_width){
	cudaProfilerStart();
	cudatimeStamp cudatimer(10);
	//printf("%d\n",CUDART_VERSION);

	size_t numTiles = (input_width / TILEWIDTH) * (input_width / TILEWIDTH);
	size_t numblocks = numTiles;
	size_t num_items = input_width*input_width;
	
	// malloc
	T *d_in;
	cudaMalloc((void **)&d_in,sizeof(T)*num_items);
	T *d_out;
	cudaMalloc((void **)&d_out,sizeof(T)*num_items);
	uint64 *d_xy_pos;
	cudaMalloc((void **)&d_xy_pos,sizeof(uint64)*numTiles );


	//compute TILE index
	uint64 *xy_pos = (uint64 *)malloc(sizeof(uint64)*numTiles );
	int64 tw = input_width/TILEWIDTH;
    int64 th = input_width/TILEWIDTH;
 	int64 index = 0;
	for(int64 k=0;k<=tw+th-2;k++){
		int64 x = (k < tw-1) ? k : tw-1;
		int64 y = (0 > k-tw+1) ? 0 :  k-tw+1;
		for(int64 l=0;l<((x+1<th-y)?x+1:th-y);l++){
			xy_pos[index] = ((x-l)<<32ll)|(y+l);
			index++;
		}
	}


	// Change dynamic Shared memory size (CC7.0~)
	size_t sharedSize = TILEWIDTH * TILEWIDTH * sizeof(T);
	if(sharedSize > 48*1024)
		cudaFuncSetAttribute(zigzagCopy <T,TILEWIDTH,NUMTHREADS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedSize);
		

	cudaMemcpy(d_xy_pos,xy_pos,sizeof(uint64)*numTiles ,cudaMemcpyHostToDevice);
	checkCudaStatus();
	
	cudatimer.stamp();
	for(int i=0;i<LOOPCOUNT;i++)
		zigzagCopy <T,TILEWIDTH,NUMTHREADS> <<< numblocks , NUMTHREADS , sharedSize >>> (d_in, d_out, d_xy_pos, input_width);

	cudatimer.stamp();
	//memcpy Device->Deivce
	for(int i=0;i<LOOPCOUNT;i++)
		cudaMemcpy(d_out,d_in,sizeof(T)*num_items,cudaMemcpyDeviceToDevice);
	cudatimer.stamp();
	
	printf("input,%zux%zu,tile,%dx%d,numthreads,%d,numblocks,%zu,type,%s,zigzagcopy,%f,ms,cudaMemcpy,%f,ms,",
		input_width,input_width,TILEWIDTH,TILEWIDTH,NUMTHREADS,numblocks,typeid(T).name(),cudatimer.interval(0,1)/LOOPCOUNT,cudatimer.interval(1,2)/LOOPCOUNT);

	printf("occupancy,%4.3f,SMcount,%d,activeblock,%d,",
		occupancy(zigzagCopy<T,TILEWIDTH,NUMTHREADS>,NUMTHREADS,sharedSize),
		get_sm_count(),
		get_activeblock_per_device(zigzagCopy<T,TILEWIDTH,NUMTHREADS>,NUMTHREADS,sharedSize) );

	printCudaLastError();
	fflush(stdout);
	
	//memory free
	free(xy_pos);
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_xy_pos);
	cudaProfilerStop();

}

int main(int argc,char **argv){
	
	GPUBoost(64);

	for(size_t n=256;n<32*1024;n=n*2)
		execGPUkernel<float,  32, 512>(n); // type:float, tilesize:32x32, the numeber of threads:512
	for(size_t n=256;n<=32*1024;n=n*2)
		execGPUkernel<float,  64, 512>(n);
    // Change dynamic Shared memory size (CC7.0~)
	for(size_t n=256;n<=32*1024;n=n*2)
		execGPUkernel<float, 128, 512>(n);

	return 0;
}
