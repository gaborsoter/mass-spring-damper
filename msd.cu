#include <stdio.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream> 
void checkCUDAError(const char *msg);
#include <cuda_runtime.h>
using namespace std;


// --------------------INPUT DATA---------------------
const int Nx = 24, Ny = 120, Nz = 20; // Number of mass points
float maxtime = 60; // End time [sec]
const int Nstep = 1200; // Number of time steps
__device__ __constant__ float dt = 0.05; // maxtime / Nstep; // Time step size [sec]
float dtcpu=0.05;

const int xlength =  (4 + 2 * Nx)*(2 + Ny)*(2 + Nz); // Solution array in x-direction
const int ylength =  (2 + Nx)*(4 + 2 * Ny)*(2 + Nz); // Solution array in y-direction
const int zlength =  (2 + Nx)*(2 + Ny)*(4 + 2 * Nz); // Solution array in z-direction
const int masslength = Nx * Ny * Nz;
const int kxlength = Nz * Ny * (Nx + 1);
const int kylength = Nz * Nx * (Ny + 1);
const int kzlength = Ny * Nx * (Nz + 1);
const int bxlength = kxlength;
const int bylength = kylength;
const int bzlength = kzlength;






//------------------------DEVICE FUNCTIONS----------------------------//
//x-displacement
__device__ float fxx(int n, int i, int j, int k, float*xold)
{
	return xold[ (Ny + 2)*(4 + 2*Nx) + (k - 1)*(Ny + 2)*(4 + 2*Nx) + 4 +
		2*Nx + (i - 1)*(4 + 2*Nx) + 2 + (2*j - 1)-1];
}

//x-velocity
__device__ float fvx(int n, int i, int j, int k, float*xold)
{
	return xold[ (Ny + 2)*(4 + 2*Nx) + (k - 1)*(Ny + 2)*(4 + 2*Nx) + 4 +
		2*Nx + (i - 1)*(4 + 2*Nx) + 2 + (2*j)-1];
}

//y-displacement
__device__ float fyy(int n, int i, int j, int k, float*yold)
{
	return yold[ (Nx + 2)*(4 + 2*Ny) + (k - 1)*(Nx + 2)*(4 + 2*Ny) + 4 +
		2*Ny + (j - 1)*(4 + 2*Ny) + 2 + (2*i - 1)-1];
}

//y-velocity
__device__ float fvy(int n, int i, int j, int k, float*yold)
{
	return yold[ (Nx + 2)*(4 + 2 * Ny) + (k - 1)*(Nx + 2)*(4 + 2 * Ny) + 4 +
		2 * Ny + (j - 1)*(4 + 2 * Ny) + 2 + (2 * i)-1];
}

//z-displacement
__device__ float fzz(int n, int i, int j, int k, float*zold)
{
	return zold[ (Nx + 2)*(4 + 2*Nz) + (i - 1)*(Nx + 2)*(4 + 2*Nz) + 4 +
		2*Nz + (j - 1)*(4 + 2*Nz) + 2 + (2*k - 1)-1];
}

//z-velocity
__device__ float fvz(int n, int i, int j, int k, float*zold)
{
	return zold[ (Nx + 2)*(4 + 2 * Nz) + (i - 1)*(Nx + 2)*(4 + 2 * Nz) + 4 +
		2 * Nz + (j - 1)*(4 + 2 * Nz) + 2 + (2 * k)-1];
}

//mass
__device__ float fm(int i, int j, int k, float*m)
{
	return m[(k - 1)*Ny*Nx + (i - 1)*Nx + j-1];
}

//x-stiffness
__device__ float fkx(int i, int j, int k, float*kx)
{
	return kx[(k - 1)*Ny*(Nx + 1) + (i - 1)*(Nx + 1) + j-1];
}

//y-stiffness
__device__ float fky(int i, int j, int k, float*ky)
{
	return ky[(k - 1)*Nx*(Ny + 1) + (i - 1)*Nx + j-1];
}

//z-stiffness
__device__ float fkz(int i, int j, int k, float*kz)
{
	return kz[(k - 1)*Nx*Ny + (i - 1)*Nx + j-1];
}

//x-damping
__device__ float fbx(int i, int j, int k, float*bx)
{
	return bx[(k - 1)*Ny*(Nx + 1) + (i - 1)*(Nx + 1) + j-1];
}

//y-damping
__device__ float fby(int i, int j, int k, float*by)
{
	return by[(k - 1)*Nx*(Ny + 1) + (i - 1)*Nx + j-1];
}

//z-damping
__device__ float fbz(int i, int j, int k, float*bz)
{
	return bz[(k - 1)*Nx*Ny + (i - 1)*Nx + j-1];
}

//x-force
__device__ float fFx(int i, int j, int k, float*Fx)
{
	return Fx[(k - 1)*Ny*Nx + (i - 1)*Nx + j-1];
}

//y-force
__device__ float fFy(int i, int j, int k, float*Fy)
{
	return Fy[(k - 1)*Ny*Nx + (i - 1)*Nx + j-1];
}

//z-force
__device__ float fFz(int i, int j, int k, float*Fz)
{
	return Fz[(k - 1)*Ny*Nx + (i - 1)*Nx + j-1];
}

//x-acceleration
__device__ float ax(int i, int j, int k, float*Fx, float*xold, float*kx, float*ky, float*kz, float*bx, float*by, float*bz, float*m)
{
	return (fFx(i, j, k, Fx) - fby(i, j, k, by)*(-fvx(1, -1 + i, j, k, xold) + fvx(1, i, j, k, xold)) -
		fbx(i, j, k, bx)*(-fvx(1, i, -1 + j, k, xold) + fvx(1, i, j, k, xold)) - fbz(i, j, k, bz)*(-fvx(1, i, j, -1 + k, xold) + fvx(1, i, j, k, xold)) +
		fbz(i, j, 1 + k, bz)*(-fvx(1, i, j, k, xold) + fvx(1, i, j, 1 + k, xold)) +
		fbx(i, 1 + j, k, bx)*(-fvx(1, i, j, k, xold) + fvx(1, i, 1 + j, k, xold)) +
		fby(1 + i, j, k, by)*(-fvx(1, i, j, k, xold) + fvx(1, 1 + i, j, k, xold)) -
		fky(i, j, k, ky)*(-fxx(1, -1 + i, j, k, xold) + fxx(1, i, j, k, xold)) - fkx(i, j, k, kx)*(-fxx(1, i, -1 + j, k, xold) + fxx(1, i, j, k, xold)) -
		fkz(i, j, k, kz)*(-fxx(1, i, j, -1 + k, xold) + fxx(1, i, j, k, xold)) +
		fkz(i, j, 1 + k, kz)*(-fxx(1, i, j, k, xold) + fxx(1, i, j, 1 + k, xold)) +
		fkx(i, 1 + j, k, kx)*(-fxx(1, i, j, k, xold) + fxx(1, i, 1 + j, k, xold)) +
		fky(1 + i, j, k, ky)*(-fxx(1, i, j, k, xold) + fxx(1, 1 + i, j, k, xold))) / fm(i, j, k, m);
}

//y-acceleration
__device__ float ay(int i, int j, int k, float*Fy, float*yold, float*kx, float*ky, float*kz, float*bx, float*by, float*bz, float*m)
{
	return (fFy(i, j, k, Fy) - fby(i, j, k, by)*(-fvy(1, -1 + i, j, k, yold) + fvy(1, i, j, k, yold)) -
		fbx(i, j, k, bx)*(-fvy(1, i, -1 + j, k, yold) + fvy(1, i, j, k, yold)) - fbz(i, j, k, bz)*(-fvy(1, i, j, -1 + k, yold) + fvy(1, i, j, k, yold)) +
		fbz(i, j, 1 + k, bz)*(-fvy(1, i, j, k, yold) + fvy(1, i, j, 1 + k, yold)) +
		fbx(i, 1 + j, k, bx)*(-fvy(1, i, j, k, yold) + fvy(1, i, 1 + j, k, yold)) +
		fby(1 + i, j, k, by)*(-fvy(1, i, j, k, yold) + fvy(1, 1 + i, j, k, yold)) -
		fky(i, j, k, ky)*(-fyy(1, -1 + i, j, k, yold) + fyy(1, i, j, k, yold)) - fkx(i, j, k, kx)*(-fyy(1, i, -1 + j, k, yold) + fyy(1, i, j, k, yold)) -
		fkz(i, j, k, kz)*(-fyy(1, i, j, -1 + k, yold) + fyy(1, i, j, k, yold)) +
		fkz(i, j, 1 + k, kz)*(-fyy(1, i, j, k, yold) + fyy(1, i, j, 1 + k, yold)) +
		fkx(i, 1 + j, k, kx)*(-fyy(1, i, j, k, yold) + fyy(1, i, 1 + j, k, yold)) +
		fky(1 + i, j, k, ky)*(-fyy(1, i, j, k, yold) + fyy(1, 1 + i, j, k, yold))) / fm(i, j, k, m);
}

//z-acceleration
__device__ float az(int i, int j, int k, float*Fz, float*zold, float*kx, float*ky, float*kz, float*bx, float*by, float*bz, float*m)
{
	return (fFz(i, j, k, Fz) - fby(i, j, k, by)*(-fvz(1, -1 + i, j, k, zold) + fvz(1, i, j, k, zold)) -
		fbx(i, j, k, bx)*(-fvz(1, i, -1 + j, k, zold) + fvz(1, i, j, k, zold)) - fbz(i, j, k, bz)*(-fvz(1, i, j, -1 + k, zold) + fvz(1, i, j, k, zold)) +
		fbz(i, j, 1 + k, bz)*(-fvz(1, i, j, k, zold) + fvz(1, i, j, 1 + k, zold)) +
		fbx(i, 1 + j, k, bx)*(-fvz(1, i, j, k, zold) + fvz(1, i, 1 + j, k, zold)) +
		fby(1 + i, j, k, by)*(-fvz(1, i, j, k, zold) + fvz(1, 1 + i, j, k, zold)) -
		fky(i, j, k, ky)*(-fzz(1, -1 + i, j, k, zold) + fzz(1, i, j, k, zold)) - fkx(i, j, k, kx)*(-fzz(1, i, -1 + j, k, zold) + fzz(1, i, j, k, zold)) -
		fkz(i, j, k, kz)*(-fzz(1, i, j, -1 + k, zold) + fzz(1, i, j, k, zold)) +
		fkz(i, j, 1 + k, kz)*(-fzz(1, i, j, k, zold) + fzz(1, i, j, 1 + k, zold)) +
		fkx(i, 1 + j, k, kx)*(-fzz(1, i, j, k, zold) + fzz(1, i, 1 + j, k, zold)) +
		fky(1 + i, j, k, ky)*(-fzz(1, i, j, k, zold) + fzz(1, 1 + i, j, k, zold))) / fm(i, j, k, m);
}











__global__ void SolveKernel(int dimBlockX,int dimBlockY,int dimBlockZ,float*xoldd,float*yoldd,float*zoldd,float*xnewd,float*ynewd,float*znewd,float*md,float*kxd,float*kyd,float*kzd,float*bxd,float*byd,float*bzd,float*Fxd,float*Fyd,float*Fzd)
{
//	int tx=threadIdx.x;
//	int ty=threadIdx.y;
	int tx=blockIdx.x*dimBlockX+threadIdx.x;
	int ty=blockIdx.y*dimBlockY+threadIdx.y;
	int tz=blockIdx.z*dimBlockZ+threadIdx.z;
	
	int i=ty+1;
	int j=tx+1;
	int k=tz+1;

	xnewd[ (Ny + 2)*(4 + 2*Nx) + (k - 1)*(Ny + 2)*(4 + 2*Nx) + 4 +
		2 * Nx + (i - 1)*(4 + 2 * Nx) + 2 + (2 * j - 1) - 1] = fxx(1, i, j, k, xoldd) + fvx(1, i, j, k, xoldd)*dt;
	xnewd[ (Ny + 2)*(4 + 2 * Nx) + (k - 1)*(Ny + 2)*(4 + 2 * Nx) + 4 +
		2 * Nx + (i - 1)*(4 + 2 * Nx) + 2 + (2 * j) - 1] = fvx(1, i, j, k, xoldd) + ax(i, j, k, Fxd, xoldd, kxd, kyd, kzd, bxd, byd, bzd, md)*dt;

	ynewd[ (Nx + 2)*(4 + 2*Ny) + (k - 1)*(Nx + 2)*(4 + 2*Ny) + 4 +
		2*Ny + (j - 1)*(4 + 2*Ny) + 2 + (2*i - 1)-1] = fyy(1, i, j, k, yoldd) + fvy(1, i, j, k, yoldd)*dt;
	ynewd[ (Nx + 2)*(4 + 2*Ny) + (k - 1)*(Nx + 2)*(4 + 2*Ny) + 4 +
		2*Ny + (j - 1)*(4 + 2*Ny) + 2 + (2*i)-1] = fvy(1, i, j, k, yoldd) + ay(i, j, k, Fyd, yoldd, kxd, kyd, kzd, bxd, byd, bzd, md)*dt;

	znewd[ (Nx + 2)*(4 + 2*Nz) + (i - 1)*(Nx + 2)*(4 + 2*Nz) + 4 +
		2*Nz + (j - 1)*(4 + 2*Nz) + 2 + (2*k - 1)-1] = fzz(1, i, j, k, zoldd) + fvz(1, i, j, k, zoldd)*dt;
	znewd[ (Nx + 2)*(4 + 2*Nz) + (i - 1)*(Nx + 2)*(4 + 2*Nz) + 4 +
		2*Nz + (j - 1)*(4 + 2*Nz) + 2 + (2*k)-1] = fvz(1, i, j, k, zoldd) + az(i, j, k, Fzd, zoldd, kxd, kyd, kzd, bxd, byd, bzd, md)*dt;
}









void Solve(float*xold,float*yold,float*zold,float*xnew,float*ynew,float*znew,float*m,float*kx,float*ky,float*kz,float*bx,float*by,float*bz,float*Fx,float*Fy,float*Fz)
{
	float *xoldd,*yoldd,*zoldd,*xnewd,*ynewd,*znewd,*md,*kxd,*kyd,*kzd,*bxd,*byd,*bzd,*Fxd,*Fyd,*Fzd;
	
	int sizexoldd=xlength*sizeof(float);
	cudaMalloc((void**)&xoldd,sizexoldd);
	cudaMemcpy(xoldd,xold,sizexoldd,cudaMemcpyHostToDevice);
	int sizeyoldd=ylength*sizeof(float);
	cudaMalloc((void**)&yoldd,sizeyoldd);
	cudaMemcpy(yoldd,yold,sizeyoldd,cudaMemcpyHostToDevice);
	int sizezoldd=zlength*sizeof(float);
	cudaMalloc((void**)&zoldd,sizezoldd);
	cudaMemcpy(zoldd,zold,sizezoldd,cudaMemcpyHostToDevice);
	int sizexnewd=xlength*sizeof(float);
	cudaMalloc((void**)&xnewd,sizexnewd);
	cudaMemcpy(xnewd,xnew,sizexnewd,cudaMemcpyHostToDevice);
	int sizeynewd=ylength*sizeof(float);
	cudaMalloc((void**)&ynewd,sizeynewd);
	cudaMemcpy(ynewd,ynew,sizeynewd,cudaMemcpyHostToDevice);
	int sizeznewd=zlength*sizeof(float);
	cudaMalloc((void**)&znewd,sizeznewd);
	cudaMemcpy(znewd,znew,sizeznewd,cudaMemcpyHostToDevice);
	int sizemd=masslength*sizeof(float);
	cudaMalloc((void**)&md,sizemd);
	cudaMemcpy(md,m,sizemd,cudaMemcpyHostToDevice);
	int sizekxd=kxlength*sizeof(float);
	cudaMalloc((void**)&kxd,sizekxd);
	cudaMemcpy(kxd,kx,sizekxd,cudaMemcpyHostToDevice);
	int sizekyd=kylength*sizeof(float);
	cudaMalloc((void**)&kyd,sizekyd);
	cudaMemcpy(kyd,ky,sizekyd,cudaMemcpyHostToDevice);
	int sizekzd=kzlength*sizeof(float);
	cudaMalloc((void**)&kzd,sizekzd);
	cudaMemcpy(kzd,kz,sizekzd,cudaMemcpyHostToDevice);
	int sizebxd=bxlength*sizeof(float);
	cudaMalloc((void**)&bxd,sizebxd);
	cudaMemcpy(bxd,bx,sizebxd,cudaMemcpyHostToDevice);
	int sizebyd=bylength*sizeof(float);
	cudaMalloc((void**)&byd,sizebyd);
	cudaMemcpy(byd,by,sizebyd,cudaMemcpyHostToDevice);
	int sizebzd=bzlength*sizeof(float);
	cudaMalloc((void**)&bzd,sizebzd);
	cudaMemcpy(bzd,bz,sizebzd,cudaMemcpyHostToDevice);
	int sizeFxd=masslength*sizeof(float);
	cudaMalloc((void**)&Fxd,sizeFxd);
	cudaMemcpy(Fxd,Fx,sizeFxd,cudaMemcpyHostToDevice);
	int sizeFyd=masslength*sizeof(float);
	cudaMalloc((void**)&Fyd,sizeFyd);
	cudaMemcpy(Fyd,Fy,sizeFyd,cudaMemcpyHostToDevice);
	int sizeFzd=masslength*sizeof(float);
	cudaMalloc((void**)&Fzd,sizeFzd);
	cudaMemcpy(Fzd,Fz,sizeFzd,cudaMemcpyHostToDevice);

	//Malloc result
	//cudaMalloc((void**)&Pd,size);
	//Dimensions of the run
	//int SubMtxWidth=SubWidth;
	int NBlockX=4;
	int NBlockY=3;
	int NBlockZ=5;
	int dimBlockX=Nx/NBlockX;
	int dimBlockY=Ny/NBlockY;
	int dimBlockZ=Nz/NBlockZ;
	dim3 dimBlock(dimBlockX,dimBlockY,dimBlockZ);
	dim3 dimGrid(NBlockX,NBlockY,NBlockZ);
	//Running Kernel
	SolveKernel<<<dimGrid,dimBlock>>>(dimBlockX,dimBlockY,dimBlockZ,xoldd,yoldd,zoldd,xnewd,ynewd,znewd,md,kxd,kyd,kzd,bxd,byd,bzd,Fxd,Fyd,Fzd);
	cudaThreadSynchronize();
	//Copy data back
	cudaMemcpy(xnew,xnewd,sizexnewd,cudaMemcpyDeviceToHost);
	cudaMemcpy(ynew,ynewd,sizeynewd,cudaMemcpyDeviceToHost);
	cudaMemcpy(znew,znewd,sizeznewd,cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy");
	//Free memory
	//cudaFree(Md);
	//cudaFree(Nd);
	//cudaFree(Pd);
	//NEWSHIT
	cudaFree(xoldd);
	cudaFree(yoldd);
	cudaFree(zoldd);
	cudaFree(xnewd);
	cudaFree(ynewd);
	cudaFree(znewd);
	cudaFree(md);
	cudaFree(kxd);
	cudaFree(kyd);
	cudaFree(kzd);
	cudaFree(bxd);
	cudaFree(byd);
	cudaFree(bzd);
	cudaFree(Fxd);
	cudaFree(Fyd);
	cudaFree(Fzd);
}
















int main(int argc,char* argv[])
{
float *xold,*yold,*zold,*xnew,*ynew,*znew,*m,*kx,*ky,*kz,*bx,*by,*bz,*Fx,*Fy,*Fz;

//----------------------------------INITIALIZATION START----------------------------------
// Solution vectors 
xold=(float *)malloc(xlength*sizeof(float));
yold=(float *)malloc(ylength*sizeof(float));
zold=(float *)malloc(zlength*sizeof(float));
xnew=(float *)malloc(xlength*sizeof(float));
ynew=(float *)malloc(ylength*sizeof(float));
znew=(float *)malloc(zlength*sizeof(float));

// Mass vector
m=(float *)malloc(masslength*sizeof(float));

// Stiffness vectors
kx=(float *)malloc(kxlength*sizeof(float));
ky=(float *)malloc(kylength*sizeof(float));
kz=(float *)malloc(kzlength*sizeof(float));

// Damping vectors
bx=(float *)malloc(bxlength*sizeof(float));
by=(float *)malloc(bylength*sizeof(float));
bz=(float *)malloc(bzlength*sizeof(float));

// Force vectors
Fx=(float *)malloc(masslength*sizeof(float));
Fy=(float *)malloc(masslength*sizeof(float));
Fz=(float *)malloc(masslength*sizeof(float));

// Initial conditions
for (int i = 0; i < xlength ; i++)
{
	xold[i]=0.0f;
	xnew[i]=0.0f;
}

for (int i = 0; i < ylength ; i++)
{
	yold[i]=0.0f;
	ynew[i]=0.0f;
}

for (int i = 0; i < zlength ; i++)
{
	zold[i]=0.0f;
	znew[i]=0.0f;
}

// Mass [kg] and forces
for (int i = 0; i < masslength ; i++)
{
	m[i]=1.0f;
	Fx[i]=0.0f;
	Fy[i]=0.0f;
	Fz[i]=0.0f;
}

// Stiffness [N/m] and damping [N sec/m] in x-direction
for (int i = 0; i < kxlength ; i++)
{
	kx[i]=0.2f;
	bx[i]=0.05f;
}

// Stiffness [N/m] and damping [N sec/m] in y-direction
for (int i = 0; i < kylength ; i++)
{
	ky[i]=0.2f;
	by[i]=0.05f;
}

// Stiffness [N/m] and damping [N sec/m] in z-direction
for (int i = 0; i < kzlength ; i++)
{
	kz[i]=0.2f;
	bz[i]=0.05f;
}
//----------------------------------INITIALIZATION END--------------------------------------



//-------------------------------BOUNDARY CONDITIONS START----------------------------------
// No connections with Top wall B.C.'s
for (int i = 1; i <= Nx; i++)
	{
		for (int k = 1; k <= Nz; k++)
		{
			ky[i + Nx*Ny + (-1 + k)*Nx*(1 + Ny) - 1] = 0.0f;
			by[i + Nx*Ny + (-1 + k)*Nx*(1 + Ny) - 1] = 0.0f;
		}
	}
//--------------------------------BOUNDARY CONDITIONS END-----------------------------------



//--------------------------------------SOLVER START-----------------------------------------
clock_t t;
t=clock();

for (int n = 1; n <= Nstep-1; n++)
	{
		// Excitation
		Fx[(2 - 1)*Ny*Nx + (6 - 1)*Nx + 8 - 1] = sin(3 * n*dtcpu); // omega = 3 [rad/sec]
		Fy[(2 - 1)*Ny*Nx + (6 - 1)*Nx + 8 - 1] = sin(3 * n*dtcpu);
		Fz[(2 - 1)*Ny*Nx + (6 - 1)*Nx + 8 - 1] = sin(3 * n*dtcpu);

		Fx[(2 - 1)*Ny*Nx + (7 - 1)*Nx + 8 - 1] = sin(3 * n*dtcpu);
		Fy[(2 - 1)*Ny*Nx + (7 - 1)*Nx + 8 - 1] = sin(3 * n*dtcpu);
		Fz[(2 - 1)*Ny*Nx + (7 - 1)*Nx + 8 - 1] = sin(3 * n*dtcpu);

		Fx[(2 - 1)*Ny*Nx + (5 - 1)*Nx + 8 - 1] = sin(3 * n*dtcpu);
		Fy[(2 - 1)*Ny*Nx + (5 - 1)*Nx + 8 - 1] = sin(3 * n*dtcpu);
		Fz[(2 - 1)*Ny*Nx + (5 - 1)*Nx + 8 - 1] = sin(3 * n*dtcpu);

		Solve(xold,yold,zold,xnew,ynew,znew,m,kx,ky,kz,bx,by,bz,Fx,Fy,Fz);
		cudaThreadSynchronize();
		// OLD=NEW
		for (int ix = 0; ix < xlength; ix++)
		{
			xold[ix] = xnew[ix];
		}
		for (int iy = 0; iy < ylength; iy++)
		{
			yold[iy] = ynew[iy];
		}
		for (int iz = 0; iz < zlength; iz++)
		{
			zold[iz] = znew[iz];
		}
	}
ofstream fout("test.txt");
if (fout.is_open())
	{
		//file opened successfully so we are here
		cout << "File Opened successfully!!!. Writing data from array to file" << endl;

		for (int j = 0; j < zlength; j++)
			{
				fout << znew[j] << ' '; //writing ith character of array in the file
			}
			fout << '\n';
		cout << "Array data successfully saved into the file test.txt" << endl;
	}
	else //file could not be opened
	{
		cout << "File could not be opened." << endl;
	}

t=clock()-t;
printf("%f seconds\n",((float)t)/CLOCKS_PER_SEC);
printf("%f,%f,%f\n",xold[60],yold[60],zold[60]);

free(xold);
free(yold);
free(zold);
free(xnew);
free(ynew);
free(znew);
free(m);
free(kx);
free(ky);
free(kz);
free(bx);
free(by);
free(bz);
free(Fx);
free(Fy);
free(Fz);

return 0;
}


void checkCUDAError(const char *msg)
{
cudaError_t err = cudaGetLastError();
if(cudaSuccess!= err)
{
fprintf(stderr,"Cuda error: %s: %s.\n",msg,cudaGetErrorString(err));
exit(EXIT_FAILURE);
}
}