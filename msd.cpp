#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream> 
#include <math.h>  
using namespace std;


//-----------------------------INPUT----------------------------//
const int Nx = 15, Ny = 6, Nz = 3; // Number of mass points

float maxtime = 60; // End time [sec]
const int Nstep = 1200; // Number of time steps
float dt = maxtime / Nstep; // Time step size [sec]

// Array lengths
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

//Solution matrices
typedef vector<vector<float> > Matrix;
vector<float> x(xlength, 0.0f);
vector<float> y(ylength, 0.0f);
vector<float> z(zlength, 0.0f);
Matrix xmatrix(2, x);
Matrix ymatrix(2, y);
Matrix zmatrix(2, z);

// Mass vector
vector<float> m(masslength, 1.0f);

// Stiffness vectors
vector<float> kx(kxlength, 0.2f);
vector<float> ky(kylength, 0.2f);
vector<float> kz(kzlength, 0.2f);

// Damping vectors
vector<float> bx(bxlength, 0.05f);
vector<float> by(bylength, 0.05f);
vector<float> bz(bzlength, 0.05f);

// Force matrices
vector<float> Fx(masslength, 0.0f);
vector<float> Fy(masslength, 0.0f);
vector<float> Fz(masslength, 0.0f);
Matrix Fxmatrix(Nstep, Fx);
Matrix Fymatrix(Nstep, Fy);
Matrix Fzmatrix(Nstep, Fz);


//-----------------------------FUNCTIONS----------------------------//
//x-displacement
float fxx(int n, int i, int j, int k)
{
	return xmatrix[n-1][ (Ny + 2)*(4 + 2*Nx) + (k - 1)*(Ny + 2)*(4 + 2*Nx) + 4 +
		2*Nx + (i - 1)*(4 + 2*Nx) + 2 + (2*j - 1)-1];
}
//x-velocity
float fvx(int n, int i, int j, int k)
{
	return xmatrix[n-1][ (Ny + 2)*(4 + 2*Nx) + (k - 1)*(Ny + 2)*(4 + 2*Nx) + 4 +
		2*Nx + (i - 1)*(4 + 2*Nx) + 2 + (2*j)-1];
}
//y-displacement
float fyy(int n, int i, int j, int k)
{
	return ymatrix[n-1][ (Nx + 2)*(4 + 2*Ny) + (k - 1)*(Nx + 2)*(4 + 2*Ny) + 4 +
		2*Ny + (j - 1)*(4 + 2*Ny) + 2 + (2*i - 1)-1];
}
//y-velocity
float fvy(int n, int i, int j, int k)
{
	return ymatrix[n-1][ (Nx + 2)*(4 + 2 * Ny) + (k - 1)*(Nx + 2)*(4 + 2 * Ny) + 4 +
		2 * Ny + (j - 1)*(4 + 2 * Ny) + 2 + (2 * i)-1];
}
//z-displacement
float fzz(int n, int i, int j, int k)
{
	return zmatrix[n-1][ (Nx + 2)*(4 + 2*Nz) + (i - 1)*(Nx + 2)*(4 + 2*Nz) + 4 +
		2*Nz + (j - 1)*(4 + 2*Nz) + 2 + (2*k - 1)-1];
}
//z-velocity
float fvz(int n, int i, int j, int k)
{
	return zmatrix[n-1][ (Nx + 2)*(4 + 2 * Nz) + (i - 1)*(Nx + 2)*(4 + 2 * Nz) + 4 +
		2 * Nz + (j - 1)*(4 + 2 * Nz) + 2 + (2 * k)-1];
}
//mass
float fm(int i, int j, int k)
{
	return m[(k - 1)*Ny*Nx + (i - 1)*Nx + j-1];
}
//x-stiffness
float fkx(int i, int j, int k)
{
	return kx[(k - 1)*Ny*(Nx + 1) + (i - 1)*(Nx + 1) + j-1];
}
//y-stiffness
float fky(int i, int j, int k)
{
	return ky[(k - 1)*Nx*(Ny + 1) + (i - 1)*Nx + j-1];
}
//z-stiffness
float fkz(int i, int j, int k)
{
	return kz[(k - 1)*Nx*Ny + (i - 1)*Nx + j-1];
}
//x-damping
float fbx(int i, int j, int k)
{
	return bx[(k - 1)*Ny*(Nx + 1) + (i - 1)*(Nx + 1) + j-1];
}
//y-damping
float fby(int i, int j, int k)
{
	return by[(k - 1)*Nx*(Ny + 1) + (i - 1)*Nx + j-1];
}
//z-damping
float fbz(int i, int j, int k)
{
	return bz[(k - 1)*Nx*Ny + (i - 1)*Nx + j-1];
}
//x-force
float fFx(int i, int j, int k, int n)
{
	return Fxmatrix[n-1][(k - 1)*Ny*Nx + (i - 1)*Nx + j-1];
}
//y-force
float fFy(int i, int j, int k, int n)
{
	return Fymatrix[n-1][(k - 1)*Ny*Nx + (i - 1)*Nx + j-1];
}
//z-force
float fFz(int i, int j, int k, int n)
{
	return Fzmatrix[n-1][(k - 1)*Ny*Nx + (i - 1)*Nx + j-1];
}
//x-acceleration
float ax(int i, int j, int k, int n)
{
	return (fFx(i, j, k, n) - fby(i, j, k)*(-fvx(1, -1 + i, j, k) + fvx(1, i, j, k)) -
		fbx(i, j, k)*(-fvx(1, i, -1 + j, k) + fvx(1, i, j, k)) - fbz(i, j, k)*(-fvx(1, i, j, -1 + k) + fvx(1, i, j, k)) +
		fbz(i, j, 1 + k)*(-fvx(1, i, j, k) + fvx(1, i, j, 1 + k)) +
		fbx(i, 1 + j, k)*(-fvx(1, i, j, k) + fvx(1, i, 1 + j, k)) +
		fby(1 + i, j, k)*(-fvx(1, i, j, k) + fvx(1, 1 + i, j, k)) -
		fky(i, j, k)*(-fxx(1, -1 + i, j, k) + fxx(1, i, j, k)) - fkx(i, j, k)*(-fxx(1, i, -1 + j, k) + fxx(1, i, j, k)) -
		fkz(i, j, k)*(-fxx(1, i, j, -1 + k) + fxx(1, i, j, k)) +
		fkz(i, j, 1 + k)*(-fxx(1, i, j, k) + fxx(1, i, j, 1 + k)) +
		fkx(i, 1 + j, k)*(-fxx(1, i, j, k) + fxx(1, i, 1 + j, k)) +
		fky(1 + i, j, k)*(-fxx(1, i, j, k) + fxx(1, 1 + i, j, k))) / fm(i, j, k);
}
//y-acceleration
float ay(int i, int j, int k, int n)
{
	return (fFy(i, j, k, n) - fby(i, j, k)*(-fvy(1, -1 + i, j, k) + fvy(1, i, j, k)) -
		fbx(i, j, k)*(-fvy(1, i, -1 + j, k) + fvy(1, i, j, k)) - fbz(i, j, k)*(-fvy(1, i, j, -1 + k) + fvy(1, i, j, k)) +
		fbz(i, j, 1 + k)*(-fvy(1, i, j, k) + fvy(1, i, j, 1 + k)) +
		fbx(i, 1 + j, k)*(-fvy(1, i, j, k) + fvy(1, i, 1 + j, k)) +
		fby(1 + i, j, k)*(-fvy(1, i, j, k) + fvy(1, 1 + i, j, k)) -
		fky(i, j, k)*(-fyy(1, -1 + i, j, k) + fyy(1, i, j, k)) - fkx(i, j, k)*(-fyy(1, i, -1 + j, k) + fyy(1, i, j, k)) -
		fkz(i, j, k)*(-fyy(1, i, j, -1 + k) + fyy(1, i, j, k)) +
		fkz(i, j, 1 + k)*(-fyy(1, i, j, k) + fyy(1, i, j, 1 + k)) +
		fkx(i, 1 + j, k)*(-fyy(1, i, j, k) + fyy(1, i, 1 + j, k)) +
		fky(1 + i, j, k)*(-fyy(1, i, j, k) + fyy(1, 1 + i, j, k))) / fm(i, j, k);
}
//z-acceleration
float az(int i, int j, int k, int n)
{
	return (fFz(i, j, k, n) - fby(i, j, k)*(-fvz(1, -1 + i, j, k) + fvz(1, i, j, k)) -
		fbx(i, j, k)*(-fvz(1, i, -1 + j, k) + fvz(1, i, j, k)) - fbz(i, j, k)*(-fvz(1, i, j, -1 + k) + fvz(1, i, j, k)) +
		fbz(i, j, 1 + k)*(-fvz(1, i, j, k) + fvz(1, i, j, 1 + k)) +
		fbx(i, 1 + j, k)*(-fvz(1, i, j, k) + fvz(1, i, 1 + j, k)) +
		fby(1 + i, j, k)*(-fvz(1, i, j, k) + fvz(1, 1 + i, j, k)) -
		fky(i, j, k)*(-fzz(1, -1 + i, j, k) + fzz(1, i, j, k)) - fkx(i, j, k)*(-fzz(1, i, -1 + j, k) + fzz(1, i, j, k)) -
		fkz(i, j, k)*(-fzz(1, i, j, -1 + k) + fzz(1, i, j, k)) +
		fkz(i, j, 1 + k)*(-fzz(1, i, j, k) + fzz(1, i, j, 1 + k)) +
		fkx(i, 1 + j, k)*(-fzz(1, i, j, k) + fzz(1, i, 1 + j, k)) +
		fky(1 + i, j, k)*(-fzz(1, i, j, k) + fzz(1, 1 + i, j, k))) / fm(i, j, k);
}



int main(int argc, char *argv[])
{
	ofstream fout("test.txt");
	
	// No connections with Top wall B.C.'s
	for (int i = 1; i <= Nx; i++)
	{
		for (int k = 1; k <= Nz; k++)
		{
			ky[i + Nx*Ny + (-1 + k)*Nx*(1 + Ny) - 1] = 0.0f;
			by[i + Nx*Ny + (-1 + k)*Nx*(1 + Ny) - 1] = 0.0f;
		}
	}

	// Excitation
	for (int i = 1; i <= Nstep; i++)
	{
		Fxmatrix[i - 1][(2 - 1)*Ny*Nx + (6 - 1)*Nx + 8 - 1] = sin(3 * i*dt); // omega = 3 [rad/sec]
		Fymatrix[i - 1][(2 - 1)*Ny*Nx + (6 - 1)*Nx + 8 - 1] = sin(3 * i*dt);
		Fzmatrix[i - 1][(2 - 1)*Ny*Nx + (6 - 1)*Nx + 8 - 1] = sin(3 * i*dt);

		Fxmatrix[i - 1][(2 - 1)*Ny*Nx + (7 - 1)*Nx + 8 - 1] = sin(3 * i*dt); 
		Fymatrix[i - 1][(2 - 1)*Ny*Nx + (7 - 1)*Nx + 8 - 1] = sin(3 * i*dt);
		Fzmatrix[i - 1][(2 - 1)*Ny*Nx + (7 - 1)*Nx + 8 - 1] = sin(3 * i*dt);

		Fxmatrix[i - 1][(2 - 1)*Ny*Nx + (5 - 1)*Nx + 8 - 1] = sin(3 * i*dt);
		Fymatrix[i - 1][(2 - 1)*Ny*Nx + (5 - 1)*Nx + 8 - 1] = sin(3 * i*dt);
		Fzmatrix[i - 1][(2 - 1)*Ny*Nx + (5 - 1)*Nx + 8 - 1] = sin(3 * i*dt);
	}

	//-------------------------SOLVER--------------------------//
	
	for (int n = 1; n <= Nstep-1; n++)
	{

		printf("%d", n);
		for (int i = 1; i <= Ny; i++)
		{
			for (int j = 1; j <= Nx; j++)
			{
				for (int k = 1; k <= Nz; k++)
				{
					xmatrix[1][ (Ny + 2)*(4 + 2*Nx) + (k - 1)*(Ny + 2)*(4 + 2*Nx) + 4 +
						2 * Nx + (i - 1)*(4 + 2 * Nx) + 2 + (2 * j - 1) - 1] = fxx(1, i, j, k) + fvx(1, i, j, k)*dt;
					xmatrix[1][ (Ny + 2)*(4 + 2 * Nx) + (k - 1)*(Ny + 2)*(4 + 2 * Nx) + 4 +
						2 * Nx + (i - 1)*(4 + 2 * Nx) + 2 + (2 * j) - 1] = fvx(1, i, j, k) + ax(i, j, k, n)*dt;

					ymatrix[1][ (Nx + 2)*(4 + 2*Ny) + (k - 1)*(Nx + 2)*(4 + 2*Ny) + 4 +
						2*Ny + (j - 1)*(4 + 2*Ny) + 2 + (2*i - 1)-1] = fyy(1, i, j, k) + fvy(1, i, j, k)*dt;
					ymatrix[1][ (Nx + 2)*(4 + 2*Ny) + (k - 1)*(Nx + 2)*(4 + 2*Ny) + 4 +
						2*Ny + (j - 1)*(4 + 2*Ny) + 2 + (2*i)-1] = fvy(1, i, j, k) + ay(i, j, k, n)*dt;

					zmatrix[1][ (Nx + 2)*(4 + 2*Nz) + (i - 1)*(Nx + 2)*(4 + 2*Nz) + 4 +
						2*Nz + (j - 1)*(4 + 2*Nz) + 2 + (2*k - 1)-1] = fzz(1, i, j, k) + fvz(1, i, j, k)*dt;
					zmatrix[1][ (Nx + 2)*(4 + 2*Nz) + (i - 1)*(Nx + 2)*(4 + 2*Nz) + 4 +
						2*Nz + (j - 1)*(4 + 2*Nz) + 2 + (2*k)-1] = fvz(1, i, j, k) + az(i, j, k, n)*dt;
				}
			}
		}
		//OLD=NEW
		for (int ix = 0; ix < xlength; ix++)
		{
			xmatrix[0][ix] = xmatrix[1][ix];
		}
		for (int iy = 0; iy < ylength; iy++)
		{
			ymatrix[0][iy] = ymatrix[1][iy];
		}
		for (int iz = 0; iz < zlength; iz++)
		{
			zmatrix[0][iz] = zmatrix[1][iz];
		}
	}
	return 0;
}

