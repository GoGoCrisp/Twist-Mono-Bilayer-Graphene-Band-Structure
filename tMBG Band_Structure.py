# Author: ZIJIE ZHENG
# Thanks to previous work you can find at 'https://github.com/zihaophys/twisted_bilayer_graphene'

from tqdm import tqdm
from numpy import *
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import os

#define constant
start_time = time.time()
degree_theta  = 1.25    # degree
const  = 2.35
omega  = 117            # twist tunneling energy
d      = 1.42           # whatever is ok.
hv     = 1.5*d*2970     # y_0,meV*angstrom, Fermi velocity for SLG 2610meV
t1     = 361
t3     = 283
t4     = 138
Dis    = 0              # v/nm
ep_bg  = 4              
N      = 5              # truncate range
valley = +1             # +1 for K, -1 for K'
KDens  = 100            # density of k points for Band Strcture, 100 is good.

#tune parameters
theta  = degree_theta/180.0*np.pi 
I      = complex(0, 1)
ei120  = cos(2*pi/3) + valley*I*sin(2*pi/3)
ei240  = cos(2*pi/3) - valley*I*sin(2*pi/3)

b1m    = 8*np.pi*sin(theta/2)/3/d*np.array([0.5, -np.sqrt(3)/2])    
b2m    = 8*np.pi*sin(theta/2)/3/d*np.array([0.5, np.sqrt(3)/2])
qb     = 8*np.pi*sin(theta/2)/3/sqrt(3)/d*array([0, -1])
K1     = 8*np.pi*sin(theta/2)/3/sqrt(3)/d*array([-sqrt(3)/2,-0.5])
K2     = 8*np.pi*sin(theta/2)/3/sqrt(3)/d*array([-sqrt(3)/2,0.5])

Tqb    = omega*np.array([[1*0.7,1], [1,1*0.7]], dtype=complex) #reduced the diagonal elements of Tj to wAA =0.7wAB = 82 meV to account[2] for corrugation and strain effects.
Tqtr   = omega*np.array([[ei120*0.7, 1], [ei240, ei120*0.7]], dtype=complex) #reduced the diagonal elements of Tj to wAA =0.7wAB = 82 meV to account[2] for corrugation and strain effects.
Tqtl   = omega*np.array([[ei240*0.7, 1], [ei120, ei240*0.7]], dtype=complex) #reduced the diagonal elements of Tj to wAA =0.7wAB = 82 meV to account[2] for corrugation and strain effects.
TqbD   = np.array(np.matrix(Tqb).H)
TqtrD  = np.array(np.matrix(Tqtr).H)
TqtlD  = np.array(np.matrix(Tqtl).H)

#define Lattice
L = []
invL = np.zeros((2*N+1, 2*N+1), int)
def Lattice(n):
    count = 0
    for i in np.arange(-n, n+1):
        for j in np.arange(-n, n+1):
            L.append([i, j])
            invL[i+n, j+n] = count
            count = count + 1
    for i in np.arange(-n, n+1):
        for j in np.arange(-n, n+1):
            L.append([i, j])
    for i in np.arange(-n, n+1):
        for j in np.arange(-n, n+1):
            L.append([i, j])

Lattice(N)
siteN = (2*N+1)*(2*N+1)
L = np.array(L)

def Hamiltonian(kx, ky,D=Dis):
    H = array(zeros((6*siteN, 6*siteN)), dtype=complex)
    e_alpha= -D*3.3/ep_bg*100 # = e * E * d / ep_bg into meV
    e_beta = -e_alpha       

    for i in np.arange(siteN):
        #diagonal term
        ix = L[i, 0]
        iy = L[i, 1]
        ax = kx - valley*K1[0] + ix*b1m[0] + iy*b2m[0] 
        ay = ky - valley*K1[1] + ix*b1m[1] + iy*b2m[1] 

        qx = cos(theta/2) * ax + sin(theta/2) * ay 
        qy =-sin(theta/2) * ax + cos(theta/2) * ay
        H[2*i,2*i]  = e_alpha
        H[2*i+1,2*i+1] = e_alpha
        H[2*i, 2*i+1] = hv * (valley*qx - I*qy) 
        H[2*i+1, 2*i] = hv * (valley*qx + I*qy)    
        #off-diagonal term
        j = i + siteN
        H[2*j, 2*i]     = TqbD[0, 0]
        H[2*j, 2*i+1]   = TqbD[0, 1]
        H[2*j+1, 2*i]   = TqbD[1, 0]
        H[2*j+1, 2*i+1] = TqbD[1, 1]   
        if (iy != valley*N):
            j = invL[ix+N, iy+valley*1+N] + siteN
            H[2*j, 2*i]     = TqtrD[0, 0]
            H[2*j, 2*i+1]   = TqtrD[0, 1]
            H[2*j+1, 2*i]   = TqtrD[1, 0]
            H[2*j+1, 2*i+1] = TqtrD[1, 1]     
        if (ix != -valley*N):
            j = invL[ix-valley*1+N, iy+N] + siteN
            H[2*j, 2*i]     = TqtlD[0, 0]
            H[2*j, 2*i+1]   = TqtlD[0, 1]
            H[2*j+1, 2*i]   = TqtlD[1, 0]
            H[2*j+1, 2*i+1] = TqtlD[1, 1]

    for i in np.arange(siteN, 2*siteN):
        #diagnoal term
        j = i - siteN
        ix = L[j, 0]
        iy = L[j, 1]
        ax = kx  - valley*K2[0] + ix*b1m[0] + iy*b2m[0] 
        ay = ky  - valley*K2[1] + ix*b1m[1] + iy*b2m[1]

        qx = cos(theta/2) * ax - sin(theta/2) * ay
        qy = sin(theta/2) * ax + cos(theta/2) * ay

        H[2*i,2*i]  =    0
        H[2*i+1,2*i+1] = 0
        H[2*i, 2*i+1] = hv * (valley*qx - I*qy)
        H[2*i+1, 2*i] = hv * (valley*qx + I*qy)
        j = i + siteN
        fk  = -(valley*qx - I*qy)
        fkd = -(valley*qx + I*qy)
        H[2*j, 2*i]     = t4*fkd
        H[2*j, 2*i+1]   = t1
        H[2*j+1, 2*i]   = t3*fk
        H[2*j+1, 2*i+1] = t4*fkd
        j = i - siteN
        H[2*j, 2*i]     = Tqb[0, 0]
        H[2*j, 2*i+1]   = Tqb[0, 1]
        H[2*j+1, 2*i]   = Tqb[1, 0]
        H[2*j+1, 2*i+1] = Tqb[1, 1]
        
        if (iy != (-valley*N)):
            j = invL[ix+N, iy-valley*1+N]
            H[2*j, 2*i]     = Tqtr[0, 0]
            H[2*j, 2*i+1]   = Tqtr[0, 1]
            H[2*j+1, 2*i]   = Tqtr[1, 0]
            H[2*j+1, 2*i+1] = Tqtr[1, 1]
        if (ix != valley*N):
            j = invL[ix+valley*1+N, iy+N]
            H[2*j, 2*i]     = Tqtl[0, 0]
            H[2*j, 2*i+1]   = Tqtl[0, 1]
            H[2*j+1, 2*i]   = Tqtl[1, 0]
            H[2*j+1, 2*i+1] = Tqtl[1, 1]
    for i in np.arange(2*siteN, 3*siteN):
         #diagnoal term
        j = i - siteN
        ix = L[j, 0]
        iy = L[j, 1]
        ax = kx  - valley*K2[0] + ix*b1m[0] + iy*b2m[0] 
        ay = ky  - valley*K2[1] + ix*b1m[1] + iy*b2m[1]

        qx = cos(theta/2) * ax - sin(theta/2) * ay
        qy = sin(theta/2) * ax + cos(theta/2) * ay

        H[2*i,2*i]  = e_beta+0
        H[2*i+1,2*i+1] = e_beta
        H[2*i, 2*i+1] = hv * (valley*qx - I*qy)
        H[2*i+1, 2*i] = hv * (valley*qx + I*qy)
        #off-diagonal term T_Bernal
        fk  = -(valley*qx - I*qy)
        fkd = -(valley*qx + I*qy)
        H[2*j, 2*i]     = t4*fk
        H[2*j, 2*i+1]   = t3*fkd
        H[2*j+1, 2*i]   = t1
        H[2*j+1, 2*i+1] = t4*fk

    eigenvalue,featurevector=np.linalg.eig(H)
    eig_vals_sorted = np.sort(eigenvalue)
    #eig_vecs_sorted = featurevector[:,eigenvalue.argsort()]
    e=eig_vals_sorted
    return e


def band_strcture():
    kD = -qb[1]
    KptoK = np.arange(1/2, -1/2, -1/KDens)
    KtoG = np.arange(-1/2, 0, 1/2/KDens)
    GtoG  = np.arange(0, sqrt(3), sqrt(3)/2/KDens)
    GtoKp  = np.arange(0,1, 1/KDens)

    AllK  = len(KptoK) + len(KtoG) + len(GtoG) + len(GtoKp)
    E  = np.zeros((AllK,6*siteN), float)


    for i in tqdm(range(0,len(KptoK))):
        k = KptoK[i]
        E[i] = np.real(Hamiltonian(sqrt(3)/2*kD,k*kD))
    for i in tqdm(range(len(KptoK), len(KtoG)+len(KptoK))):
        k = KtoG[i-len(KptoK)]
        E[i] = np.real(Hamiltonian(-sqrt(3)*k*kD, k*kD))
    for i in tqdm(range(len(KptoK)+len(KtoG), len(KptoK)+len(KtoG)+len(GtoG))):
        k = GtoG[i-len(KptoK)-len(KtoG)]
        E[i] = np.real(Hamiltonian(-1/2.0*k*kD, -sqrt(3)/2*k*kD))
    for i in tqdm(range(len(KptoK)+len(KtoG)+len(GtoG), AllK)):
        k = GtoKp[i-len(KptoK)-len(KtoG)-len(GtoG)]
        E[i] = np.real(Hamiltonian((-sqrt(3)/2 + sqrt(3)*k/2)*kD, (-3/2+k/2)*kD) )

    print("Progarm takes --- %s seconds ---" % (time.time() - start_time))


    #plot output
    for j in range(0,6*siteN):
        plt.plot(np.arange(AllK), E[:,j], linestyle="-", linewidth=2) 
    save_dir = ""   # your save_path
    plt.rcParams['savefig.directory'] = os.path.expanduser(save_dir)
    plt.title("Moir$\\'{e}$ bands of tMBG at " + str(degree_theta) + "degree under " + str(Dis) + " Display" + datetime.now().strftime("%Y-%m-%d %H-%M-%S"),  fontsize=10)
    plt.xlim(0, AllK)
    plt.ylim(-60, 80)
    plt.xticks([0, len(KptoK), len(KtoG)+len(KptoK), len(KptoK)+len(KtoG)+len(GtoG), AllK], ("K$^‘$", 'K', '$\Gamma$', '$\Gamma$', "K$^‘$"), fontsize=10)
    plt.yticks(fontsize=13)
    plt.ylabel('E(meV)', fontsize=20)
    plt.tight_layout()
    plt.show()
    return


def main():
    band_strcture()  


if __name__ == "__main__":
    main()
