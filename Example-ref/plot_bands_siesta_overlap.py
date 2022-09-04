from numpy import *
import matplotlib.pyplot as plt
from matplotlib import colors
import glob
plt.rcParams["figure.figsize"] = (6/2.54,6/2.54)
plt.rcParams["font.size"] = 8

def main():
    fn = glob.glob('./siesta/*.bands')[0]
    kpath, enk, xt, labels = read_bands(fn)
    labels = [r'$\Gamma$','K','M',r'$\Gamma$']
    ifermi = 46
    vbm = max(enk[ifermi-1,:])
    overlap = loadtxt('overlap_matrix._ibndmin.037_ibndmax.056.txt',dtype=complex64)
    print(overlap.shape)
    cmap = plt.get_cmap('YlOrRd')
    divnorm=colors.TwoSlopeNorm(vmin=0.9,vcenter=0.95, vmax=1.0)

    for ibnd, ek in enumerate(enk) :
        plt.plot(kpath, ek-vbm, c='grey', lw=0.5, alpha=0.5)

    for ibnd, ek in enumerate(enk) :
        if ibnd in list(range(37-1,57-1)):
            overlap_ibnd = overlap[:,ibnd-36]
            overlap_sq = array([abs(z)**2 for z in overlap_ibnd])
            plt.scatter(kpath, ek-vbm, c=overlap_sq,cmap=cmap,s=0.25,norm=divnorm)


    ### Plot styling
    gridlinespec = { 'color': 'grey',
                     'linestyle': 'dotted',
                     'linewidth': 0.5 }
    plt.grid(True, axis='x', **gridlinespec)
    plt.axhline(0,  **gridlinespec)
    plt.ylabel('Energy (eV)')
    plt.xticks(xt, labels)
    plt.xlim(xt[0],xt[-1])
    plt.ylim(-2,3)
    plt.colorbar()
    plt.title(r'$|\left\langle u_{n,\mathbf{k}}^{SIE}|u_{n,\mathbf{k}}^{QE}\right\rangle|^2$')
    plt.savefig('bands.png', dpi=400, bbox_inches='tight', pad_inches=0.01)

    return

def read_bands(fname, ef=None):
    Bohr_Ang = 0.529177
    Ry_eV = 13.60580
    f = open(fname, "r")
    if ef:
        f.readline()
    else:
        ef = float64(f.readline())

    kmin, kmax = float64(f.readline().split())/Bohr_Ang # kpath min, max

    emin, emax = float64(f.readline().split()) # energy min, max

    nband, nspin, nkpts = int32(f.readline().split())

    nlines = int((nband-1)/10)+1 # .EIG dumps 10 eigenvalues per one line

    kpath = zeros(nkpts, dtype=float64)
    enk_path = zeros((nband, nkpts), dtype=float64)
    for ik in range(nkpts):
        data = []
        for il in range(nlines):
            data += f.readline().split()
        if len(data) != nband+1:
            print ("[Error] Insufficient number of eigenvalues for ik=%d"%(ik+1))
            return
        kpath[ik] = data[0]
        enk_path[:,ik] = array(data[1:], dtype=float64)-ef

    nt = int(f.readline())
    xt = []
    labels = []
    for i in range(nt):
        row = f.readline().split()
        xt.append(float(row[0]))
        labels.append(row[1].strip("'"))
    f.close()
    return kpath, enk_path, xt, labels

main()
