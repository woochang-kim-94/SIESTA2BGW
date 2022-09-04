from numpy import *
import matplotlib.pyplot as plt
import glob
plt.rcParams["figure.figsize"] = (3.8/2.54,4.3/2.54)
plt.rcParams["font.size"] = 8

def main():
    fn = glob.glob('*.bands')[0]
    kpath, enk, xt, labels = read_bands(fn)
    ifermi = 46
    vbm = max(enk[ifermi-1,:])

    for ek in enk:
        plt.plot(kpath, ek-vbm, c='crimson',lw=0.5)

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
