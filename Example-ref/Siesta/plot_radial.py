from matplotlib import pyplot as plt
import numpy as np
import sisl
import os



def main():
    ################################################
    # Start
    ################################################
    print(f'Plot radial functions of SIESTA')
    print(f'Woochang Kim, 2022/08/30')
    print(f'Parameters')

    ################################################
    # run parameters
    ################################################
    fdf_file = f'./input.fdf'
    unfold = Unfold(fdfname=fdf_file)
    R_set = unfold.give_radial()
    print(len(R_set))

    plt.axhline(0,c='k',lw=0.5)
    plt.plot(R_set[0][0], R_set[0][1], c='tab:blue',label=r'$R_{n=5,l=0}$')
    plt.plot(R_set[1][0], R_set[1][1], c='tab:orange',label=r'$R_{n=6,l=0}$')
    plt.plot(R_set[3][0], R_set[3][1], c='tab:purple',label=r'$R_{n=5,l=1}$')
    plt.title('W, Radial functions' )
    plt.legend(frameon=False)
    plt.xlabel(r'$\AA$')
    plt.savefig('radialfunc.png', dpi=400)


    return None

class Unfold:
    def __init__(self, fdfname,
                 lowest_band=None, highest_band=None):
        self.fdf = sisl.io.siesta.fdfSileSiesta(fdfname)
        self.geom = self.fdf.read_geometry()
        self.cell = self.geom.cell
#        print(geometry)
        print(self.cell) # in Ang unit
        self.bcell = 2*np.pi*np.linalg.inv(self.cell).T # in Ang^-1 unit
        print(self.bcell) #
        self.fdf_path = self.fdf.dir_file()
        self.lowest_band = lowest_band
        self.highest_band = highest_band
        self.outdir = self.fdf_path.parents[0]
        self.system_label = self.fdf.get('SystemLabel')
        self.ef = self.fdf.read_fermi_level()
        self._siesta = sisl.io.siesta._siesta

        self.wfsx_file = f'{self.outdir}/{self.system_label}.bands.WFSX'
        if not os.path.exists(self.wfsx_file):
            self.error('[Error] .WFSX file not found')
            return

        # First query information
        self.nspin, self.nou, self.nk, self.Gamma = self._siesta.read_wfsx_sizes(
            self.wfsx_file)
        print(self.nspin)
        print('nk',self.nk)

        if self.nspin in [4, 8]:
            self.nspin = 1  # only 1 spin
            self._read_wfsx = self._siesta.read_wfsx_index_4
        elif self.Gamma:
            self._read_wfsx = self._siesta.read_wfsx_index_1
        else:
            self._read_wfsx = self._siesta.read_wfsx_index_2

        self.basis = self.fdf.read_basis()
        print(self.basis)

        return None



    def give_radial(self,  nr=500):
        """
        ------
        input
        ------
        ------
        output
        ------
        selected R(r)
        [[rgrid, R(r) ], ....]
        len will be len(io)
        """
        import scipy.special as sp
        from scipy.integrate import simps

        R_set = []

        for isp, specie in enumerate(self.basis):

            orbs = []
            for io, orb in enumerate(specie.orbitals):
                m = orb.m
                l = orb.l

                rgrid = np.linspace(0, orb.R, nr) # length unit is Ang
                phi_r = orb.radial(rgrid, is_radius=True)
                pkg = [rgrid, phi_r]

                R_set.append(pkg)

        self.R_set = R_set

        return R_set


main()
