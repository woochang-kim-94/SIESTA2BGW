#import tim
#import sys
import numpy as np
from mpi4py import MPI
import os
import sisl
from siesta_parser import SiestaParser
#from datetime import datetime as dt
#import logging
#import matplotlib.pyplot as plt
#import itertools

comm = MPI.COMM_WORLD
def main():
    ################################################
    # Start
    ################################################
    if comm.rank == 0:
        print(f'##################SIESTA2BGW###################')
        print(f'Under developing.... ')
        print(f'Supports non-colinear calculation')
        print(f'W. Kim, M. Naik, 2022/08/30')
        print(f"Motivated by the Y.W.Choi's unfold.py code")
        print(f'Parameters')

    ################################################
    # run parameters
    ################################################
    fdf_file = f'./input.fdf'
    ecut = 40.0*13.605693122994  # eV; good to be matched with the ecutwfc of QE
    lowest_band  = 37             # lowest_band in WFSX file
    highest_band = 56            # highest_band in WFSX file
    #ik , ibnd = 31, 23           # both of them starting from 1
    ibnd_lst = list(range(lowest_band, highest_band+1))
    ik_lst = list(range(1,92))
    #ik_lst = [31]
    #print('ik, ibnd:', ik, ibnd)
    if comm.rank == 0:
        print('ik_lst, ibnd_lst', ik_lst, ibnd_lst)


    unfold = Unfold(fdfname=fdf_file,
            ecut=ecut,
            lowest_band=lowest_band,
            highest_band=highest_band)
    for ik in ik_lst:
        unkg_blst, miller_ids = unfold.calc_planewave(ik, ibnd_lst)

        ########################################################
        # save the wavefunction and FFT grid in numpy format
        ########################################################
        if comm.rank == 0:
            unkg_blst_fn = 'unkg_blst._ik.{0:03d}._ibndmin.{1:03d}_ibndmax.{2:03d}.txt'.format(ik,ibnd_lst[0],ibnd_lst[-1])
            miller_ids_fn = 'miller_ids.{0:03d}.txt'.format(ik)
            #print('Norm of unkg:', np.linalg.norm(unkg))
            np.savetxt(unkg_blst_fn, unkg_blst)
            np.savetxt(miller_ids_fn, miller_ids)

    return None



class Unfold:
    def __init__(self, fdfname, ecut,
                 lowest_band=None, highest_band=None):
        self.fdf = sisl.io.siesta.fdfSileSiesta(fdfname)
        self.ecut = ecut
        self.geom = self.fdf.read_geometry()
        self.cell = self.geom.cell
#        print(geometry)
        if comm.rank == 0:
            print(self.cell) # in Ang unit
        self.bcell = 2*np.pi*np.linalg.inv(self.cell).T # in Ang^-1 unit
        if comm.rank == 0:
            print(self.bcell) #
        self.fdf_path = self.fdf.dir_file()
        self.lowest_band = lowest_band
        self.highest_band = highest_band
        self.outdir = self.fdf_path.parents[0]
        self.system_label = self.fdf.get('SystemLabel')
        self.ef = self.fdf.read_fermi_level()
        self._siesta = sisl.io.siesta._siesta


        #########SiestaParser###############
        self.wfsx_txt_file = f'{self.outdir}/{self.system_label}.bands.WFSX.txt'
        if not os.path.exists(self.wfsx_txt_file):
            self.error('[Error] .WFSX.txt file not found')
            return
        self.myparser = SiestaParser(wfsx_file_name=self.wfsx_txt_file)

        self.wfsx_file = f'{self.outdir}/{self.system_label}.bands.WFSX'
        if not os.path.exists(self.wfsx_file):
            self.error('[Error] .WFSX file not found')
            return
        if comm.rank == 0 and self.myparser.noncolin:
            print('This calculation is a non-collinear calculation')

        # First query information
        self.nspin, self.nou, self.nk, self.Gamma = self._siesta.read_wfsx_sizes(
            self.wfsx_file)

        if self.nspin in [4, 8]:
            self._read_wfsx = self._siesta.read_wfsx_index_4
        elif self.Gamma:
            self._read_wfsx = self._siesta.read_wfsx_index_1
        else:
            self._read_wfsx = self._siesta.read_wfsx_index_2


        # setup band range in WFSX file
        if self.lowest_band is None:
            self.lowest_band = 1

        if self.highest_band is None:
            self.highest_band = self.nou


        self.nbnd = self.highest_band-self.lowest_band+1
        self.basis = self.fdf.read_basis()

        return None

    def calc_planewave(self, ik, ibnd_lst):
        """
        ------
        Input
        ------
        ik: k-point index. starting from 1
        ibnd_lst: list of band index. starting from lowest occupied states

        ------
        output
        ------
        unkg:{<k+G|psi_nk>}
        miller_ids: miller index of planewave basis

        unkg is an array of
        arrary(igvec) in complex
        where 3 represent px, py, pz

        """
        ibnd_lst = np.array(ibnd_lst, dtype=np.int32)
        Ang2Bohr = sisl.unit.unit_convert('Ang', 'Bohr')
        #unkg = np.zeros(self.ngvec, dtype=np.cdouble)
        nbnd = len(ibnd_lst)
        ibnd_in_wfsx_lst = ibnd_lst - self.lowest_band

        pos = self.geom.axyz()
        io2ia = np.zeros(self.nou, dtype=np.int32)  # orbital id to atom id
        io2isp = np.zeros(self.nou, dtype=np.int32) # orbital id to species id
        io2ioa = np.zeros(self.nou, dtype=np.int32) # orbital id in a given atom
        for ia, io in self.geom.iter_orbitals(local=False):
            io2ia[io] = ia
            io2ioa[io] = io-self.geom.atoms.firsto[ia]
            io2isp[io] = self.geom.atoms.specie[ia]

        # k is in Bohr^{-1} unit
        k_bohr = self.myparser.kpoints[ik-1]
        nwf    = self.myparser.nwf
        eig    = self.myparser.enk[ik-1,:]
        state_up  = self.myparser.coeff[ik-1,:,:,0]  # ik, ibnd, nou, ispin
        coeff_up  = state_up[ibnd_in_wfsx_lst[0]:ibnd_in_wfsx_lst[-1]+1,:]
        if self.myparser.noncolin:
            state_dw  = self.myparser.coeff[ik-1,:,:,1]  # ik, ibnd, nou, ispin
            coeff_dw  = state_dw[ibnd_in_wfsx_lst[0]:ibnd_in_wfsx_lst[-1]+1,:]

        k = k_bohr * Ang2Bohr # Now k is in Ang^{-1} unit

        if comm.rank == 0:
            print('Generating FFT grid')
        igvec_at_k = self.find_igvec_at_k(k, self.bcell, self.ecut)
        gvec_at_k = igvec_at_k.dot(self.bcell) # Ang^{-1}
        if comm.rank == 0:
            print('Number of FFT grid:', len(igvec_at_k))

        phase = np.zeros(
        (len(igvec_at_k), len(pos)), dtype=np.cdouble) #len(pos) = # ia

        for ig, g in enumerate(gvec_at_k):
            phase[ig, :] = np.exp(-1j*pos.dot(g))   # gauge convention 1
            #phase[ig, :] = 1.0                     # gauge convention 2

        unkg_up = np.zeros((nbnd, len(igvec_at_k)), dtype=np.cdouble)
        if self.myparser.noncolin:
            unkg_dw = np.zeros((nbnd, len(igvec_at_k)), dtype=np.cdouble)

        ngs_pp = int(len(igvec_at_k)/comm.size)
        Grange = range(len(igvec_at_k))
        ig_rank = []
        for ip in range(comm.size):
            ig_rank.append(Grange[ip:len(Grange):comm.size])

        if comm.rank == 0:
            print('starting G vector loop')
        #for ig, g in enumerate(gvec_at_k):


        if self.myparser.noncolin:
            for ig in ig_rank[comm.rank]:
                #kpg = k+g # in Ang^{-1}
                gvec = gvec_at_k[ig]
                kpg = k+gvec # in Ang^{-1}
                # First we calculate all the FT of PAO basis
                PAO_FT = self.calc_PAO_FT(kpg) # len(isp) x len(ioa)
                ftfac = np.zeros(self.nou, dtype=np.cdouble)

                for ia, io in self.geom.iter_orbitals(local=False):
                    ftfac[io] = PAO_FT[io2isp[io]][io2ioa[io]]
                orbfac = phase[ig, io2ia[:]] * ftfac # len = num(io)

                for ibnd in range(len(ibnd_in_wfsx_lst)):
                    #matel = np.einsum('i,i', orbfac, coeff[:,ibnd])
                    matel_up = np.einsum('i,i', orbfac, coeff_up[ibnd,:])
                    matel_dw = np.einsum('i,i', orbfac, coeff_dw[ibnd,:])
                    #matel /= np.sqrt(self.geom.volume)
                    unkg_up[ibnd,ig] = matel_up
                    unkg_dw[ibnd,ig] = matel_dw


            if comm.rank == 0:
                unkg_up_coll = np.zeros_like(unkg_up)
                unkg_dw_coll = np.zeros_like(unkg_dw)
            else:
                unkg_up_coll = None
                unkg_dw_coll = None
            comm.Reduce(
                     [unkg_up, MPI.COMPLEX16],
                     [unkg_up_coll, MPI.COMPLEX16],
                      op = MPI.SUM,
                      root = 0
                    )
            comm.Reduce(
                     [unkg_dw, MPI.COMPLEX16],
                     [unkg_dw_coll, MPI.COMPLEX16],
                      op = MPI.SUM,
                      root = 0
                    )
            if comm.rank == 0:
                print('end of G vector loop')

            if comm.rank == 0:
                unkg = np.concatenate((unkg_up_coll, unkg_dw_coll),axis=1)
                unkg /= np.sqrt(self.geom.volume)
                #Renormalization is not necessory but we do anyway...
                #unkg /= np.linalg.norm(unkg)
                return unkg, igvec_at_k
            else:
                return 0,0


        else:
            for ig in ig_rank[comm.rank]:
                #kpg = k+g # in Ang^{-1}
                gvec = gvec_at_k[ig]
                kpg = k+gvec # in Ang^{-1}
                # First we calculate all the FT of PAO basis
                PAO_FT = self.calc_PAO_FT(kpg) # len(isp) x len(ioa)
                ftfac = np.zeros(self.nou, dtype=np.cdouble)

                for ia, io in self.geom.iter_orbitals(local=False):
                    ftfac[io] = PAO_FT[io2isp[io]][io2ioa[io]]
                orbfac = phase[ig, io2ia[:]] * ftfac # len = num(io)

                for ibnd in range(len(ibnd_in_wfsx_lst)):
                    matel_up = np.einsum('i,i', orbfac, coeff_up[ibnd,:])
                    unkg_up[ibnd,ig] = matel_up


            if comm.rank == 0:
                unkg_up_coll = np.zeros_like(unkg_up)
            else:
                unkg_up_coll = None
            comm.Reduce(
                     [unkg_up, MPI.COMPLEX16],
                     [unkg_up_coll, MPI.COMPLEX16],
                      op = MPI.SUM,
                      root = 0
                    )
            if comm.rank == 0:
                print('end of G vector loop')

            if comm.rank == 0:
                unkg_up_coll /= np.sqrt(self.geom.volume)
                #Renormalization is not necessory but we do anyway...
                #unkg /= np.linalg.norm(unkg)

                return unkg_up_coll, igvec_at_k
            else:
                return 0,0


    def calc_PAO_FT(self, k, nr=500):
        """
        ------
        input
        ------
        k(3): k-vector (kx,ky,kz) in Ang^-1
        nr: number of grid points between 0 ~ R_cut

        ------
        output
        ------
        PAOFT integral exp(-ivec{k}*vec{r}) PAO(vec{r}) dV
        PAOFT is a list of following shape
        [[orbitals of the specie 1], [orbitals of the specie 2], ...]
        so PAOFT = PAOFT[isp, ioa]
        """
        import scipy.special as sp
        from scipy.integrate import simps

        knorm = np.linalg.norm(k)
        # gamma point
        if knorm < 1e-12:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(k[2]/knorm)
            phi = np.arctan2(k[1], k[0])

        PAOFT = []
        for isp, specie in enumerate(self.basis):
            PAOFT.append([])

            orbs = []
            for io, orb in enumerate(specie.orbitals):
                m = orb.m
                l = orb.l

                rgrid = np.linspace(0, orb.R, nr) # length unit is Ang
                phi_r = orb.radial(rgrid, is_radius=True)
                kr = knorm * rgrid
                j_l = sp.spherical_jn(l, kr)
                sbt = simps(rgrid*rgrid*j_l*phi_r, rgrid)

                FT = 4.0*np.pi*(-1)**m*(-1j)**l*sbt
                if m < 0:
                    # note that scipy uses different names for theta, phi
                    FT *= 1.j/np.sqrt(2.0)*(sp.sph_harm(m, l, phi,
                                                        theta)-(-1)**m*sp.sph_harm(-m, l, phi, theta))
                elif m == 0:
                    FT *= sp.sph_harm(0, l, phi, theta)
                else:  # m>0
                    FT *= 1.0/np.sqrt(2.0)*(sp.sph_harm(-m, l, phi,
                                                        theta)+(-1)**m*sp.sph_harm(m, l, phi, theta))

                PAOFT[isp].append(FT)

        return PAOFT


    def find_igvec(self, bcell_in, ecut_in):
        """Find G(n1,n2,n3) vectors such that {(n1,n2,n3).dot(bcell)}^2 < ecut

        Args:
            bcell_in (float64, 3x3): reciprocal lattice vectors (Ang^{-1})
            ecut_in (float64): enery cutoff in eV

        Returns:
            int32, array: list of integers (n1,n2,n3)
        """
        eV2Ry = sisl.unit.unit_convert('eV', 'Ry')
        Ang2Bohr = sisl.unit.unit_convert('Ang', 'Bohr')

        bcell = bcell_in / Ang2Bohr
        ecut = ecut_in * eV2Ry

        # find nmax such that (nmax*bcell[idir,:])^2 < ecut along each direction.
        b2 = np.sum(bcell**2, axis=1)
        nmax = np.int32(2*np.sqrt(ecut/b2))

        n1 = np.arange(-nmax[0], nmax[0]+1)
        n2 = np.arange(-nmax[1], nmax[1]+1)
        n3 = np.arange(-nmax[2], nmax[2]+1)
        # n1grid, n1grid, n3grid = np.meshgrid(n1, n2, n3)
        igvec_all = np.int32([(i, j, k) for i in n1 for j in n2 for k in n3])
        gvec = igvec_all.dot(bcell)
        g2 = np.einsum('ix,ix->i', gvec, gvec)

        igvec = igvec_all[np.where(g2 < ecut)]
        # test = igvec.dot(bcell_uc)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(test[:, 0], test[:, 1], test[:, 2], s=1, c='r')
        # c = plt.Circle((0, 0), np.sqrt(ecut), color='k', fill=False)
        # ax.add_patch(c)
        # plt.savefig('test.png')

        return igvec

    def find_igvec_at_k(self, k_in, bcell_in, ecut_in):
        """Find G(n1,n2,n3) vectors such that {((n1,n2,n3)+k_crys).dot(bcell)}^2 < ecut

        Args:
            k_in (float64, 1x3): k-point in 1st BZ zone (Ang^{-1})
            bcell_in (float64, 3x3): reciprocal lattice vectors (Ang^{-1})
            ecut_in (float64): enery cutoff in eV

        Returns:
            int32, array: list of integers (n1,n2,n3)
        """
        eV2Ry = sisl.unit.unit_convert('eV', 'Ry')
        Ang2Bohr = sisl.unit.unit_convert('Ang', 'Bohr')

        k_crys = np.dot(k_in, np.linalg.inv(bcell_in)) # dimensionless

        bcell = bcell_in / Ang2Bohr
        ecut = ecut_in * eV2Ry

        # find nmax such that (nmax*bcell[idir,:])^2 < ecut along each direction.
        b2 = np.sum(bcell**2, axis=1)
        nmax = np.int32(2*np.sqrt(ecut/b2))

        n1 = np.arange(-nmax[0], nmax[0]+1)
        n2 = np.arange(-nmax[1], nmax[1]+1)
        n3 = np.arange(-nmax[2], nmax[2]+1)
        # n1grid, n1grid, n3grid = np.meshgrid(n1, n2, n3)
        igvec_all = np.int32([(i, j, k) for i in n1 for j in n2 for k in n3])
        shifted_igvec_all = igvec_all + k_crys
        shifted_gvec = shifted_igvec_all.dot(bcell)
        shifted_g2 = np.einsum('ix,ix->i', shifted_gvec, shifted_gvec)

        igvec = igvec_all[np.where(shifted_g2 < ecut)]

        return igvec

if __name__=='__main__':
    main()
