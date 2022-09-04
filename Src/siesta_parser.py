import numpy as np
import os


class SiestaParser:
    """Read a WFSX file in AScii format and store it in a instance.
    From WFSX
    -----------read before iteration-------------------------
    1. nk (int32, scalar): num. of k-points
    2. nspinblock (int32, scalar): we don't use it bcz we don't do spin-pol calc
    3. noncolin (bool): True if it is non-collinear calc.
    4. nou (int32, scalar): num. of total 'spatial' orbital (basis functions)

    -----------read at ik=0, ibnd_in_wfsx=0 iteration-----
    5. nwf (int32, scalar): num. of wavefunctions in each k-point
            normally they are all independent of ik
    6. io2ia (int32, nou x 1): global orbital id to atom id
    7. io2ioa: (int32, nou x 1): global orbital id to local orbital id
        Note! for io2XXX we use python convention (starting from 0)
        for indexing because it is for python indexing conversion
    8. io2species: (string, nou x 1): global orbital id to species id
    9. io2orbtype (string, nou x1): global orbital id to orb-type


    -----------read at ik=0 iteration---------------------
    10. ibnd_lst_wfsx (int32, nwf x 1):
           list of band indices contained in the WFSX file.
           this indices are all with respect the lowest occupied.
           Assume they are independent of ik

    -----------read from all---------------------------------
    11. kpoints (float64, nk x 3): k-points in Bohr^{-1}
    12. enk (float64, nk x nwf): eigenvalues in eV
    13. coeff (cdouble, nk x nwf x nou x 2(spin)):
             wavefunction coefficients w.r.t 'global orbital' index
             for collinear case, coeff[:,:,:,1] = zero
             for non-collinear case, coeff[:,:,:,1] = non-zero
    """
    def __init__(self, wfsx_file_name, fdfname=None,
                 lowest_band=None, highest_band=None):
    #    print(f'##################SIESTA-PARSER###################')
    #    print(f'W. Kim, 2022/09/03')

        self.fdf_fn = fdfname
        self.wfsx_fn = wfsx_file_name
        if not os.path.exists(self.wfsx_fn):
            self.error('[Error] .WFSX AScii file not found')
            return None
        self.read_wfsx()

        return None

    def read_wfsx(self):
        """Parse a WFSX file and save the information as attributes."""

    #    print(f'Start reading the {self.wfsx_fn}')
        f = open(self.wfsx_fn, 'r')
        f.readline()
        self.nk = np.int32(f.readline().split()[-1])
        line = f.readline() #Nr of Spins blocks
        if 'non-collinear' in line:
            self.noncolin = True
            self.nspinblock = np.int32(line.split()[-1])
        else:
            self.noncolin = False
            self.nspinblock = np.int32(line.split()[-1])

        line = f.readline()
        self.nou = np.int32(line.split()[-1])
        #Now we know the shapes of following quantities
        self.kpoints = np.zeros((self.nk,3), dtype=np.float64)
        self.io2ia = np.zeros(self.nou, dtype=np.int32)
        self.io2ioa = np.zeros(self.nou, dtype=np.int32)
        self.io2species = [] # list of len nou
        self.io2orbtype = [] # list of len nou

        f.readline()

        for ik in range(self.nk): # new ik iter. Note ik start from 0
        #    print('readling ik:',ik)
            f.readline()
            words = f.readline().split()
            kpoint = np.float64(words[3:6]) # in Bohr^{-1}
            nwf = np.int32(words[-1])
            self.kpoints[ik,:] = kpoint

            if ik == 0: #we now know the shapes of following quatities
                self.nwf = nwf
                self.ibnd_lst_wfsx = np.zeros(self.nwf, dtype=np.int32)
                self.enk = np.zeros((self.nk, self.nwf), dtype=np.float64)
                self.coeff = np.zeros((self.nk, self.nwf, self.nou, 2), dtype=np.cdouble)

            for ibnd_in_wfsx in range(nwf): # new ibnd iter. in ik
                words =  f.readline().split()
                self.enk[ik, ibnd_in_wfsx] = np.float64(words[-1])
                if ik == 0:
                    self.ibnd_lst_wfsx[ibnd_in_wfsx] = np.int32(words[2])

                f.readline() #--------------
                f.readline() # Atom  Species Orb-global ...

                for io in range(self.nou): # new io iter. in ik and ibnd
                    words = f.readline().split()
                    self.coeff[ik,ibnd_in_wfsx,io,0] = np.cdouble(np.float64(words[5])+1j*np.float64(words[6]))
                    if self.noncolin:
                        self.coeff[ik,ibnd_in_wfsx,io,1] = np.cdouble(np.float64(words[7])+1j*np.float64(words[8]))
                    if ik == 0:
                        # We are all in the python indexing convention
                        self.io2ia[io]  = np.int32(words[0])-1
                        self.io2ioa[io] = np.int32(words[3])-1
                        self.io2species.append(words[1])
                        self.io2orbtype.append(words[4])
                f.readline() #----------------
                #End of io iter.
            #End of ibnd iter.
        #End of ik iter.
        f.close()
        #print(f'End reading')
        return None
