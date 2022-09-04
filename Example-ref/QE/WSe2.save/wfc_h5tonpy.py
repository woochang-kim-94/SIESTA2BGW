#!/usr/bin/env python
import h5py
from numpy import *
import os
from os.path import exists
from scipy.spatial import KDTree
bohr = 0.529177211 #Angstrom


def main():
    ibnd_lst = list(range(37, 56+1))
    ik_lst = list(range(1,92))
    for ik in ik_lst:
        print('Working on ik:',ik)
        unkg_blst_fn = 'unkg_blst._ik.{0:03d}._ibndmin.{1:03d}_ibndmax.{2:03d}.txt'.format(ik,ibnd_lst[0],ibnd_lst[-1])
        miller_ids_fn = 'miller_ids.{0:03d}.txt'.format(ik)
        unkg_blst,  miller_ids = h5tonpy(ik, ibnd_lst)
        savetxt(unkg_blst_fn, unkg_blst)
        savetxt(miller_ids_fn, miller_ids)
    return None


def h5tonpy(ik, ibnd_lst):
    wfc_fn = 'wfc'+str(ik)+'.hdf5'.format(ik)
    unkg_set, miller_ids, npol = read_wfc(wfc_fn)

    return unkg_set[ibnd_lst[0]-1:ibnd_lst[-1],:], miller_ids



def read_wfc(fn):

    #Read wfc file
    print("Read ",fn)
    f = h5py.File(fn)

    gamma_only = f.attrs['gamma_only'] # bool
    igwx = f.attrs['igwx'] # int
    ik = f.attrs['ik'] # int k-point index
    ispin = f.attrs['ispin'] # int
    nbnd = f.attrs['nbnd'] # int
    ngw = f.attrs['ngw'] # int
    npol = f.attrs['npol'] # int
    scale_factor = f.attrs['scale_factor'] #float
    xk = f.attrs['xk'] # 3 components array in Bohr^-1?

    miller_ids = array(f['MillerIndices']) # (3 x 3)
#    savetxt('miller_ids.{0:03d}.txt'.format(ik),miller_ids)
    print("\nMiller indices are saved in a text file")
    rec_latt = zeros((3,3))
    rec_latt[0,:] = array(f['MillerIndices'].attrs['bg1'])
    rec_latt[1,:] = array(f['MillerIndices'].attrs['bg2'])
    rec_latt[2,:] = array(f['MillerIndices'].attrs['bg3'])

    #rec_latt = (2*pi/alat)*rec_latt


    evc = array(f['evc'])
    if npol == 2:
        print('\nReading non-colinear spinor wavefunction')
        unkg_set = read_evc_non_colinear(evc, igwx, nbnd)
    else:
        unkg_set = read_evc(evc, igwx, nbnd)
#    savetxt('unkg.{0:03d}.txt'.format(ik),unkg_set)
#    print("\nWavefunctions are saved in a text file")
    return unkg_set, miller_ids, npol


def read_evc(evc, igwx, nbnd):

    print('converting the wavefunction coefficents to numpy format')
    psi_k_set = zeros((nbnd,igwx), dtype=complex64)
    for n, row in enumerate(evc):
        psi = add(row[::2], 1j*row[1::2])
        psi_k_set[n,:] = psi

    print('converting a wavefunction file is done')

    return psi_k_set


def read_evc_non_colinear(evc, igwx, nbnd):
    #if npol == 2 the len of array will be doulbed

    print('converting the wavefunction coefficents to numpy format')
    psi_k_set = zeros((nbnd,igwx*2), dtype=complex64)
    for n, row in enumerate(evc):
        psi = add(row[::2], 1j*row[1::2])
        psi_k_set[n,:] = psi

    print('converting a wavefunction file is done')

    return psi_k_set



if __name__=='__main__':
    main()
