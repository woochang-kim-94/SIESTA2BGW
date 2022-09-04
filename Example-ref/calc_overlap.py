#!/usr/bin/env python

import numpy as np
from scipy.spatial import KDTree
bohr = 0.529177211 #Angstrom




def main():
    ik_lst = list(range(1, 92))
    nk = len(ik_lst)
    overlap_matrix = np.zeros((nk, 56-37+1), dtype=np.complex64)
    for ik in ik_lst:
        wfc_qe_fn = './QE/WSe2.save/unkg_blst._ik.{0:03d}._ibndmin.037_ibndmax.056.txt'.format(ik)
        miller_ids_qe_fn = './QE/WSe2.save/miller_ids.{0:03d}.txt'.format(ik)
 #       wfc_sie_fn = './siesta/unkg._ik.031._ibnd.024.txt'
        miller_ids_sie_fn = './Siesta/Siesta2bgw.save/miller_ids.{0:03d}.txt'.format(ik)
        wfn_sie_fn = './Siesta/Siesta2bgw.save/unkg_blst._ik.{0:03d}._ibndmin.037_ibndmax.056.txt'.format(ik)
        wfc_qe = np.loadtxt(wfc_qe_fn, dtype=np.complex64)
        miller_ids_qe = np.loadtxt(miller_ids_qe_fn, dtype=np.int32)
        wfc_sie = np.loadtxt(wfn_sie_fn, dtype=np.complex64)
        miller_ids_sie = np.loadtxt(miller_ids_sie_fn, dtype=np.int32)
        for ibnd in range(0, 56-37+1):
            overlap = calc_overlap_noncolin(wfc_qe[ibnd,:], miller_ids_qe, wfc_sie[ibnd,:], miller_ids_sie)
            print('ik, ibnd:', ik, ibnd+37)
            print('   overlap:', overlap)
            print('overlap_sq:', abs(overlap)**2)
            print('\n')
            overlap_matrix[ik-1, ibnd] = overlap
    np.savetxt('overlap_matrix._ibndmin.037_ibndmax.056.txt', overlap_matrix)

    return None


def calc_overlap_noncolin(wfc_qe, miller_ids_qe, wfc_sie, miller_ids_sie):
    ngw_qe  = len(miller_ids_qe)
    ngw_sie = len(miller_ids_sie)
    print('Num of qe  FFT grid:',ngw_qe)
    print('Num of sie FFT grid:',ngw_sie)
    overlap = np.complex64(0.0)
    tree = KDTree(miller_ids_sie)
    count = 0
    for igvec_qe, hkl_qe in enumerate(miller_ids_qe): #For a given G_qe in {G_qe}
        dist, igvec_sie = tree.query([hkl_qe],k=1)    #Find the id of G_sie=G_qe in {G_sie}
        if dist[0] < 1e-8: #IF same one found
            count += 1
            igvec_sie = igvec_sie[0]
            overlap += np.conjugate(wfc_sie[igvec_sie])*wfc_qe[igvec_qe] #Add the spin_up part
            overlap += np.conjugate(wfc_sie[ngw_sie+igvec_sie])*wfc_qe[ngw_qe+igvec_qe] #Add the spin_dw part

    print('Num of matched FFT grid:',count)
    return overlap


def calc_overlap(wfc_qe, miller_ids_qe, wfc_sie, miller_ids_sie):

    overlap = np.complex64(0.0)
    tree = KDTree(miller_ids_sie)
    count = 0
    for igvec_qe, hkl_qe in enumerate(miller_ids_qe): #For a given G_qe in {G_qe}
        dist, igvec_sie = tree.query([hkl_qe],k=1)    #Find the id of G_sie=G_qe in {G_sie}
        if dist[0] < 1e-8: #IF same one found
            count += 1
            igvec_sie = igvec_sie[0]
            overlap += np.conjugate(wfc_sie[igvec_sie])*wfc_qe[igvec_qe] #Add the coeff
    print('Num of FFT grid:',count)

    return overlap

if __name__=='__main__':
    main()
