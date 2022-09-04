#!/usr/bin/env python

"""
Written by W. Kim

Aug. 10. 2022

Plot band structures from the pw.x input fle and the output xml file.
Here, we assume that, pw.bands.in is the input file for bands calculation.
"""
from numpy import *
import re
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (3.8/2.54,4.3/2.54)
plt.rcParams["font.size"] = 8
hatree  =  2*13.605662285137 #eV

def main():
    fermi = 0.1979
    eigs, k_points = read_xml_bands('WSe2.xml')
    high_sym_i = read_high_sym('./pw.bands.in')
    high_sym_labels = [r'$\Gamma$','K','M', r'$\Gamma$']

    eigs = eigs - fermi
    x_k = zeros(len(k_points))
    x = 0
    x_k[0] = x
    for i_k in range(len(k_points)):
        if i_k != 0:
            x += linalg.norm(k_points[i_k]-k_points[i_k-1])
        x_k[i_k] = x

    high_sym_points = x_k[high_sym_i]

    nbnd = shape(eigs)[1]
    for i_bnd in range(nbnd):
        plt.plot(x_k, eigs[:,i_bnd], c='black', linewidth=0.5)

    plt.axhline(0, c='grey', linestyle='dotted', linewidth=0.5)
    for x in high_sym_points[1:-1]:
        plt.axvline(x, c='grey', linestyle='dotted', linewidth=0.5)
    plt.xticks(high_sym_points, high_sym_labels)
    plt.xlim(x_k[0], x_k[-1])
    #plt.ylim(-0.6, 0.6)
    #plt.yticks(0.3*array(range(-2,3)))
    plt.ylim(-2, 3)
    plt.ylabel('Energy (eV)')
 #   plt.yticks([-0.1, 0, 0.1])
    plt.savefig('bands.png', dpi=400, bbox_inches='tight', pad_inches=0.01)

    return None

def read_high_sym(fn):
    f = open(fn, 'r')

    line = f.readline()
    while not 'K_POINTS' in line:
        line = f.readline()

    num_sym = int(f.readline().split()[0])
    num_k = 0
    i_ks = zeros(num_sym)
    for i_sym in range(1,len(i_ks)):
        num_k += int(f.readline().split()[3])
        i_ks[i_sym] = num_k
    i_ks = array(i_ks, dtype=int)
    return i_ks

def read_xml_bands(fn):
    f = open(fn, 'r')

    nks = 0
    nbnd = 0
    line = f.readline()
    while not '<nbnd>' in line:
        line = f.readline()
    words = re.split('<|>|\n',line)
    nbnd = int(words[2])

    while not'<nks>' in line:
        line = f.readline()
    words = re.split('<|>|\n',line)
    nks = int(words[2])
    k_points = zeros((nks, 3))
    eigs = zeros((nks, nbnd))
    line = f.readline()
    for ik in range(nks):
        line = f.readline() #<ks_energies>
        words = re.split('<|>|\n',f.readline()) #<k_point~
        k_point = float64(words[2].split())
        k_points[ik,:] = k_point
        line = f.readline() # <npw>42767</npw>
        line = f.readline() #  <eigenvalues size="360">
        eigs_ik = []
        while not '</eigenvalues>' in line:
            line = f.readline()
            eigs_ik.extend(line.split())
        eigs_ik.pop() # last element is '</eigenvalues>'
        eigs[ik,:] = array(eigs_ik)
        while not '</ks_energies>' in line:
            line = f.readline()

    savetxt('eigs.txt', eigs)
    savetxt('k_points.txt', k_points)
    return hatree*eigs, k_points

if __name__=='__main__':
    main()
