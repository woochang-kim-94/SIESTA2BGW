import numpy as np
import sys
import h5py
import scipy.spatial

def get_coeffs_kbarr(ik_arr, ib_arr, gvecs, ngk, coeffs, nspinor):
    coeffs_kbarr = []
    gvecs_kbarr = []
    for ik in ik_arr:
        tmp_c = []
        tmp_g = []
        for ib in ib_arr:
            coeffs_ikib, gvecs_ikib = get_coeffs(ik, ib, gvecs, ngk, coeffs, nspinor)
            tmp_c.append(coeffs_ikib)
            tmp_g.append(gvecs_ikib)
        coeffs_kbarr.append(tmp_c)
        gvecs_kbarr.append(tmp_g)
    return np.array(coeffs_kbarr), np.array(gvecs_kbarr)

def get_eig_siesta(f_name, ib_start, ib_end):
	fp = open(f_name, 'r')
	lines = fp.readlines()
	for i in range(10):
		if  "E_F" in lines[i]:
			w = lines[i].split()
			Ef = eval(w[-1])
		if  "Nbands, Nspin, Nk" in lines[i]:
			w = lines[i].split()
			Nk, nbnd = eval(w[-1]), eval(w[-3])
	fp.close()
	print(Nk, nbnd)
	Eig = np.loadtxt(f_name)

	Eig_bk = Eig.reshape((nbnd, Nk, 3))
	Eig_bk_trunc = Eig_bk[ib_start -1: ib_end, :,1]
	return Eig_bk_trunc


def get_coeffs(ik,ib,gvecs,ngk, coeffs, nspinor):
    kstart = np.sum(ngk[:ik])
    kend = kstart + ngk[ik]
    coeffs_x = coeffs[ib,0:nspinor,kstart:kend, 0] + 1j*coeffs[ib,0:nspinor,kstart:kend, 1]
    gvecs_x = gvecs[kstart:kend]
    return coeffs_x, gvecs_x


if __name__ == "__main__":
	ib_start, ib_end = 349, 380
	nb = 32
	Ry2ev = 13.605698
	
	Eig_bk_S = get_eig_siesta("WS2.bands.txt", ib_start, ib_end)
	np.save("Eig_bk_S", Eig_bk_S)
	
	f_QE = h5py.File("WFN_16v16c.h5", 'r')
	
	f_sies = h5py.File("WFN_16v16c_QE2SIESTA.h5", 'w')
	f_sies.copy(f_QE['mf_header'], 'mf_header')
	f_sies.create_group('/wfns')
	f_sies.copy(f_QE['/wfns/gvecs'], '/wfns/gvecs')
	f_sies.create_dataset('/wfns/coeffs', shape=f_QE['/wfns/coeffs'].shape, dtype=f_QE['/wfns/coeffs'].dtype)
	
	nspinor = f_QE['mf_header/kpoints/nspinor'][()]
	ngk_Q = f_QE['mf_header/kpoints/ngk'][:]
	el_Q = f_QE['mf_header/kpoints/el'][:]
	gvecs_Q = f_QE['wfns/gvecs'][:]
	coeffs_Q = f_QE['wfns/coeffs'][:]
	
	# Initialize new coeffs and eigenvalues
	coeffs_S2Q = np.zeros_like(coeffs_Q)
	el_S = np.zeros_like(el_Q)
	print(coeffs_S2Q.shape)
	nrk  = f_QE['mf_header/kpoints/nrk'][()]
	
	#Eig_bk_trunc = np.load("Eig_bk_trunc.npy")
	#print(np.max(Eig_bk_trunc - Eig_sies))
	
	for ik in range(nrk):
		ik_arr = [ik]
		ib_arr = range(nb)
		gvk_S = np.load("Gvecs_k" + str(ik + 1) + ".npy")
		cG_S = np.load("Unkg_k" + str(ik + 1) + ".npy")
		print(len(gvk_S), ngk_Q[ik])
		cG_S = cG_S.reshape((nb, nspinor, len(gvk_S)))
		cG_kb_Q, gv_kb_Q = get_coeffs_kbarr(ik_arr, ib_arr, gvecs_Q, ngk_Q, coeffs_Q, nspinor)
		tree_QE = scipy.spatial.cKDTree(gvk_S)
		dist, ind = tree_QE.query(gv_kb_Q[0,0], distance_upper_bound = 1e-9)
		if ind.any() == len(gv_kb_Q[0,0]):
			print("G not found!!")
		cG_S_sort = cG_S[:,:,ind]	
		gvk_S_sort = gvk_S[ind]	
		for ib in range(nb):
			#dp1 = np.dot(np.conj(cG_kb_Q[0,ib,0,:]), cG_S_sort[ib,0,:])
			#dp2 = np.dot(np.conj(cG_kb_Q[0,ib,1,:]), cG_S_sort[ib,1,:])
			#dp1 = np.dot(np.conj(cG_kb_Q[0,ib,0,:]), cG_kb_Q[0,ib,0,:])
			#dp2 = np.dot(np.conj(cG_kb_Q[0,ib,1,:]), cG_kb_Q[0,ib,1,:])
			dp1 = np.dot(np.conj(cG_S_sort[ib,0,:]), cG_S_sort[ib,0,:])
			dp2 = np.dot(np.conj(cG_S_sort[ib,1,:]), cG_S_sort[ib,1,:])
			renorm = np.sqrt(np.abs(dp1 + dp2))
			print(ik, ib, renorm**2)
			cG_S_sort[ib,:,:] = cG_S_sort[ib,:,:]/renorm
			el_S[0,ik,ib] = Eig_bk_S[ib,ik]
		ikstart = np.sum(ngk_Q[:ik])
		ikend = ikstart + ngk_Q[ik]
		coeffs_S2Q[:,:,ikstart:ikend, 0] = cG_S_sort.real
		coeffs_S2Q[:,:,ikstart:ikend, 1] = cG_S_sort.imag
		
	f_sies['/wfns/coeffs'][:] = coeffs_S2Q
	f_sies['mf_header/kpoints/el'][:] = el_S/Ry2ev
