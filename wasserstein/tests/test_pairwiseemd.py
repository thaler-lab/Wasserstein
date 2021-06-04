import platform

import energyflow as ef
import numpy as np
import pytest

import wasserstein

@pytest.mark.pairwise_emd
@pytest.mark.parametrize('store_sym_raw', [True, False])
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('print_every', [-2, -1, 0, 1, 2, 1000000000])
@pytest.mark.parametrize('num_threads', [1, 2, -1])
@pytest.mark.parametrize('num_events', [1, 2, 16, 64])
def test_pairwise_emd(num_events, num_threads, print_every, norm, store_sym_raw):

    beta, R = 1.0, 1.0
    eventsA, eventsB = np.random.rand(2, num_events, 10, 3)

    wassEMD = wasserstein.EMD(beta=beta, R=R, norm=norm)
    wassPairwiseEMD = wasserstein.PairwiseEMD(beta=beta, R=R, norm=norm, print_every=print_every,
                        num_threads=num_threads, store_sym_emds_raw=store_sym_raw, verbose=False)

    # symmetric computation
    wassPairwiseEMD(eventsA)
    wassPairedEMDs = wassPairwiseEMD.emds()
    wassVecEMDs = np.asarray(wassPairwiseEMD.emds_vec()).reshape(wassPairedEMDs.shape)
    assert np.all(wassPairedEMDs == wassVecEMDs)

    wassEMDs = np.zeros((num_events, num_events))
    for i in range(num_events):
        for j in range(i):
            wassEMDs[i,j] = wassEMDs[j,i] = wassEMD(eventsA[i][:,0], eventsA[i][:,1:], eventsA[j][:,0], eventsA[j][:,1:])

    assert np.all(np.abs(wassPairedEMDs - wassEMDs) < 1e-16)

    # all pairs computation
    wassPairwiseEMD(eventsA, eventsB)
    wassPairedEMDs = wassPairwiseEMD.emds()
    wassVecEMDs = np.asarray(wassPairwiseEMD.emds_vec()).reshape(wassPairedEMDs.shape)
    assert np.all(wassPairedEMDs == wassVecEMDs)

    wassEMDs = np.zeros((num_events, num_events))
    for i in range(num_events):
        for j in range(num_events):
            wassEMDs[i,j] = wassEMD(eventsA[i][:,0], eventsA[i][:,1:], eventsB[j][:,0], eventsB[j][:,1:])

    assert np.all(np.abs(wassPairedEMDs - wassEMDs) < 1e-16)

@pytest.mark.energyflow
@pytest.mark.pairwise_emd
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('R', [0.4, 0.7, 1.0, 1.3])
@pytest.mark.parametrize('beta', [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize('num_threads', [1, -1])
@pytest.mark.parametrize('num_particles', [4, 12])
@pytest.mark.parametrize('num_events', [1, 2, 16, 64])
def test_pairwise_emd_with_ef(num_events, num_particles, num_threads, beta, R, norm):

    #if platform.system() == 'Windows':
    #    pytest.skip()

    import energyflow as ef
    eventsA, eventsB = np.random.rand(2, num_events, num_particles, 3)

    wassPairwiseEMD = wasserstein.PairwiseEMD(beta=beta, R=R, norm=norm, num_threads=num_threads, verbose=False)

    # symmetric computation
    wassPairwiseEMD(eventsA)
    wassEMDs = wassPairwiseEMD.emds()
    wassVecEMDs = np.asarray(wassPairwiseEMD.emds_vec()).reshape(wassEMDs.shape)
    assert np.all(wassEMDs == wassVecEMDs)

    nj = num_threads if num_threads != -1 else None
    if hasattr(ef.emd, 'emds_pot'):
        efEMDs = ef.emd.emds_pot(eventsA, R=R, beta=beta, norm=norm, n_jobs=nj)
    else:
        efEMDs = ef.emd.emds(eventsA, R=R, beta=beta, norm=norm, n_jobs=nj)

    print(wassEMDs)
    assert np.all(np.abs(efEMDs - wassEMDs) < 1e-13)

    # all pairs computation
    wassPairwiseEMD(eventsA, eventsB)
    wassEMDs = wassPairwiseEMD.emds()
    wassVecEMDs = np.asarray(wassPairwiseEMD.emds_vec()).reshape(wassEMDs.shape)
    assert np.all(wassEMDs == wassVecEMDs)

    nj = num_threads if num_threads != -1 else None
    if hasattr(ef.emd, 'emds_pot'):
        efEMDs = ef.emd.emds_pot(eventsA, eventsB, R=R, beta=beta, norm=norm, n_jobs=nj)
    else:
        efEMDs = ef.emd.emds(eventsA, eventsB, R=R, beta=beta, norm=norm, n_jobs=nj)

    print(wassEMDs)
    assert np.all(np.abs(efEMDs - wassEMDs) < 1e-13)