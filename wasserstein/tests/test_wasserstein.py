import platform

import numpy as np
import ot
import pytest

import wasserstein

@pytest.mark.emd
@pytest.mark.parametrize('norm', [True, False, 'extra'])
@pytest.mark.parametrize('R', np.arange(0.2, 2.1, 0.2).tolist() + [4, 6, 8, 10])
@pytest.mark.parametrize('beta', [0.5, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('num_particles', [2, 4, 8, 16, 32])
def test_emd(num_particles, dim, beta, R, norm):

    for i in range(5):
        ws0, ws1 = np.random.rand(2, num_particles)
        coords0, coords1 = 2*np.random.rand(2, num_particles, dim) - 1

        # by hand computation with pot
        dists = (ot.dist(coords0, coords1, metric='euclidean')/R)**beta
        if norm is True:
            ws0 /= np.sum(ws0)
            ws1 /= np.sum(ws1)
        elif norm == 'extra':
            diff = np.sum(ws0) - np.sum(ws1)
            if diff > 0:
                ws1 = np.append(ws1, diff)
                dists = np.hstack((dists, np.ones((num_particles, 1))))
            elif diff < 0:
                ws0 = np.append(ws0, abs(diff))
                dists = np.vstack((dists, np.ones(num_particles)))
        else:
            ws0 /= np.sum(ws0)
            ws0 *= np.sum(ws1)
            ws0[0] += np.sum(ws1) - np.sum(ws0)

        pot_emd = ot.emd2(ws0, ws1, dists)

        # sometimes pot fails (especially on mac)
        if pot_emd == 0:
            pytest.skip()

        # wasserstein computation
        wassEMD = wasserstein.EMDArrayEuclidean(beta=beta, R=R, norm=(norm is True))
        wass_emd = wassEMD(ws0[:num_particles], coords0, ws1[:num_particles], coords1)

        emd_diff = abs(pot_emd - wass_emd)
        emd_percent_diff = 2*emd_diff/(pot_emd + wass_emd)
        assert emd_percent_diff < 1e-13 or emd_diff < 1e-13, 'emds do not match'

@pytest.mark.emd
@pytest.mark.flows
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('R', np.arange(0.2, 2.1, 0.2))
@pytest.mark.parametrize('beta', [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize('num_particles', [2, 4, 8, 16])
def test_emd_flows(num_particles, beta, R, norm):

    for i in range(5):
        ws0, ws1 = np.random.rand(2, num_particles)
        coords0, coords1 = 2*np.random.rand(2, num_particles, 2) - 1

        # by hand computation with pot
        dists = (ot.dist(coords0, coords1, metric='euclidean')/R)**beta
        if norm is True:
            ws0 /= np.sum(ws0)
            ws1 /= np.sum(ws1)
        else:
            diff = np.sum(ws0) - np.sum(ws1)
            if diff > 0:
                ws1 = np.append(ws1, diff)
                dists = np.hstack((dists, np.ones((num_particles, 1))))
            elif diff < 0:
                ws0 = np.append(ws0, abs(diff))
                dists = np.vstack((dists, np.ones(num_particles)))

        pot_flows, pot_log = ot.emd(ws0, ws1, dists, log=True)
        pot_emd = pot_log['cost']

        # sometimes pot fails (especially on mac)
        if pot_emd == 0:
            pytest.skip()

        # wasserstein computation
        wassEMD = wasserstein.EMDArrayEuclidean(beta=beta, R=R, norm=norm)
        wassEMD(ws0[:num_particles], coords0, ws1[:num_particles], coords1)

        # check that numpy and vector agree for wasserstein
        wass_flows = wassEMD.flows()
        wass_flows_vec = np.asarray(wassEMD.flows_vec()).reshape(wass_flows.shape)
        assert np.all(wass_flows == wass_flows_vec)

        # check flows
        print(np.max(np.abs(pot_flows - wass_flows)), 'worst flow')
        assert np.all(np.abs(pot_flows - wass_flows) < 1e-14), 'flows do not match'

@pytest.mark.emd
@pytest.mark.dists
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('R', np.arange(0.4, 1.7, 0.2))
@pytest.mark.parametrize('beta', [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize('num_particles', [2, 4, 8, 16])
def test_emd_dists(num_particles, beta, R, norm):

    for i in range(5):
        ws0, ws1 = np.random.rand(2, num_particles)
        coords0, coords1 = 2*np.random.rand(2, num_particles, 2) - 1

        # by hand computation with pot
        dists = (ot.dist(coords0, coords1, metric='euclidean')/R)**beta

        # wasserstein computation
        wassEMD = wasserstein.EMDArrayEuclidean(beta=beta, R=R, norm=norm)
        wassEMD(ws0, coords0, ws1, coords1)

        # check that numpy and vector agree for wasserstein
        wass_dists = wassEMD.dists()
        wass_dists_vec = np.asarray(wassEMD.dists_vec()).reshape(wass_dists.shape)
        assert np.all(wass_dists == wass_dists_vec)

        # check flows
        print(np.max(np.abs(dists - wass_dists[:num_particles,:num_particles])), 'worst dist diff')
        if norm:
            assert np.all(np.abs(dists - wass_dists) < 5e-14)
        elif wassEMD.extra() == 0:
            assert np.all(np.abs(dists - wass_dists[:num_particles]) < 5e-14)
            assert np.all(wass_dists[-1] == 1)
        elif wassEMD.extra() == 1:
            assert np.all(np.abs(dists - wass_dists[:,:num_particles]) < 5e-14)
            assert np.all(wass_dists[:,-1] == 1)

@pytest.mark.emd
@pytest.mark.attributes
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('R', np.arange(0.2, 2.1, 0.2))
@pytest.mark.parametrize('beta', [0.5, 1.0, 1.5, 2.0])
def test_emd_attributes(beta, R, norm):

    n0, n1 = np.random.randint(5, 50, size=2)
    ws0, ws1 = np.random.rand(n0), np.random.rand(n1)
    coords0, coords1 = 2*np.random.rand(n0, 2) - 1, 2*np.random.rand(n1, 2) - 1

    # wasserstein computation
    wassEMD = wasserstein.EMDArrayEuclidean(beta=beta, R=R, norm=norm)
    wassEMD(ws0, coords0, ws1, coords1)

    if norm:
        assert wassEMD.n0() == n0 and wassEMD.n1() == n1
        assert wassEMD.scale() == 1
    else:
        extra = wassEMD.extra()
        if extra == 0:
            assert wassEMD.n0() == n0 + 1 and wassEMD.n1() == n1
        elif extra == 1:
            assert wassEMD.n0() == n0 and wassEMD.n1() == n1 + 1
        assert abs(wassEMD.scale() - max(np.sum(ws0), np.sum(ws1))) < 5e-14

@pytest.mark.pairwise_emd
@pytest.mark.parametrize('store_sym_flattened', [True, False])
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('chunksize', [-2, -1, 0, 1, 2, 1000000000])
@pytest.mark.parametrize('num_threads', [1, 2, 4, -1])
@pytest.mark.parametrize('num_events', [1, 2, 16, 128])
def test_pairwise_emd(num_events, num_threads, chunksize, norm, store_sym_flattened):

    beta, R = 1.0, 1.0
    eventsA, eventsB = np.random.rand(2, num_events, 10, 3)

    wassEMD = wasserstein.EMDArrayEuclidean(beta=beta, R=R, norm=norm)
    wassPairwiseEMD = wasserstein.PairwiseEMDArrayEuclidean(beta=beta, R=R, norm=norm, 
                        num_threads=num_threads, store_sym_emds_flattened=store_sym_flattened)

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
@pytest.mark.parametrize('num_threads', [1, 2, -1])
@pytest.mark.parametrize('num_particles', [4, 12])
@pytest.mark.parametrize('num_events', [1, 2, 16, 128])
def test_pairwise_emd_with_ef(num_events, num_particles, num_threads, beta, R, norm):

    if platform.system() == 'Windows':
        pytest.skip()

    import energyflow as ef
    eventsA, eventsB = np.random.rand(2, num_events, num_particles, 3)

    wassPairwiseEMD = wasserstein.PairwiseEMDArrayEuclidean(beta=beta, R=R, norm=norm, num_threads=num_threads)

    # symmetric computation
    wassPairwiseEMD(eventsA)
    wassEMDs = wassPairwiseEMD.emds()
    wassVecEMDs = np.asarray(wassPairwiseEMD.emds_vec()).reshape(wassEMDs.shape)
    assert np.all(wassEMDs == wassVecEMDs)

    nj = num_threads if num_threads != -1 else None
    efEMDs = ef.emd.emds(eventsA, R=R, beta=beta, norm=norm, n_jobs=nj)

    print(wassEMDs)
    assert np.all(np.abs(efEMDs - wassEMDs) < 1e-13)

    # all pairs computation
    wassPairwiseEMD(eventsA, eventsB)
    wassEMDs = wassPairwiseEMD.emds()
    wassVecEMDs = np.asarray(wassPairwiseEMD.emds_vec()).reshape(wassEMDs.shape)
    assert np.all(wassEMDs == wassVecEMDs)

    nj = num_threads if num_threads != -1 else None
    efEMDs = ef.emd.emds(eventsA, eventsB, R=R, beta=beta, norm=norm, n_jobs=nj)

    print(wassEMDs)
    assert np.all(np.abs(efEMDs - wassEMDs) < 1e-13)
