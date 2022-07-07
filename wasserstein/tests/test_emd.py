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

        print(wasserstein)

        # wasserstein computation
        wassEMD = wasserstein.EMD(beta=beta, R=R, norm=(norm is True))
        wass_emd = wassEMD(ws0[:num_particles], coords0, ws1[:num_particles], coords1)

        emd_diff = abs(pot_emd - wass_emd)
        emd_percent_diff = 2*emd_diff/(pot_emd + wass_emd)
        assert emd_percent_diff < 1e-13 or emd_diff < 1e-13, 'emds do not match'

@pytest.mark.emd
@pytest.mark.emdcustom
@pytest.mark.parametrize('norm', [True, False, 'extra'])
@pytest.mark.parametrize('R', np.arange(0.2, 2.1, 0.2).tolist() + [4, 6, 8, 10])
@pytest.mark.parametrize('beta', [0.5, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('num_particles', [2, 4, 8, 16, 32])
def test_emd_custom(num_particles, dim, beta, R, norm):

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
        wassEMD = wasserstein.EMD(norm=(norm is True))
        wass_emd = wassEMD(ws0, ws1, dists)

        emd_diff = abs(pot_emd - wass_emd)
        emd_percent_diff = 2*emd_diff/(pot_emd + wass_emd)
        assert emd_percent_diff < 1e-13 or emd_diff < 1e-13, 'emds do not match'

@pytest.mark.emd
@pytest.mark.emdcustom
@pytest.mark.emdyphi
@pytest.mark.parametrize('norm', [True, False, 'extra'])
@pytest.mark.parametrize('R', np.arange(0.2, 2.1, 0.2).tolist() + [4, 6, 8, 10])
@pytest.mark.parametrize('beta', [0.5, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize('dim', [2, pytest.param(3, marks=pytest.mark.xfail(strict=True))])
@pytest.mark.parametrize('num_particles', [2, 4, 8, 16, 32])
def test_emd_yphi(num_particles, dim, beta, R, norm):

    for i in range(5):
        ws0, ws1 = np.random.rand(2, num_particles)
        coords0, coords1 = 2*np.random.rand(2, num_particles, dim) - 1
        coords0[:,1] = (coords0[:,1] + 1)/2*np.pi
        coords1[:,1] = (coords1[:,1] + 1)/2*np.pi

        # by hand computation with pot
        dists = np.sqrt([[(coords0[i,0]-coords1[j,0])**2 +
                          (np.pi - abs(np.pi - abs(coords0[i,1]-coords1[j,1])))**2
                          for j in range(num_particles)] for i in range(num_particles)])/R
        dists **= beta

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

        # wasserstein computation
        wassEMD = wasserstein.EMD(norm=(norm is True))
        wassEMDyphi = wasserstein.EMDYPhi(R=R, beta=beta, norm=(norm is True))
        wass_emd = wassEMD(ws0, ws1, dists)
        wass_emdyphi = wassEMDyphi(ws0[:num_particles], coords0, ws1[:num_particles], coords1)

        emd_diff = abs(wass_emdyphi - wass_emd)
        emd_percent_diff = 2*emd_diff/(wass_emdyphi + wass_emd)
        assert emd_percent_diff < 1e-13 or emd_diff < 1e-13, 'emds do not match'

@pytest.mark.emd
@pytest.mark.dtype
@pytest.mark.parametrize('norm', [True, False, 'extra'])
@pytest.mark.parametrize('R', np.arange(0.2, 2.1, 0.2).tolist() + [4, 6, 8])
@pytest.mark.parametrize('beta', [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('num_particles', [2, 4, 8, 16, 32])
def test_emd_dtype(num_particles, dim, beta, R, norm):

    nfail = 0
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

        # wasserstein computation
        wassEMD64 = wasserstein.EMD(beta=beta, R=R, norm=(norm is True), dtype='float64')
        wassEMD32 = wasserstein.EMD(beta=beta, R=R, norm=(norm is True), dtype='float32')
        wass_emd64 = wassEMD64(ws0[:num_particles], coords0, ws1[:num_particles], coords1)
        wass_emd32 = wassEMD32(ws0[:num_particles].astype('float32'), coords0.astype('float32'),
                               ws1[:num_particles].astype('float32'), coords1.astype('float32'))

        emd_diff = abs(wass_emd64 - wass_emd32)
        emd_percent_diff = 2*emd_diff/(wass_emd64 + wass_emd32)
        if not (emd_percent_diff < 1e-5 or emd_diff < 5e-5):
            nfail += 1

    assert nfail <= 1, 'more than 1 emds did not match'

@pytest.mark.emd
@pytest.mark.flows
@pytest.mark.dtype
@pytest.mark.parametrize('norm', [True, False])
@pytest.mark.parametrize('R', np.arange(0.2, 2.1, 0.2))
@pytest.mark.parametrize('beta', [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize('num_particles', [2, 4, 8, 16])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_emd_flows(dtype, num_particles, beta, R, norm):

    eps = {'float32': 5e-4, 'float64': 1e-14}

    nfail = 0
    for i in range(5):
        ws0, ws1 = np.random.rand(2, num_particles)
        coords0, coords1 = (2*np.random.rand(2, num_particles, 2) - 1).astype(dtype)

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
        wassEMD = wasserstein.EMD(beta=beta, R=R, norm=norm, dtype=dtype)
        wassEMD(ws0[:num_particles].astype(dtype), coords0, ws1[:num_particles].astype(dtype), coords1)

        # check that numpy and vector agree for wasserstein
        wass_flows = wassEMD.flows()
        wass_flows_vec = np.asarray(wassEMD.flows_vec()).reshape(wass_flows.shape)
        if not np.all(wass_flows == wass_flows_vec):
            nfail += 1
            m = 'array does not match vec'

        # check flows
        diff = np.abs(pot_flows - wass_flows)
        pct_diff = 2*np.abs(pot_flows - wass_flows)/(pot_flows + wass_flows + 1e-40)
        print(np.max(diff), np.max(pct_diff), 'worst flow')
        if not (np.all(diff < eps[dtype]) or np.all(pct_diff < eps[dtype])):
            nfail += 1
            m = 'flows do not match'

    # allow one failure
    assert nfail <= 1, m

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
        wassEMD = wasserstein.EMD(beta=beta, R=R, norm=norm)
        wassEMD(ws0, coords0, ws1, coords1)

        # check that numpy and vector agree for wasserstein
        wass_dists = wassEMD.dists()
        wass_dists_vec = np.asarray(wassEMD.dists_vec()).reshape(wass_dists.shape)
        assert np.all(wass_dists == wass_dists_vec)

        # check flows
        print(np.max(np.abs(dists - wass_dists[:num_particles,:num_particles])), 'worst dist diff')
        if norm:
            assert np.all(np.abs(dists - wass_dists) < 5e-12)
        elif wassEMD.extra() == 0:
            assert np.all(np.abs(dists - wass_dists[:num_particles]) < 5e-12)
            assert np.all(wass_dists[-1] == 1)
        elif wassEMD.extra() == 1:
            assert np.all(np.abs(dists - wass_dists[:,:num_particles]) < 5e-12)
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
    wassEMD = wasserstein.EMD(beta=beta, R=R, norm=norm)
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
