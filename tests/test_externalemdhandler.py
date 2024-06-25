import platform

import energyflow as ef
import numpy as np
import ot
import pytest

import wasserstein

loaded_ef_events = False

@pytest.mark.externalemdhandler
@pytest.mark.corrdim
@pytest.mark.parametrize('nbins', [pytest.param(0, marks=pytest.mark.xfail), 1, 2, 4, 1000])
@pytest.mark.parametrize('high', [10.0000001, 100, 1000])
@pytest.mark.parametrize('low', [1e-10, 1, 10])
def test_corrdim(low, high, nbins):

    nevents = 150

    corrdim = wasserstein.CorrelationDimension(nbins, low, high)
    emds = wasserstein.PairwiseEMD(throw_on_error=True)
    emds.set_external_emd_handler(corrdim)

    #if platform.system() == 'Darwin':
    #    pytest.skip()

    global X, y, loaded_ef_events
    if not loaded_ef_events:
        X, y = ef.qg_jets.load(num_data=nevents, pad=False)
        for x in X:
            x[:,1:3] -= np.average(x[:,1:3], weights=x[:,0], axis=0)
        loaded_ef_events = True

    emds(X, gdim=2)

    # ensure sizes of things
    assert corrdim.num_calls() == nevents*(nevents - 1)//2
    assert corrdim.nbins() == nbins
    assert len(corrdim.bin_centers()) == nbins
    assert len(corrdim.bin_edges()) == nbins + 1
    assert len(corrdim.hist_vals_errs()[0]) == nbins + 2
    assert len(corrdim.hist_vals_errs(False)[0]) == nbins
    assert len(corrdim.cumulative_vals_vars()[0]) == nbins
    assert len(corrdim.corrdims()[0]) == nbins - 1
    assert len(corrdim.corrdim_bins()) == nbins - 1

    # ensure numpy arrays match vectors
    assert np.all(corrdim.bin_centers() == corrdim.bin_centers_vec())
    assert np.all(corrdim.bin_edges() == corrdim.bin_edges_vec())
    assert np.all(np.asarray(corrdim.hist_vals_vars(True)) == np.asarray(corrdim.hist_vals_vars_vec(True)))
    assert np.all(np.asarray(corrdim.hist_vals_vars(False)) == np.asarray(corrdim.hist_vals_vars_vec(False)))
    assert np.all(np.asarray(corrdim.cumulative_vals_vars()) == np.asarray(corrdim.cumulative_vals_vars_vec()))
    assert np.all(np.asarray(corrdim.corrdims()) == np.asarray(corrdim.corrdims_vec()))
