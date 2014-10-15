# Authors: Christian Brodbeck <christianbrodbeck@nyu.edu>
#
# License: BSD (3-clause)

import os

import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_false, assert_equal

import mne
from mne.datasets import testing
from mne.utils import (_TempDir, requires_traits, 
                       get_subjects_dir, run_tests_if_main)
from mne.io.constants import FIFF
 
sample_path = testing.data_path(download=False)
subjects_dir = os.path.join(sample_path, 'subjects')

fsaverage_hs_points_fname = os.path.join( \
        os.path.dirname(__file__), 'fsaverage_hs_points.npy')
fsaverage_fid_attach_fname = os.path.join( \
        os.path.dirname(__file__), 'fsaverage_fid_attach.npy')

@testing.requires_testing_data
@requires_traits
def test_mri_model():
    """Test MRIHeadWithFiducialsModel Traits Model"""
    from mne.gui._fiducials_gui import MRIHeadWithFiducialsModel
    tempdir = _TempDir()
    tgt_fname = os.path.join(tempdir, 'test-fiducials.fif')

    model = MRIHeadWithFiducialsModel(subjects_dir=subjects_dir)
    model.subject = 'sample'
    assert_equal(model.default_fid_fname[-20:], "sample-fiducials.fif")
    assert_false(model.can_reset)
    assert_false(model.can_save)
    model.lpa = [[-1, 0, 0]]
    model.nasion = [[0, 1, 0]]
    model.rpa = [[1, 0, 0]]
    assert_false(model.can_reset)
    assert_true(model.can_save)

    bem_fname = os.path.basename(model.bem.file)
    assert_false(model.can_reset)
    assert_equal(bem_fname, 'sample-head.fif')

    model.save(tgt_fname)
    assert_equal(model.fid_file, tgt_fname)

    # resetting the file should not affect the model's fiducials
    model.fid_file = ''
    assert_array_equal(model.lpa, [[-1, 0, 0]])
    assert_array_equal(model.nasion, [[0, 1, 0]])
    assert_array_equal(model.rpa, [[1, 0, 0]])

    # reset model
    model.lpa = [[0, 0, 0]]
    model.nasion = [[0, 0, 0]]
    model.rpa = [[0, 0, 0]]
    assert_array_equal(model.lpa, [[0, 0, 0]])
    assert_array_equal(model.nasion, [[0, 0, 0]])
    assert_array_equal(model.rpa, [[0, 0, 0]])

    # loading the file should assign the model's fiducials
    model.fid_file = tgt_fname
    assert_array_equal(model.lpa, [[-1, 0, 0]])
    assert_array_equal(model.nasion, [[0, 1, 0]])
    assert_array_equal(model.rpa, [[1, 0, 0]])

    # after changing from file model should be able to reset
    model.nasion = [[1, 1, 1]]
    assert_true(model.can_reset)
    model.reset = True
    assert_array_equal(model.nasion, [[0, 1, 0]])


def test_auto_fid_registration():
    '''Test for the automatic fiducial points registration
       The test checks if the coreg.auto_calc_fid returns the same
       fiducial points like expected, with and without
       attaching the points into the fsaverage head
       shape surface.
    '''
    # Load the fsaverage data files (included in the installation)
    fsaverage_hs_points = np.load(fsaverage_hs_points_fname)
    fsaverage_fid_attach = np.load(fsaverage_fid_attach_fname)
    
    # Indentity transformation from fsaverage to MNI_TAL
    fsaverage_trans = { \
        'to': FIFF.FIFFV_MNE_COORD_MNI_TAL, 
        'from': FIFF.FIFFV_PROJ_ITEM_HOMOG_FIELD, 
        'trans': np.eye(4, dtype=np.float32)}

    fsaverage_fid = mne.coreg.read_sfaverage_fid()
    fsaverage_auto_fid = mne.coreg.auto_calc_fid('fsaverage',
        fsaverage_hs_points, False, fsaverage_trans, subjects_dir)
    fsaverage_auto_fid_attach = mne.coreg.auto_calc_fid('fsaverage',
        fsaverage_hs_points, True, fsaverage_trans, subjects_dir)
    assert_array_equal(fsaverage_fid, fsaverage_auto_fid)
    assert_array_equal(fsaverage_fid_attach, fsaverage_auto_fid_attach)
    

run_tests_if_main()
