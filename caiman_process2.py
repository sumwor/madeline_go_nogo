import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.components_evaluation import estimate_components_quality_auto
from caiman_process import caiman_main
import numpy as np
import h5py,os, time

# For simplicity reason use this to replace multiprocessing forking child

if __name__ == '__main__':
    root = "/Users/albertqu/Documents/2.Courses/CogSci127/proj/data/"  # DATA ROOT
    fr=4
    decay = root+"merge_decay.tif"
    while not os.path.exists(decay):
        time.sleep(1)
    print('found decay')
    caiman_main(fr, fnames=[decay], out=os.path.join(root, 'out_decay.hdf5'))