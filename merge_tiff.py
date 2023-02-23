import tifffile, os
import numpy as np
from skimage import io

# comment

def merge_tiffs(fls, reps, outpath, decay=1.0, fmm='bigmem', tifn='merge', order='F', del_mmap=True):
    # Takes in a list of single tiff fls and save them in memmap

    segs = len(fls)
    totlen = segs *reps
    dims = tifffile.imread(fls[0]).shape
    d3 = dims[2] if len(dims) == 3 else 1
    d1, d2 = dims[0], dims[1]
    fnamemm = os.path.join(outpath, '{}_d1_{}_d2_{}_d3_{}_order_{}_frames_{}_.mmap'.format(fmm, d1, d2, d3, order,
                                                                                          totlen))
    bigmem = np.memmap(fnamemm, mode='w+', dtype=np.float32, shape=(totlen, dims[0], dims[1]), order=order)
    # Fill the mmap
    for i in range(reps):
        for j in range(segs):
            img = tifffile.imread(fls[j])
            seq = i * segs+j
            bigmem[seq, :, :] = img * decay ** (seq)
    bigmem.flush()

    # Read from mmap, save as tifs
    tifn = os.path.join(outpath, tifn)
    fname = tifn+"_{}decay.tif".format("" if decay < 1 else "no")
    io.imsave(fname, bigmem, plugin='tifffile')
    # Delete mmap
    if del_mmap:
        os.remove(fnamemm)
        del bigmem
    return fname

if __name__ == '__main__':
    root = "/Users/albertqu/Documents/2.Courses/2019F/CogSci127/proj/data/"  # DATA ROOT
    out = os.path.join(root, 'merged')
    if not os.path.exists(out):
        os.makedirs(out)
    tiff_path = os.path.join(root, "MEK003_b-006/MEK003_b-006_Cycle00001_CurrentSettings_Ch1_{:06d}.tif")
    N, reps = 200, 25
    fnames = [tiff_path.format(i) for i in range(1, N + 1)]
    # nodecay
    fname2 = merge_tiffs(fnames, reps, out)
    #decay
    fname1 = merge_tiffs(fnames, reps, out, decay=0.9999)
