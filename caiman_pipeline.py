from merge_tiff import *
from caiman_process import *
from data_proc import *
from gui import *
from folder_scan import *
from itertools import product
import os
import time


"""
Pipeline Settings
===============================================================================
gui_enable: set to enable gui for selecting input
folder_scan_enable: set to create input_folders at runtime by searching root
    * in this case, parameter input_folders is ignored
"""
gui_enable = False
folder_scan_enable = False


"""
File Parameters:
===============================================================================
root_input_folder: default folder to begin input path
input_folders: subfolders of root that contain independent trial data
output_folder: folder to save output file(s)
"""
root_input_folder = r'\\filenest.diskstation.me\Wilbrecht_file_server\Madeline\raw_imaging'
input_folders = [
    r'JUV011\JUV011-211215-gonogo-001'
]
output_folder = r'C:\Users\right\Desktop\lab\hard drive\madeline_data_output_4'

"""
CaImAn Parameters:
===============================================================================
=> All possible outputs will be produced
fr: framerate
K: number of components per patch
rf: half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf: amount of overlap between the patches in pixels
gnb: number of global background components
"""
fr = 10.055
param_names = ['K', 'rf', 'stride_cnmf', 'gnb']
params = [
    [10, 15],  # K
    [20, 25, 30],  # rf
    [8, 10, 12],  # stride_cnmf
    # [1, 2, 3],  # gnb
]

# create pandas dataframe from caiman parameters
param_configs = list(product(*params))

# get input from gui
if gui_enable:
    gui_input = run_gui(root_input_folder, output_folder, fr)
    if gui_input is None:
        raise RuntimeError('No input')
    root_input_folder, output_folder, fr = gui_input
    print('--successful gui input--')

# ensure output folder exists
print('--looking for output folder--')
while not os.path.exists(output_folder):
    time.sleep(1)
print('--found output folder--')

# folder scan starting at root input folder
if folder_scan_enable:
    while not os.path.exists(root_input_folder):
        time.sleep(1)
    print('--found input folder--')
    input_folders = scan_folder(root_input_folder)
    print('--successful folder scan--')

# ensure input folders exist
print('--looking for input folders--')
for input_folder in input_folders:
    input_folder = os.path.join(root_input_folder, input_folder)
    while not os.path.exists(input_folder):
        time.sleep(1)
print('--found all input folders--')

# analyze all input folders
for input_folder in input_folders:
    input_folder = os.path.join(root_input_folder, input_folder)
    try:
        print(f'--BEGINNING PROCESSING OF INPUT FOLDER: {input_folder}--')

        # merge tif files
        tif_name = os.path.basename(input_folder)+'_merge0.tif'
        merged_tif = os.path.join(output_folder, tif_name)
        if os.path.exists(merged_tif):
            print('--merged tiff file already found! skipping merge')
        else:
            print('--begin merging tiff files--')
            merge_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if ('.tif' in f)]
            outfile = merge_tiffs(merge_files, output_folder, tifn=tif_name)
            print('--tiff files merged--')

            # clean mmap files
            for f in os.listdir(output_folder):
                if f.endswith('.mmap'):
                    os.remove(os.path.join(output_folder, f))

            # ensure merged tif file exists
            merged_tif = outfile[0]
            while not os.path.exists(merged_tif):
                time.sleep(1)
            print('--found merged tif file--')

        # process raw tif and output hdf5 for all param configs
        for param_config in param_configs:
            print('Parameters:', ', '.join([a + " = " + str(b) for a, b in zip(param_names, param_config)]))
            param_str = "-".join(map(str, param_config))
            output_file = os.path.basename(input_folder) + "_" + param_str + '.hdf5'
            outpath = os.path.join(output_folder, output_file)
            if os.path.exists(outpath):
                print('--skipping since folder already processed--')
                continue
            caiman_main(fr, [merged_tif], outpath, *param_config)
            print('--raw tif file processed--')

        # ensure hdf5 file exists
        while not os.path.exists(outpath):
            time.sleep(1)
        print('--found hdf5 file--')
    except Exception as E:
        print(f'ERROR: processing of input folder failed: {input_folder}')
        print(E)
        f = open(outpath, 'w')
        f.write(str(E))
        f.close()
