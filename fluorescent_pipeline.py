# pipeline for fluorescent analysis
# todo:
# 0. check the sessions are correcly cut
# 0. logistic regression - check the impact of prior trial outcomes
# 1. decoding info from PCA results? - to confirm the info is maintained after PCA
# 1.1: decoding percentage of neurons important for s/action/outcome
# 2. PCA:  single trial plot? check the PCA results first: time frame [-1 - 2 s]
#    distance precue across different PCs
# 3. noise variation analysis - refer to Rowland 2023.

# %matplotlib inline
import pandas as pd
from collections import Counter
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from pyPlotHW import *
from utility_HW import bootstrap, count_consecutive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.stats import binomtest
from behavioral_pipeline import BehaviorMat, GoNogoBehaviorMat
import os
from utils_signal import *
import pickle
from packages.decodanda_master.decodanda import Decodanda
import glob
import seaborn as sns

from scipy.stats import wilcoxon, mannwhitneyu,ttest_ind
from imblearn.over_sampling import SMOTE
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import random

#import matlab.engine

warnings.filterwarnings("ignore")
# read df/f and behavior data, create a class with behavior and df/f data
class Suite2pSeries:

    def __init__(self, suite2p):
        Fraw = np.load(os.path.join(suite2p, 'F.npy'))
        ops = np.load(os.path.join(suite2p, 'ops.npy'), allow_pickle=True)
        neuropil = np.load(os.path.join(suite2p, 'Fneu.npy'))
        cells = np.load(os.path.join(suite2p, 'iscell.npy'))
        stat = np.load(os.path.join(suite2p, 'stat.npy'), allow_pickle=True)
        self.Fraw = Fraw
        self.ops = ops
        self.neuropil = neuropil
        self.cells = cells
        self.stat = stat

    def get_dFF(self, beh_object, savefluopath):
        if not os.path.exists(savefluopath):
            F = self.Fraw - self.neuropil * 0.7  # subtract neuropil
        # find number of cells
            numcells = np.sum(self.cells[:, 0] == 1.0)
        # create a new array (Fcells) with F data for each cell
            Fcells = F[self.cells[:, 0] == 1.0]

        # filter the raw fluorscent data, finding the baseline?
            F0_AQ = np.zeros(Fcells.shape)
            for cell in tqdm(range(Fcells.shape[0])):
                F0_AQ[cell] = robust_filter(Fcells[cell], method=12, window=200, optimize_window=2, buffer=False)[:, 0]

            dFF = np.zeros(Fcells.shape)
            print("Calculating dFFs........")
            for cell in tqdm(range(0, Fcells.shape[0])):
                for frame in range(0, Fcells.shape[1]):
                    dFF[cell, frame] = (Fcells[cell, frame] - F0_AQ[cell, frame]) / F0_AQ[cell, frame]

            self.neural_df = pd.DataFrame(data=dFF.T, columns=[f'neuron{i}' for i in range(numcells)])
            self.neural_df['time'] = np.arange(self.neural_df.shape[0])

            self.realign_time(beh_object)
            self.neural_df.to_csv(savefluopath)
        else:
            self.neural_df = pd.read_csv(savefluopath)
            self.neural_df = self.neural_df.drop(self.neural_df.columns[0], axis=1)

    def realign_time(self, reference=None):  # need the behavior mat as reference
        if isinstance(reference, BehaviorMat):
            transform_func = lambda ts: reference.align_ts2behavior(ts)

        if self.neural_df is not None:
            # aligned to time 0
            self.neural_df['time'] = transform_func(self.neural_df['time'])-reference.time_0


    # def calculate_dff(self):
    #     rois = list(self.neural_df.columns[1:])
    #     melted = pd.melt(self.neural_df, id_vars='time', value_vars=rois, var_name='roi', value_name='ZdFF')
    #     return melted
    #
    def calculate_dff(self, method='robust', melt=True): # provides different options for how to calculate dF/F
        # results are wrong
        time_axis = self.neural_df['time']
        if method == 'old':
            Fcells = self.neural_df.values.T
            F0 = []
            for cell in range(0, Fcells.shape[0]):
                include_frames = []
                std = np.std(Fcells[cell])
                avg = np.mean(Fcells[cell])
                for frame in range(0, Fcells.shape[1]):
                    if Fcells[cell, frame] < std + avg:
                        include_frames.append(Fcells[cell, frame])
                F0.append(np.mean(include_frames))
            dFF = np.zeros(Fcells.shape)
            for cell in range(0, Fcells.shape[0]):
                for frame in range(0, Fcells.shape[1]):
                    dFF[cell, frame] = (Fcells[cell, frame] - F0[cell]) / F0[cell]
        elif method == 'robust':
            Fcells = self.neural_df.values.T
            dFF = np.zeros(Fcells.shape) # d
            for cell in tqdm(range(Fcells.shape[0])):
                f0_cell = robust_filter(Fcells[cell], method=12, window=200, optimize_window=2, buffer=False)[:, 0]
                dFF[cell] = (Fcells[cell] - f0_cell) / f0_cell
        dff_df = pd.DataFrame(data=dFF.T, columns=[f'neuron{i}' for i in range(Fcells.shape[0])])
        dff_df['time'] = time_axis
        if melt:
            rois = [c for c in dff_df.columns if c != 'time']
            melted = pd.melt(dff_df, id_vars='time', value_vars=rois, var_name='roi', value_name='ZdFF')
            return melted
        else:
            return dff_df

    def plot_cell_location_dFF(self, neuron_range, savefigpath):
        if not os.path.exists(os.path.join(savefigpath,'Cells in the field of view.tiff')):
            import random

            cellstat = []
            for cell in range(0, self.Fraw.shape[0]):
                if self.cells[cell, 0] > 0:
                    cellstat.append(self.stat[cell])

            fluoCellPlot = StartPlots()
            im = np.zeros((256, 256))

            for cell in neuron_range:

                xs = cellstat[cell]['xpix']
                ys = cellstat[cell]['ypix']
                im[ys, xs] = random.random()


            fluoCellPlot.ax.imshow(im, cmap='CMRmap')
            fluoCellPlot.save_plot('Cells in the field of view.tiff', 'tiff', savefigpath)
            #plt.show()
            plt.close()

    def plot_cell_dFF(self, time_range, savefigpath):

        cellsPerPlot = 32
        nCells = self.neural_df.shape[1] - 1
        nFigs = int(np.ceil(nCells/cellsPerPlot))
        plot_ind = np.arange(self.neural_df.shape[0])[np.logical_and(self.neural_df['time'] >= time_range[0],
                                                                     self.neural_df['time'] < time_range[1])]
        for fig in range(nFigs):
            fluoTracePlot = StartPlots()
            neuron_range = np.arange(cellsPerPlot*(fig),cellsPerPlot*(fig+1))
            for cell in neuron_range:
                if cell < nCells:
                    fluoTracePlot.ax.plot(self.neural_df.iloc[plot_ind, cell] + cell,
                                          label="Neuron " + str(cell), linewidth=0.5)
            #plt.show()
            title = 'Cell traces # ' + str(fig) + '.tiff'
            fluoTracePlot.save_plot(title, 'tiff', savefigpath)
            plt.close()

def robust_filter(ys, method=12, window=200, optimize_window=2, buffer=False):
    """
    First 2 * windows re-estimate with mode filter
    To avoid edge effects as beginning, it uses mode filter; better solution: specify initial conditions
    Return:
        dff: np.ndarray (T, 2)
            col0: dff
            col1: boundary scale for noise level
    """
    if method < 10:
        mf, mDC = median_filter(window, method)
    else:
        mf, mDC = std_filter(window, method%10, buffer=buffer)
    opt_w = int(np.rint(optimize_window * window))
    # prepend
    init_win_ys = ys[:opt_w]
    prepend_ys = init_win_ys[opt_w-1:0:-1]
    ys_pp = np.concatenate([prepend_ys, ys])
    f0 = np.array([(mf(ys_pp, i), mDC.get_dev()) for i in range(len(ys_pp))])[opt_w-1:]
    return f0

class fluoAnalysis:

    def __init__(self, beh_file, fluo_file):
        # check file type
        if beh_file.partition('.')[2] == 'csv':
            self.beh = pd.read_csv(beh_file)
        elif beh_file.partition('.')[2] == 'pickle':
            with open(beh_file, 'rb') as pf:
                # Load the data from the pickle file
                beh_data = pickle.load(pf)
                pf.close()
            self.beh = beh_data['behDF']
        self.fluo = pd.read_csv(fluo_file)

    def align_fluo_beh(self, savedatapath):
        # align the fluorescent data to behavior based on trials
        # interpolate fluorescent data and running speed
        alignedFile = os.path.join(savedatapath, 'aligned.pickle')

        if not os.path.exists(alignedFile):# check if aligned data exist already
            nTrials = self.beh.shape[0]
            nCells = self.fluo.shape[1] - 2 # exclude the first column (index) and last column (time)
            startTime = -3
            endTime = 8
            step = 0.01
            interpT = np.arange(startTime, endTime, step)
            self.dFF_aligned = np.full((len(interpT), nTrials, nCells), np.nan)
            self.dFF_original = {}
            self.t_original = {}
            self.run_aligned = np.full((len(interpT), nTrials), np.nan)
            self.interpT = interpT

            for tt in range(nTrials):

            # align speed
                speed = self.beh['running_speed'][tt]
                speed = np.array(speed)
                speed_time = self.beh['running_time'][tt]
                speed_time = np.array(speed_time)
                if speed.size != 0:
                    t = speed_time - self.beh['onset'][tt]
                    y = speed
                    self.run_aligned[:, tt] = np.interp(interpT, t, y)

            # align dFF
                tempStart = self.beh['onset'][tt] + startTime
                tempEnd = self.beh['onset'][tt] + endTime
                t_dFF = self.fluo['time'].values

                for cc in range(nCells):
                    dFF = self.fluo['neuron'+str(cc)].values
                # get the trial timing
                    tempT = t_dFF[np.logical_and(t_dFF>tempStart, t_dFF<=tempEnd)]-self.beh['onset'][tt]
                    tempdFF = dFF[np.logical_and(t_dFF>tempStart, t_dFF<=tempEnd)]
                    if cc==0:
                        self.dFF_original[str(tt)] =  tempdFF[:,np.newaxis]
                    else:
                        self.dFF_original[str(tt)] = np.concatenate((self.dFF_original[str(tt)],
                                                                tempdFF[:,np.newaxis]),1)
                    self.t_original[str(tt)] = tempT
                # interpolate
                    if tempT.size != 0:
                        self.dFF_aligned[:,tt,cc] = np.interp(interpT, tempT, tempdFF)

        # save aligned dFF and running
            saveDict = {'run_aligned':self.run_aligned,
                        'dFF_aligned':self.dFF_aligned,
                        'interpT': self.interpT,
                        'dFF_original': self.dFF_original,
                        't_original': self.t_original}
            with open(alignedFile, 'wb') as handle:
                pickle.dump(saveDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
             with open(alignedFile, 'rb') as handle:
                 saveDict = pickle.load(handle)
             self.run_aligned = saveDict['run_aligned']
             self.dFF_aligned = saveDict['dFF_aligned']
             self.interpT = saveDict['interpT']
             self.dFF_original = saveDict['dFF_original']
             self.t_original = saveDict['t_original']
    def plot_dFF(self, savefigpath):
        # PSTH plot for different trial types
        # cue
        matplotlib.use('Agg')
        nCells = self.fluo.shape[1]-2
        goCue = [1, 2, 3, 4]
        noGoCue = [5, 6, 7, 8]
        probeGoCue = [9, 10, 11, 12]
        probeNoGoCue = [13, 14, 15, 16]

        goTrials = [i for i in range(len(self.beh['sound_num'])) if self.beh['sound_num'][i] in goCue]
        noGoTrials = [i for i in range(len(self.beh['sound_num'])) if self.beh['sound_num'][i] in noGoCue]
        probeGoTrials = [i for i in range(len(self.beh['sound_num'])) if self.beh['sound_num'][i] in probeGoCue]
        probeNoGoTrials = [i for i in range(len(self.beh['sound_num'])) if self.beh['sound_num'][i] in probeNoGoCue]

        for cc in tqdm(range(nCells)):
            figpath = os.path.join(savefigpath, 'cell' + str(cc) + '.tif')
            if not os.path.exists(figpath):
            # get dFF in trials and bootstrap

                dFFPlot = StartSubplots(2,2, ifSharey=True)

            # subplot 1: dFF traces of different cues

    ### plotting PSTH for go/nogo/probe cues--------------------------------------------
                tempGodFF = self.dFF_aligned[:, goTrials, cc]
                bootGo = bootstrap(tempGodFF, 1, 1000)

                tempNoGodFF = self.dFF_aligned[:, noGoTrials, cc]
                bootNoGo = bootstrap(tempNoGodFF, 1, 1000)

                tempProbeGodFF = self.dFF_aligned[:, probeGoTrials, cc]
                bootProbeGo = bootstrap(tempProbeGodFF, 1, 1000)

                tempProbeNoGodFF = self.dFF_aligned[:, probeNoGoTrials, cc]
                bootProbeNoGo = bootstrap(tempProbeNoGodFF, 1, 1000)

                dFFPlot.fig.suptitle('Cell ' + str(cc+1))

                dFFPlot.ax[0,0].plot(self.interpT, bootGo['bootAve'], color=(1,0,0))
                dFFPlot.ax[0,0].fill_between(self.interpT, bootGo['bootLow'], bootGo['bootHigh'], color = (1,0,0),label='_nolegend_', alpha=0.2)

                dFFPlot.ax[0, 0].plot(self.interpT, bootNoGo['bootAve'], color=(0, 1, 0))
                dFFPlot.ax[0, 0].fill_between(self.interpT, bootNoGo['bootLow'], bootNoGo['bootHigh'], color=(0, 1, 0),label='_nolegend_',
                                          alpha=0.2)
                dFFPlot.ax[0,0].legend(['Go', 'No go'])
                dFFPlot.ax[0,0].set_title('Cue')
                dFFPlot.ax[0,0].set_ylabel('dFF')

                dFFPlot.ax[0, 1].plot(self.interpT, bootProbeGo['bootAve'], color=(1, 0, 0))
                dFFPlot.ax[0, 1].fill_between(self.interpT, bootProbeGo['bootLow'], bootProbeGo['bootHigh'], color=(1, 0, 0),label='_nolegend_',
                                          alpha=0.2)

                dFFPlot.ax[0, 1].plot(self.interpT, bootProbeNoGo['bootAve'], color=(0, 1, 0))
                dFFPlot.ax[0, 1].fill_between(self.interpT, bootProbeNoGo['bootLow'], bootProbeNoGo['bootHigh'], color=(0, 1, 0),label='_nolegend_',
                                          alpha=0.2)

                dFFPlot.ax[0, 1].legend(['Probe go', 'Probe no go'])
                dFFPlot.ax[0, 1].set_title('Cue')
                dFFPlot.ax[0, 1].set_ylabel('dFF')

    ### this part is used to plot PSTH for every individual cues
    ### ------------------------------------------------------------------------------
            # cues = np.unique(self.beh['sound_num'])
            #
            # for cue in cues:
            #
            #     tempdFF = self.dFF_aligned[:,self.beh['sound_num']==cue,cc]
            #     bootTemp = bootstrap(tempdFF, 1, 1000)
            #     dFFPlot.fig.suptitle('Cell ' + str(cc+1))
            #
            #     # set color
            #     if cue <= 4:
            #         c = (1, 50*cue/255, 50*cue/255)
            #         subInd = 0
            #     elif cue > 4 and cue<=8:
            #         c = (50*(cue-4)/255, 1, 50*(cue-4)/255)
            #         subInd = 0
            #     else:
            #         c = (25*(cue-8)/255, 25*(cue-8)/255, 1)
            #         subInd = 1
            #
            #     dFFPlot.ax[0,subInd].plot(self.interpT, bootTemp['bootAve'], color=c)
            #     dFFPlot.ax[0,subInd].fill_between(self.interpT, bootTemp['bootLow'], bootTemp['bootHigh'], color = c, alpha=0.2)
            #     dFFPlot.ax[0,subInd].set_title('Cue')
            #     dFFPlot.ax[0,subInd].set_ylabel('dFF')
    ###------------------------------------------------------------------------------------------------------

                Hit_dFF = self.dFF_aligned[:,self.beh['trialType']==2,cc]
                FA_dFF = self.dFF_aligned[:, self.beh['trialType'] == -1, cc]
                Miss_dFF = self.dFF_aligned[:, self.beh['trialType'] == -2, cc]
                CorRej_dFF = self.dFF_aligned[:, self.beh['trialType'] == 0, cc]
                ProbeLick_dFF = self.dFF_aligned[:, self.beh['trialType'] == -3, cc]
                ProbeNoLick_dFF = self.dFF_aligned[:, self.beh['trialType'] == -4, cc]

                Hit_boot = bootstrap(Hit_dFF, 1, 1000)
                FA_boot = bootstrap(FA_dFF, 1, 1000)
                Miss_boot = bootstrap(Miss_dFF, 1, 1000)
                CorRej_boot = bootstrap(CorRej_dFF, 1, 1000)
                ProbeLick_boot = bootstrap(ProbeLick_dFF, 1, 1000)
                ProbeNoLick_boot = bootstrap(ProbeNoLick_dFF, 1, 1000)

            # get cmap
                cmap = matplotlib.colormaps['jet']

                dFFPlot.ax[1, 0].plot(self.interpT, Hit_boot['bootAve'], color = cmap(0.1))
                dFFPlot.ax[1, 0].fill_between(self.interpT, Hit_boot['bootLow'], Hit_boot['bootHigh'],
                                               alpha=0.2, label='_nolegend_', color = cmap(0.1))
                dFFPlot.ax[1, 0].plot(self.interpT, FA_boot['bootAve'], color = cmap(0.3))
                dFFPlot.ax[1, 0].fill_between(self.interpT, FA_boot['bootLow'], FA_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_',color = cmap(0.3))
                dFFPlot.ax[1, 0].plot(self.interpT, Miss_boot['bootAve'], color = cmap(0.5))
                dFFPlot.ax[1, 0].fill_between(self.interpT, Miss_boot['bootLow'], Miss_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_', color = cmap(0.5))
                dFFPlot.ax[1, 0].plot(self.interpT, CorRej_boot['bootAve'], color = cmap(0.7))
                dFFPlot.ax[1, 0].fill_between(self.interpT, CorRej_boot['bootLow'], CorRej_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_', color = cmap(0.7))
                dFFPlot.ax[1, 0].legend(['Hit', 'False alarm','Miss', 'Correct Rejection'])
                dFFPlot.ax[1, 0].set_title('Cue')
                dFFPlot.ax[1, 0].set_ylabel('dFF')

                dFFPlot.ax[1, 1].plot(self.interpT, ProbeLick_boot['bootAve'], color = cmap(0.25))
                dFFPlot.ax[1, 1].fill_between(self.interpT, ProbeLick_boot['bootLow'], ProbeLick_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_', color = cmap(0.25))
                dFFPlot.ax[1, 1].plot(self.interpT, ProbeNoLick_boot['bootAve'], color = cmap(0.75))
                dFFPlot.ax[1, 1].fill_between(self.interpT, ProbeNoLick_boot['bootLow'], ProbeNoLick_boot['bootHigh'],
                                          alpha=0.2, label='_nolegend_', color = cmap(0.75))
                dFFPlot.ax[1, 1].legend(['Probe lick', 'Probe no lick'])
                dFFPlot.ax[1, 1].set_title('Cue')
                dFFPlot.ax[1, 1].set_ylabel('dFF')

                dFFPlot.fig.set_size_inches(14, 10, forward=True)

            #plt.show()

            # save file
                dFFPlot.save_plot('cell' + str(cc) + '.tif', 'tif', savefigpath)
                plt.close()

            # plot  dFF with running?

    def plot_dFF_singleCell(self, cellID, trials):
        # plot the average dFF curve of a given cell for hit/FA/Hit/Miss trials

        # get the trials
        dFFCellPlot = StartPlots()

        dFF = self.dFF_aligned[:,:,cellID]
        Ind = 0
        for trial in trials:
            if self.beh['sound_num'][trial] in [1, 2, 3, 4]:
                c = (1, 0, 0)
            elif self.beh['sound_num'][trial] in [5, 6, 7, 8]:
                c = (0, 1, 0)
            else:
                c = (0, 0, 1)

            dFFCellPlot.ax.plot(self.interpT, dFF[:,trial]+Ind*1, color=c)
            Ind = Ind + 1

        plt.show()

    def process_X(self, regr_time, choiceList, rewardList, nTrials, nCells, trial):
        # re-arrange the behavior and dFF data for linear regression
        X = np.zeros((14,len(regr_time)))
        #
        Y = np.zeros((nCells, len(regr_time)))

        # need to determine whether to use exact frequency or go/no go/probe
        # separate into go/no go trials (for probe trials: 9-12: go; 13-16 no go
        # previous + next stimulus
        go_stim = [1,2,3,4,9,10,11,12]
        nogo_stim = [5,6,7,8,13,14,15,16]

        X[1, :] = np.ones(len(regr_time)) * [1 if self.beh['sound_num'][trial] in go_stim else 0]
        if trial == 0:
            X[2,:] = np.ones(len(regr_time)) * np.nan
        else:
            X[2, :] = np.ones(len(regr_time)) * [1 if self.beh['sound_num'][trial-1] in go_stim else 0]
        if trial == nTrials-1:
            X[0, :] = np.ones(len(regr_time)) * np.nan
        else:
            X[0, :] = np.ones(len(regr_time)) * [1 if self.beh['sound_num'][trial+1] in go_stim else 0]

        # choice: lick = 1; no lick = -1
        X[3, :] = np.ones(len(regr_time)) * (
            choiceList[trial + 1] if trial < nTrials - 1 else np.nan)
        X[4, :] = np.ones(len(regr_time)) * (choiceList[trial])
        X[5, :] = np.ones(len(regr_time)) * (choiceList[trial - 1] if trial > 0 else np.nan)

        # reward
        X[6, :] = np.ones(len(regr_time)) * (
            rewardList[trial + 1] if trial < nTrials - 1 else np.nan)
        X[7, :] = np.ones(len(regr_time)) * rewardList[trial]
        X[8, :] = np.ones(len(regr_time)) * (rewardList[trial - 1] if trial > 0 else np.nan)

        # interaction
        X[9, :] = X[3, :] * X[6, :]
        X[10, :] = X[4, :] * X[7, :]
        X[11, :] = X[5, :] * X[8, :]

        # running speed and licks
        tStep = np.nanmean(np.diff(regr_time))
        licks = self.beh['licks'][trial]

        for tt in range(len(regr_time)):
            t_start = regr_time[tt] - tStep / 2
            t_end = regr_time[tt] + tStep / 2
            temp_run = self.run_aligned[:, trial]
            X[12, tt] = np.nanmean(
                temp_run[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])

            X[13, tt] = np.sum(np.logical_and(licks>=t_start+self.beh['onset'][trial],
                                              licks<t_end+self.beh['onset'][trial]))

            # dependent variable: dFF
            for cc in range(nCells):
                temp_dFF = self.dFF_aligned[:, trial, cc]
                Y[cc, tt] = np.nanmean(
                    temp_dFF[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])

        return X, Y

    def linear_model(self, n_predictors):
        # arrange the independent variables and dependent variables for later linear regression
        # model:
        # y = b0 + b1*cue + b2* cn+1 + b3* cn + b4* cn-1 + b5* rn+1 + b6*rn + b7*rn-1 + b8* cn+1*rn+1 + b9* cn*rn + b10* cn-1*rn-1 + b11* running_speed

        tStart = -1.95
        tEnd = 1.95
        tStep = 0.1
        regr_time = np.arange(tStart, tEnd, tStep)
        nTrials = self.beh.shape[0]
        nCells = self.fluo.shape[1]-2
        independent_X = np.zeros((n_predictors, nTrials, len(regr_time)))
        dFF_Y = np.zeros((nCells, nTrials, len(regr_time)))

        choiceList = [1 if lick > 0 else 0 for lick in self.beh['licks_out']]
        rewardList = self.beh['reward']

        # parallel computing
        n_jobs = -1

        # Parallelize the loop over `trial`
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(self.process_X)(regr_time, choiceList, rewardList, nTrials, nCells, trial) for trial in tqdm(range(nTrials)))
        #dFF_Y = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(self.process_Y)(regr_time, nCells, trial) for trial in tqdm(range(nTrials)))

        # unpack the result of parallel computing
        for tt in range(nTrials):
            independent_X[:,tt,:], dFF_Y[:,tt,:] = results[tt]

        return np.array(independent_X), np.array(dFF_Y), regr_time

    def linear_regr(self, X, y, regr_time, saveDataFile):
        # try logistic regression for cues?
        """
        x： independent variables
        Y: dependent variables
        regr_t: time relative to cue onset
        output:
        coefficients, p-value
        R-square
        amount of variation explained
        """
        # linear regression model
        # cue, choice, reward, running speed,
        # cue (16 categorical variables)
        # choice: n-1, n, n+1 (n+1 should be a control since the choice depend on the cues)
        # reward: n-1, n, n+1 (n+1 control)
        # choice x reward： n-1, n, n+1
        # running speed

        # auto-correlation: run MLR first, then check residuals
        # reference: https://stats.stackexchange.com/questions/319296/model-for-regression-when-independent-variable-is-auto-correlated
        # fit the regression model
        nCells = self.fluo.shape[1] - 2
        n_predictors = X.shape[0]
        MLRResult = {'coeff': np.zeros((n_predictors, len(regr_time), nCells)), 'pval': np.zeros((n_predictors, len(regr_time), nCells)), 'r2': np.zeros((len(regr_time), nCells))}

        n_jobs = -1

        # find the trials without fluorescent data
        notnany_where = np.argwhere(~np.isnan(y))
        nanx_where = np.argwhere(np.isnan(X))
        if not len(nanx_where)==0:
            notnan_trials = int(np.setdiff1d(np.unique(notnany_where[:,1]), np.unique(nanx_where[:,1])))
        else:
            notnan_trials = np.unique(notnany_where[:,1])
        # Parallelize the loop over timestep
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(self.run_MLR)(X[:,notnan_trials,tt],
                    y[:,notnan_trials,tt]) for tt in
                    tqdm(range(len(regr_time))))

        for tt in range(len(regr_time)):
            MLRResult['coeff'][:,tt,:], MLRResult['pval'][:,tt,:], MLRResult['r2'][tt,:] = results[tt]

        MLRResult['regr_time'] = regr_time

        # determine significant cells (for current stimulus, choice, outcome)
        # criteria: 3 consecutive significant time points, or 10 total significant time points
        sigCells = {}
        sigCells['stimulus'] = []
        sigCells['choice'] = []
        sigCells['outcome'] = []
        pThresh = 0.01
        for cc in range(nCells):
            stiPVal = MLRResult['pval'][1,:,cc]
            choicePVal = MLRResult['pval'][4,:,cc]
            outcomePVal = MLRResult['pval'][7,:,cc]
            if sum(stiPVal<0.01) >= 10 or count_consecutive(stiPVal<0.01)>=3:
                sigCells['stimulus'].append(cc)
            if sum(choicePVal[regr_time>0] < 0.01) >= 10 or count_consecutive(choicePVal[regr_time>0] < 0.01) >= 3:
                sigCells['choice'].append(cc)
            if sum(outcomePVal[regr_time>0] < 0.01) >= 10 or count_consecutive(outcomePVal[regr_time>0] < 0.01) >= 3:
                sigCells['outcome'].append(cc)

        MLRResult['sigCells'] = sigCells
        with open(saveDataFile, 'wb') as pf:
            pickle.dump(MLRResult, pf, protocol=pickle.HIGHEST_PROTOCOL)
            pf.close()
        return MLRResult

    def plotMLRResult(self, MLRResult, labels, neuronRaw, saveFigPath):
        # get the average coefficient plot and fraction of significant neurons
        matplotlib.use('Agg')

        varList =labels
        # average coefficient
        nPredictors = MLRResult['coeff'].shape[0]

        coeffPlot = StartSubplots(4,4, ifSharey=True)

        maxY = 0
        minY = 0
        for n in range(nPredictors):
            tempBoot = bootstrap(MLRResult['coeff'][n,:,:],1, 1000)
            tempMax = max(tempBoot['bootHigh'])
            tempMin = min(tempBoot['bootLow'])
            if tempMax > maxY:
                maxY = tempMax
            if tempMin < minY:
                minY = tempMin
            coeffPlot.ax[n//4, n%4].plot(MLRResult['regr_time'], tempBoot['bootAve'], c =(0,0,0))
            coeffPlot.ax[n // 4, n % 4].fill_between(MLRResult['regr_time'], tempBoot['bootLow'], tempBoot['bootHigh'],
                                          alpha=0.2,  color = (0.7,0.7,0.7))
            coeffPlot.ax[n//4, n%4].set_title(varList[n])
        plt.ylim((minY,maxY))
        #plt.show()
        coeffPlot.save_plot('Average coefficient.tif','tiff', saveFigPath)
        #plt.close()
        # fraction of significant neurons
        sigPlot = StartSubplots(4, 4, ifSharey=True)
        pThresh = 0.001
        nCell = MLRResult['coeff'].shape[2]

        # binomial test to determine signficance

        for n in range(nPredictors):
            fracSig = np.sum(MLRResult['pval'][n, :, :]<pThresh,1)/nCell
            pResults = [binomtest(x,nCell,p=pThresh).pvalue for x in np.sum(MLRResult['pval'][n, :, :]<pThresh,1)]
            sigPlot.ax[n // 4, n % 4].plot(MLRResult['regr_time'], fracSig, c=(0, 0, 0))
            sigPlot.ax[n // 4, n % 4].set_title(varList[n])

            if n//4 == 0:
                sigPlot.ax[n // 4, n % 4].set_ylabel('Fraction of sig')
            if n > 8:
                sigPlot.ax[n // 4, n % 4].set_xlabel('Time from cue (s)')
            # plot the signifcance bar
            dt = np.mean(np.diff(MLRResult['regr_time']))
            for tt in range(len(MLRResult['regr_time'])):
                if pResults[tt]<0.05:
                    sigPlot.ax[n//4, n%4].plot(MLRResult['regr_time'][tt]+dt*np.array([-0.5,0.5]), [0.5,0.5],color=(255/255, 189/255, 53/255), linewidth = 5)
        plt.ylim((0,0.6))
        #plt.show()
        sigPlot.save_plot('Fraction of significant neurons.tif', 'tiff', saveFigPath)
        #plt.close()
        # plot r-square
        r2Boot = bootstrap(MLRResult['r2'], 1, 1000)
        r2Plot = StartPlots()
        r2Plot.ax.plot(MLRResult['regr_time'], r2Boot['bootAve'],c=(0, 0, 0))
        r2Plot.ax.fill_between(MLRResult['regr_time'], r2Boot['bootLow'], r2Boot['bootHigh'],
                                                 color=(0.7, 0.7, 0.7))
        r2Plot.ax.set_title('R-square')
        r2Plot.save_plot('R-square.tif', 'tiff', saveFigPath)
        #plt.close()

        """plot significant neurons"""
        sigCells = MLRResult['sigCells']
        cellstat = []
        for cell in range(neuronRaw.Fraw.shape[0]):
            if neuronRaw.cells[cell, 0] > 0:
                cellstat.append(neuronRaw.stat[cell])

        fluoCellPlot = StartPlots()
        im = np.zeros((256, 256,3))

        #for cell in range(decode_results[var]['importance'].shape[0]):
        for cell in range(len(cellstat)):
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
            if cell not in \
                    set(sigCells['choice'])|set(sigCells['outcome'])|set(sigCells['stimulus']):
                im[ys, xs] = [0.7, 0.7, 0.7]

        for cell in sigCells['choice']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys,xs] = [0,0,0]
            im[ys, xs] = np.add(im[ys, xs], [1.0, 0.0, 0.0])
        for cell in sigCells['outcome']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys, xs] = [0, 0, 0]
            im[ys,xs] = np.add(im[ys,xs],[0.0,1.0,0.0])
        for cell in sigCells['stimulus']:
            xs = cellstat[cell]['xpix']
            ys = cellstat[cell]['ypix']
                #im[ys, xs] = [0, 0, 0]
            im[ys,xs] = np.add(im[ys,xs],[0.0,0.0,1.0])
        action_patch = mpatches.Patch(color=(1,0,0), label='Action')
        outcome_patch = mpatches.Patch(color=(0,1,0), label = 'Outcome')
        stimulus_patch = mpatches.Patch(color=(0, 0, 1), label='Stimulus')
        # Create a custom legend with the green patch
        plt.legend(handles=[action_patch, outcome_patch, stimulus_patch],loc='center left',bbox_to_anchor=(1, 0.5))
        fluoCellPlot.ax.imshow(im, cmap='CMRmap')
        #plt.show()

        fluoCellPlot.save_plot('Regression neuron coordinates.tiff', 'tiff', saveFigPath)
        #plt.close()
    def run_MLR(self, x, y):
        # running individual MLR for parallel computing
        nCells = y.shape[0]
        n_predictors = x.shape[0]
        coeff = np.zeros((n_predictors, nCells))
        pval = np.zeros((n_predictors, nCells))
        rSquare = np.zeros((nCells))

        x = sm.add_constant(np.transpose(x))
        for cc in range(nCells):
            model = sm.OLS(y[cc,:], x[:,:]).fit()
            coeff[:,cc] = model.params[1:]
            pval[:,cc] = model.pvalues[1:]
            rSquare[cc] = model.rsquared

        return coeff, pval, rSquare

    def decoding_old(self, signal, decodeVar, varList, trialMask, classifier, regr_time):
        """
        function to decode behavior from neural activity
        running speed/reward/action/stimulus
        signal: neural signal. n x T
        var: variable to be decoded
        trialMask: trials to consider with specified conditions (Hit/FA etc.)
        """

        # run decoding for every time bin
        trialInd = np.arange(signal.shape[1])
        nullRepeats = 20
        decode_perform = {}
        #decode_perform['action'] = np.zeros(len(regr_time))
        #decode_perform['stimulus'] = np.zeros(len(regr_time))
        decode_null = {}
        #decode_null['action'] = np.zeros((len(regr_time), nullRepeats))
        #decode_null['stimulus'] = np.zeros((len(regr_time), nullRepeats))

        for varname in varList:
            decode_perform[varname] =  np.zeros(len(regr_time))
            decode_null[varname] = np.zeros((len(regr_time), nullRepeats))
            n_jobs = -1

        # Parallelize the loop over `trial`
            results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                delayed(self.run_decoder)(
                    signal, decodeVar,varname,trialInd,trialMask,
                    nullRepeats, classifier, idx) for idx in
                    tqdm(range(len(regr_time))))

            for tt in range(len(regr_time)):
                t1,t2=results[tt]
                decode_perform[varname][tt] = t1[varname]
                decode_null[varname][tt,:] = t2[varname]

        return decode_perform, decode_null

    # define the function for parallel computing
    def run_decoder_old(self, signal,decodeVar,varname,trialInd, trialMask,
                    nullRepeats, classifier, idx):
        # function for parallel computing
        if varname == 'action':
            data = {
                'raster': signal[:, trialMask, idx].transpose(),
                'action': decodeVar['action'][trialMask],
                #varname:decodeVar[varname][trialMask],
                #'stimulus': decodeVar['stimulus'][trialMask]
                'trial': trialInd[trialMask],
            }
            conditions = {
                'action': [1, 0],
            }
        elif varname == 'stimulus':
            data = {
                'raster': signal[:, trialMask, idx].transpose(),
                'stimulus': decodeVar['stimulus'][trialMask],
                'trial': trialInd[trialMask],
            }
            conditions = {
                'stimulus': {
                    'go': lambda d: d['stimulus'] <= 4,
                    'nogo': lambda d: (d['stimulus'] >= 5) & (d['stimulus'] <= 8)
                }  # this should be stimulus tested in the session
            }

        dec = Decodanda(
            data=data,
            conditions=conditions,
            classifier=classifier
        )

        performance, null = dec.decode(
            training_fraction=0.5,  # fraction of trials used for training
            cross_validations=10,  # number of cross validation folds
            nshuffles=nullRepeats)

        return performance, null

    def decoding(self, decodeSig, decodeVar, varList, trialMask, subTrialMask, classifier, regr_time, saveDataPath):

        # check if results already exist
       # if not os.path.exists(saveDataPath):

        decode_results = {}
        nCells = decodeSig.shape[0]
        nRepeats = 10 # repeat 10 times to get an average

        for varname in varList:
            print("Decoding " + varname)
            decode_results[varname] = {}
            decode_results[varname]['classifier'] = [[] for xx in range(nRepeats)]
            decode_results[varname]['classifier_shuffle'] = [[] for xx in range(nRepeats)]
            decode_results[varname]['ctrl_accuracy'] = np.zeros((len(regr_time),nRepeats))
            #decode_results[varname]['ctrl_recall'] = np.zeros((len(regr_time)))
            decode_results[varname]['ctrl_f1_score'] = np.zeros((len(regr_time),nRepeats))
            #decode_results[varname]['ctrl_precision'] = np.zeros((len(regr_time)))
            #decode_results[varname]['recall'] = np.zeros(len(regr_time))
            #decode_results[varname]['precision'] = np.zeros(len(regr_time))
            decode_results[varname]['f1_score'] = np.zeros((len(regr_time),nRepeats))
            decode_results[varname]['accuracy'] = np.zeros((len(regr_time), nRepeats))
            decode_results[varname]['params'] = {}
            decode_results[varname]['params']['n_estimators'] = np.zeros((len(regr_time),nRepeats))
            decode_results[varname]['params']['max_depth'] = np.zeros((len(regr_time),nRepeats))
            decode_results[varname]['params']['min_samples_leaf'] = np.zeros((len(regr_time),nRepeats))
            decode_results[varname]['prediction_accuracy'] = {}
            #decode_results[varname]['prediction_f1_score'] = {}
            decode_results[varname]['prediction_accuracy_ctrl'] = {}
            #decode_results[varname]['prediction_f1_score_ctrl'] = {}

            #first_not_nan_trial = np.min(np.where(~np.isnan(decodeSig[0, :, 0])))
            #last_not_nan_trial = np.max(np.where(~np.isnan(decodeSig[0, :, 0])))

            notnany_where = np.argwhere(~np.isnan(decodeSig[0, :, 0]))
            nanx_where = np.argwhere(np.isnan(decodeVar[varname]))
            if not len(nanx_where) == 0:
                notnan_trials = np.setdiff1d(np.unique(notnany_where), np.unique(nanx_where))
            else:
                notnan_trials = np.unique(notnany_where)

            # exclude probe trials

            for repeat in range(nRepeats):
                if classifier == 'RandomForest':  # use parallel computing for Random forest
                    decode_results[varname]['confidence'] = np.zeros((len(notnan_trials),
                                                                      len(regr_time)))
                    n_jobs = -1


                # Parallelize the loop over `trial`
                    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                        delayed(self.run_decoder)(
                        decodeSig[:,notnan_trials,idx].transpose(),
                        decodeVar[varname][notnan_trials],notnan_trials,
                        trialMask[notnan_trials],
                        classifier) for idx in
                        tqdm(range(len(regr_time))))

                #for tt in range(len(regr_time)):
                # unpacking results
                    for rr in range(len(results)):
                        tempResults = results[rr]
                    # load results
                        #decode_results[varname]['ctrl_precision'][rr] = tempResults['ctrl_precision']
                        #decode_results[varname]['ctrl_recall'][rr] = tempResults['ctrl_recall']
                        decode_results[varname]['ctrl_f1_score'][rr] = tempResults['ctrl_f1_score']
                        decode_results[varname]['ctrl_accuracy'][rr] = tempResults['ctrl_accuracy']
                        decode_results[varname]['accuracy'][rr] = tempResults['accuracy']
                        decode_results[varname]['importance'][:,rr] = tempResults['importance']
                        #decode_results[varname]['recall'][rr] = tempResults['recall']
                        #decode_results[varname]['precision'][rr] = tempResults['precision']
                        decode_results[varname]['f1_score'][rr] = tempResults['f1_score']
                        decode_results[varname]['classifier'].append(tempResults['classifier'])
                        decode_results[varname]['classifier_shuffle'].append(tempResults['classifier_shuffle'])
                        if rr==0:
                            decode_results[varname]['test_trials'] = np.zeros((len(tempResults['test_trials']),len(regr_time)))
                        decode_results[varname]['test_trials'][:,rr] = tempResults['test_trials']
                        decode_results[varname]['params']['n_estimators'][rr] = tempResults['params']['n_estimators']
                        decode_results[varname]['params']['max_depth'][rr] = tempResults['params']['max_depth']
                        decode_results[varname]['params']['min_samples_leaf'][rr] = tempResults['params']['min_samples_leaf']
                        decode_results[varname]['confidence'][:,rr] = tempResults['confidence']

                elif classifier=='SVC':
                    for rr in range(len(regr_time)):
                        result = self.run_decoder(
                                decodeSig[:, notnan_trials, rr].transpose(),
                                decodeVar[varname][notnan_trials], notnan_trials,
                                trialMask[notnan_trials],
                                classifier, rand_seed = repeat,decoding_type='reg')
                        #decode_results[varname]['ctrl_precision'][rr] = result['ctrl_precision']
                        #decode_results[varname]['ctrl_recall'][rr] = result['ctrl_recall']
                        decode_results[varname]['ctrl_f1_score'][rr, repeat] = result['ctrl_f1_score']
                        decode_results[varname]['ctrl_accuracy'][rr, repeat] = result['ctrl_accuracy']
                        decode_results[varname]['accuracy'][rr, repeat] = result['accuracy']
                        # if varname == 'outcome':
                        #     decode_results[varname]['importance'][:,:,rr] = result['importance']
                        # else:
                        #     decode_results[varname]['importance'][:, rr] = result['importance']
                        #decode_results[varname]['recall'][rr] = result['recall']
                        #decode_results[varname]['precision'][rr] = result['precision']
                        decode_results[varname]['f1_score'][rr, repeat] = result['f1_score']
                        decode_results[varname]['classifier'][repeat].append(result['classifier'])
                        decode_results[varname]['classifier_shuffle'][repeat].append(result['classifier_shuffle'])
                        if rr==0 and repeat == 0:
                            decode_results[varname]['test_trials'] = np.zeros((len(result['test_trials']),len(regr_time),nRepeats))
                        decode_results[varname]['test_trials'][:,rr,repeat] = result['test_trials']

                    # ensemble size analysis

                trial_types = ['Hit',
                               'FA',
                               'CorRej',
                               'Probe']
                for key in trial_types:
                    if repeat == 0:
                        decode_results[varname]['prediction_accuracy'][key] = np.zeros((len(regr_time),
                                                                                        nRepeats))
                        #decode_results[varname]['prediction_f1_score'][key] = np.zeros(len(regr_time))
                        decode_results[varname]['prediction_accuracy_ctrl'][key] = np.zeros((len(regr_time),
                                                                                             nRepeats))
                    #decode_results[varname]['prediction_f1_score_ctrl'][key] = np.zeros(len(regr_time))
                    # in probe trials, 2 -> 0; 3 -> 1

                    # from test trials, determine there trial type
                    # based on trial types, determine the trial type mask
                    # looking at FA_cue7 with prior reward/noreward specificly
                    test_trials_ori = [notnan_trials[int(tt)] for tt in decode_results[varname]['test_trials'][:,0,repeat]] # test trials in original index
                    # they are notnan_trials
                    # Outcome = [r if r == 1 else 0 for r in self.beh['reward']]
                    # preOutcome = [np.nan] + Outcome[0:-1]

                    if key == 'Hit':
                        test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == 2]
                    elif key == 'FA':
                        test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == -1]
                    elif key == 'CorRej':
                        test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == 0]
                    # elif key == 'CorRej_R':
                    #     test_trial = [tt for tt in test_trials_ori if
                    #                   (self.beh['trialType'][tt] == 0 and preOutcome[tt] == 1)]
                    # elif key == 'CorRej_NR':
                    #     test_trial = [tt for tt in test_trials_ori if
                    #                   (self.beh['trialType'][tt] == 0 and preOutcome[tt] == 0)]
                    elif key == 'Probe':
                        test_trial = [tt for tt in range(len(self.beh['trialType'])) if
                                      (self.beh['trialType'][tt] == -3 or self.beh['trialType'][tt] == -4)]
                    testTrialMask = np.array([False for tt in range(len(self.beh['trialType']))])
                    testTrialMask[test_trial] = True

                    # don't do parallel
                    for rr in range(len(regr_time)):
                        result = self.run_decoder_trained_model(decode_results[varname]['classifier'][repeat][rr],
                                                                decode_results[varname]['classifier_shuffle'][repeat][rr],
                                                                decodeSig[:, notnan_trials, rr].transpose(),
                                                                decodeVar[varname][notnan_trials],
                                                                testTrialMask[notnan_trials])
                        decode_results[varname]['prediction_accuracy'][key][rr,repeat] = result['accuracy']
                        #decode_results[varname]['prediction_f1_score'][key][rr] = result['f1_score']
                        decode_results[varname]['prediction_accuracy_ctrl'][key][rr,repeat] = result['accuracy_ctrl']
                        #decode_results[varname]['prediction_f1_score_ctrl'][key][rr] = result['f1_score_ctrl']


        # ensemble analysis
            if varname == 'stimulus':
                trial_types = ['Hit',
                               'FA',
                               'CorRej',
                               'Probe']
                nNeurons = np.arange(10, nCells, 10)
                cellCounter = np.arange(nCells)
                nRepeats =20  # repeat 20 times to get an average decoding accuracy
                decode_ensembleSize = {}
                decode_ensembleSize['accuracy'] = np.zeros((len(regr_time), len(nNeurons)))
                decode_ensembleSize['ctrl_accuracy'] = np.zeros((len(regr_time),
                                                                 len(nNeurons)))
                decode_ensembleSize['prediction_accuracy'] = {}
                decode_ensembleSize['prediction_accuracy_ctrl'] = {}
                for key in trial_types:
                    decode_ensembleSize['prediction_accuracy'][key] = np.zeros((len(regr_time), len(nNeurons)))
                    decode_ensembleSize['prediction_accuracy_ctrl'][key] = np.zeros((len(regr_time), len(nNeurons)))

                for idx, nN in tqdm(enumerate(nNeurons)):
                    tempDecode = {}
                    tempDecode['accuracy'] = np.zeros((len(regr_time), nRepeats))
                    tempDecode['ctrl_accuracy'] = np.zeros((len(regr_time), nRepeats))
                    tempDecode['prediction_accuracy'] = {}
                    tempDecode['prediction_accuracy_ctrl'] = {}
                    tempDecode['classifier'] = [[] for nn in range(nRepeats)]
                    tempDecode['classifier_shuffle'] = [[] for nn in range(nRepeats)]
                    for nR in range(nRepeats):
                        # randomly picking neurons
                        nPick = np.random.choice(cellCounter, size=nN, replace=False)
                        for rr in range(len(regr_time)):
                            temp = decodeSig[nPick, :, rr]
                            result = self.run_decoder(
                                temp[:, notnan_trials].transpose(),
                                decodeVar[varname][notnan_trials], notnan_trials,
                                trialMask[notnan_trials],
                                classifier, rand_seed = nR, decoding_type='reg')
                            tempDecode['ctrl_accuracy'][rr, nR] = result['ctrl_accuracy']
                            tempDecode['accuracy'][rr, nR] = result['accuracy']
                            #tempDecode['classifier'][nR].append(result['classifier'])
                            #tempDecode['classifier_shuffle'][nR].append(result['classifier_shuffle'])
                            if rr == 0 and nR == 0:
                                tempDecode['test_trials'] = np.zeros(
                                    (len(result['test_trials']), len(regr_time), nRepeats))
                            #tempDecode['test_trials'][:, rr, nR] = result['test_trials']
                        # get the decoding accuracy for different trialtypes

                            for key in trial_types:
                                if nR == 0:
                                    tempDecode['prediction_accuracy'][key] = np.zeros((len(regr_time),
                                                                                       nRepeats))
                                    tempDecode['prediction_accuracy_ctrl'][key] = np.zeros((len(regr_time),
                                                                                            nRepeats))
                                test_trials_ori = [notnan_trials[int(tt)] for tt in
                                                    result['test_trials']] # test trials in original index

                                if key == 'Hit':
                                    test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == 2]
                                elif key == 'FA':
                                    test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == -1]
                                elif key == 'CorRej':
                                    test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == 0]
                                elif key == 'Probe':
                                    test_trial = [tt for tt in range(len(self.beh['trialType'])) if
                                              (self.beh['trialType'][tt] == -3 or self.beh['trialType'][tt] == -4)]
                                testTrialMask = np.array([False for tt in range(len(self.beh['trialType']))])
                                testTrialMask[test_trial] = True
                        # don't do parallel

                                result_trial = self.run_decoder_trained_model(result['classifier'],
                                                                    result['classifier_shuffle'],
                                                                    temp[:, notnan_trials].transpose(),
                                                                    decodeVar[varname][notnan_trials],
                                                                    testTrialMask[notnan_trials])
                                tempDecode['prediction_accuracy'][key][rr, nR] = result_trial['accuracy']
                            # decode_results[varname]['prediction_f1_score'][key][rr] = result['f1_score']
                                tempDecode['prediction_accuracy_ctrl'][key][rr, nR] = result_trial[
                                    'accuracy_ctrl']

                    decode_ensembleSize['accuracy'][:, idx] = np.nanmean(
                        tempDecode['accuracy'], 1)
                    decode_ensembleSize['ctrl_accuracy'][:, idx] = np.nanmean(
                        tempDecode['ctrl_accuracy'], 1)
                    for key in trial_types:
                        decode_ensembleSize['prediction_accuracy'][key][:,idx] = np.nanmean(
                            tempDecode['prediction_accuracy'][key],1)
                        decode_ensembleSize['prediction_accuracy_ctrl'][key][:,idx] = np.nanmean(
                            tempDecode['prediction_accuracy_ctrl'][key],1)

                decode_ensembleSize['nNeurons'] = nNeurons
                decode_ensembleSize['regr_time'] = regr_time
                decode_results[varname]['decode_realSize'] = decode_ensembleSize


        # save the decoding results
        for key in varList:
            del decode_results[key]['classifier'],decode_results[key]['classifier_shuffle'],\
                decode_results[key]['test_trials']
            for value in decode_results[key].keys():
                if value=='prediction_accuracy' or value=='prediction_accuracy_ctrl':
                    for vv in decode_results[key][value].keys():
                        decode_results[key][value][vv] = np.nanmean(decode_results[key][value][vv],1)
                elif value != 'decode_realSize' and value != 'params':
                    decode_results[key][value] = np.nanmean(decode_results[key][value],1)
        decode_results['time'] = regr_time
        decode_results['var'] = varList

        with open(saveDataPath, 'wb') as pf:
            pickle.dump(decode_results, pf, protocol=pickle.HIGHEST_PROTOCOL)
            pf.close()
            #return decode_perform, decode_null


    def decoding_hardeasy(self, decodeSig, decodeVar, trialMask, classifier, regr_time, saveDataPath):
        decode_results= {}

        nCells = decodeSig.shape[0]
        nRepeats = 10  # repeat 10 times to get an average

        for key in ['go', 'nogo']:
            decode_results[key] = {}
            decode_results[key]['ctrl_accuracy'] = np.zeros((len(regr_time), nRepeats))
            decode_results[key]['accuracy'] = np.zeros((len(regr_time), nRepeats))
        # decode_results[varname]['prediction_f1_score_ctrl'] = {}

        # only keep trials with stimulus 1, 4, 5, 8
        tempDecode= {}
        tempDecode['go'] =[]
        tempDecode['nogo'] = []
        for idx, v in enumerate(decodeVar['stimulus']):
            if v in [1, 4, 5, 8]:
                if v in [1,4]:
                    tempDecode['go'].append(v)
                    tempDecode['nogo'].append(np.nan)
                else:
                    tempDecode['nogo'].append(v)
                    tempDecode['go'].append(np.nan)
            else:
                tempDecode['go'].append(np.nan)
                tempDecode['nogo'].append(np.nan)

        tempDecode['nogo'] = np.array(tempDecode['nogo'])
        tempDecode['go'] = np.array(tempDecode['go'])

        notnany_where = np.argwhere(~np.isnan(decodeSig[0, :, 0]))
        notnan_trials = {}
        for key in ['go', 'nogo']:
            nanx_where= np.argwhere(np.isnan(tempDecode[key]))
            if not len(nanx_where) == 0:
                notnan_trials[key] = np.setdiff1d(np.unique(notnany_where), np.unique(nanx_where))
            else:
                notnan_trials[key] = np.unique(notnany_where)

        # exclude probe trials
        decoding_type = 'hardeasy'
        for repeat in range(nRepeats):
            if classifier == 'RandomForest':  # use parallel computing for Random forest
                decode_results[varname]['confidence'] = np.zeros((len(notnan_trials),
                                                                  len(regr_time)))
                n_jobs = -1

                # Parallelize the loop over `trial`
                results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
                    delayed(self.run_decoder)(
                        decodeSig[:, notnan_trials, idx].transpose(),
                        decodeVar[varname][notnan_trials], notnan_trials,
                        trialMask[notnan_trials],
                        classifier) for idx in
                    tqdm(range(len(regr_time))))

                # for tt in range(len(regr_time)):
                # unpacking results
                for rr in range(len(results)):
                    tempResults = results[rr]
                    # load results
                    # decode_results[varname]['ctrl_precision'][rr] = tempResults['ctrl_precision']
                    # decode_results[varname]['ctrl_recall'][rr] = tempResults['ctrl_recall']
                    decode_results[varname]['ctrl_f1_score'][rr] = tempResults['ctrl_f1_score']
                    decode_results[varname]['ctrl_accuracy'][rr] = tempResults['ctrl_accuracy']
                    decode_results[varname]['accuracy'][rr] = tempResults['accuracy']
                    decode_results[varname]['importance'][:, rr] = tempResults['importance']
                    # decode_results[varname]['recall'][rr] = tempResults['recall']
                    # decode_results[varname]['precision'][rr] = tempResults['precision']
                    decode_results[varname]['f1_score'][rr] = tempResults['f1_score']
                    decode_results[varname]['classifier'].append(tempResults['classifier'])
                    decode_results[varname]['classifier_shuffle'].append(tempResults['classifier_shuffle'])
                    if rr == 0:
                        decode_results[varname]['test_trials'] = np.zeros(
                            (len(tempResults['test_trials']), len(regr_time)))
                    decode_results[varname]['test_trials'][:, rr] = tempResults['test_trials']
                    decode_results[varname]['params']['n_estimators'][rr] = tempResults['params']['n_estimators']
                    decode_results[varname]['params']['max_depth'][rr] = tempResults['params']['max_depth']
                    decode_results[varname]['params']['min_samples_leaf'][rr] = tempResults['params'][
                        'min_samples_leaf']
                    decode_results[varname]['confidence'][:, rr] = tempResults['confidence']

            elif classifier == 'SVC':
                for rr in range(len(regr_time)):
                    for key in ['go', 'nogo']:
                        trials = notnan_trials[key]
                        # check if hard and easy trials both present
                        if len(np.unique(tempDecode[key][trials]))>=2: # both trials present
                            result = self.run_decoder(
                                decodeSig[:, trials , rr].transpose(),
                                tempDecode[key][trials], trials ,
                                trialMask[trials ],
                                classifier, rand_seed=repeat, decoding_type=decoding_type)
                    # decode_results[varname]['ctrl_precision'][rr] = result['ctrl_precision']
                    # decode_results[varname]['ctrl_recall'][rr] = result['ctrl_recall']
                            decode_results[key]['ctrl_accuracy'][rr, repeat] = result['ctrl_accuracy']
                            decode_results[key]['accuracy'][rr, repeat] = result['accuracy']
                        else:
                            decode_results[key]['ctrl_accuracy'][rr, repeat] = np.nan
                            decode_results[key]['accuracy'][rr, repeat] = np.nan

                # ensemble size analysis


            # ensemble analysis
            # nNeurons = np.arange(10, nCells, 10)
            # cellCounter = np.arange(nCells)
            # nRepeats = 20  # repeat 20 times to get an average decoding accuracy
            # decode_ensembleSize = {}
            # decode_ensembleSize['accuracy'] = np.zeros((len(regr_time), len(nNeurons)))
            # decode_ensembleSize['ctrl_accuracy'] = np.zeros((len(regr_time),
            #                                                  len(nNeurons)))
            # decode_ensembleSize['prediction_accuracy'] = {}
            # decode_ensembleSize['prediction_accuracy_ctrl'] = {}
            #
            # for idx, nN in tqdm(enumerate(nNeurons)):
            #     tempDecode = {}
            #     tempDecode['accuracy'] = np.zeros((len(regr_time), nRepeats))
            #     tempDecode['ctrl_accuracy'] = np.zeros((len(regr_time), nRepeats))
            #     tempDecode['prediction_accuracy'] = {}
            #     tempDecode['prediction_accuracy_ctrl'] = {}
            #     tempDecode['classifier'] = [[] for nn in range(nRepeats)]
            #     tempDecode['classifier_shuffle'] = [[] for nn in range(nRepeats)]
            #     for nR in range(nRepeats):
            #         # randomly picking neurons
            #         nPick = np.random.choice(cellCounter, size=nN, replace=False)
            #         for rr in range(len(regr_time)):
            #             temp = decodeSig[nPick, :, rr]
            #             result = self.run_decoder(
            #                 temp[:, notnan_trials].transpose(),
            #                 decodeVar[varname][notnan_trials], notnan_trials,
            #                 trialMask[notnan_trials],
            #                 classifier, rand_seed=nR)
            #             tempDecode['ctrl_accuracy'][rr, nR] = result['ctrl_accuracy']
            #             tempDecode['accuracy'][rr, nR] = result['accuracy']
            #             # tempDecode['classifier'][nR].append(result['classifier'])
            #             # tempDecode['classifier_shuffle'][nR].append(result['classifier_shuffle'])
            #             if rr == 0 and nR == 0:
            #                 tempDecode['test_trials'] = np.zeros(
            #                     (len(result['test_trials']), len(regr_time), nRepeats))
            #             # tempDecode['test_trials'][:, rr, nR] = result['test_trials']
            #             # get the decoding accuracy for different trialtypes
            #
            #
            #     decode_ensembleSize['accuracy'][:, idx] = np.nanmean(
            #         tempDecode['accuracy'], 1)
            #     decode_ensembleSize['ctrl_accuracy'][:, idx] = np.nanmean(
            #         tempDecode['ctrl_accuracy'], 1)
            #
            # decode_ensembleSize['nNeurons'] = nNeurons
            # decode_ensembleSize['regr_time'] = regr_time
            # decode_results[varname]['decode_realSize'] = decode_ensembleSize

        # save the decoding results
        decode_results['time'] = regr_time
        for key in ['go', 'nogo']:
            for value in decode_results[key].keys():
                decode_results[key][value] = np.nanmean(decode_results[key][value],1)
        with open(saveDataPath, 'wb') as pf:
            pickle.dump(decode_results, pf, protocol=pickle.HIGHEST_PROTOCOL)
            pf.close()
            # return decode_perform, decode_null

    def decode_analysis(self, neuronRaw, saveDataPath, saveFigPath):
        ## do some other analysis
        # plot decoding accuracy and control
        # plot decoding accuracy for false alarm trials
        #  identify cells with high importance, mark their location
        with open(saveDataPath, 'rb') as pf:
            # Load the data from the pickle file
            decode_results = pickle.load(pf)
            pf.close()

        # plot decoding accuracy for different variables
        decode_var = decode_results['var']
        nVars = len(decode_var)

        ''' plot decoding accuracy for all trials'''
        # check outcome
        #decodePlot = StartSubplots(1, nVars, ifSharey=True)

        for n in range(nVars):
            decodePlot = StartPlots()
            # plot accuracy and f1-score
            decodePlot.ax.plot(decode_results['time'],
                                  decode_results[decode_var[n]]['accuracy'], c=(1, 0, 0))
            decodePlot.ax.set_title(decode_var[n])

            #if n == 0:
            decodePlot.ax.set_ylabel('Decoding accuracy')

            # plot null control
            ctrl_results = decode_results[decode_var[n]]['ctrl_accuracy']
            #bootCtrl = bootstrap(ctrl_results.transpose(),1, 0, n_sample=50)
            #decodePlot.ax[n].fill_between(decode_results['time'], bootCtrl['bootLow'],
            #                           bootCtrl['bootHigh'],color=(0.7, 0.7, 0.7))
            decodePlot.ax.plot(decode_results['time'], ctrl_results, c=(0,0,0))

            # f1-score
            # decodePlot.ax[1,n].plot(decode_results['time'],
            #                       decode_results[decode_var[n]]['f1_score'], c=(1, 0, 0))
            # ctrl_results = decode_results[decode_var[n]]['ctrl_f1_score']
            # decodePlot.ax[1, n].plot(decode_results['time'], ctrl_results, c=(0, 0, 0))
            decodePlot.ax.set_title(decode_var[n])
            # decodePlot.ax[1, n].set_xlabel('Time from cue (s)')
            # decodePlot.ax[0, n].set_ylim([0,1])
        decodePlot.save_plot('Decoding accuracy ' + decode_var[n]+'.tiff',
                             'tiff', saveFigPath)
        plt.close()
        '''plot decoding accuracy in hit/false alarm/correct rejection trials'''
        #trialTypes = decode_results[decode_var[0]]['prediction_accuracy'].keys()
        evalList = ['accuracy']
        for eval in evalList:
            for n in range(nVars):
                decodeFAPlot = StartSubplots(2, 3, ifSharey=True)
                evalKey = 'prediction_'+eval
                evalKey_ctrl = evalKey + '_ctrl'
                plot_y1 = decode_results[decode_var[n]][evalKey]['Hit']
                # plot_y1_1 = decode_results[decode_var[n]][evalKey]['Hit_R']
                # plot_y1_2 = decode_results[decode_var[n]][evalKey]['Hit_NR']
                plot_y1_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['Hit']
                # plot_y1_1_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['Hit_R']
                # plot_y1_2_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['Hit_NR']

                decodeFAPlot.ax[0, 0].plot(decode_results['time'],
                                   plot_y1, c='blue')
                # decodeFAPlot.ax[0, 0].plot(decode_results['time'],
                #                     plot_y1_1, c = 'red')
                # decodeFAPlot.ax[0, 0].plot(decode_results['time'],
                #                     plot_y1_2, c = 'black')
        # plot ctrl in dashed line
                decodeFAPlot.ax[0, 0].plot(decode_results['time'],plot_y1_ctrl,
                                    linewidth = 0.5,linestyle = 'dashed', c='blue')
                # decodeFAPlot.ax[0, 0].plot(decode_results['time'], plot_y1_1_ctrl,
                #                    linewidth = 0.5,linestyle = 'dashed', c = 'red')
                # decodeFAPlot.ax[0, 0].plot(decode_results['time'],plot_y1_2_ctrl,
                #                    linewidth = 0.5,linestyle = 'dashed', c = 'black')
                decodeFAPlot.ax[0,0].set_title('Hit')
                decodeFAPlot.ax[0,0].set_ylabel('Accuracy')
                decodeFAPlot.ax[0, 0].set_ylim([0,1])
        # FA alarm
                plot_y1 = decode_results[decode_var[n]][evalKey]['FA']
                # plot_y1_1 = decode_results[decode_var[n]][evalKey]['FA_R']
                # plot_y1_2 = decode_results[decode_var[n]][evalKey]['FA_NR']
                plot_y1_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['FA']
                # plot_y1_1_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['FA_R']
                # plot_y1_2_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['FA_NR']

                decodeFAPlot.ax[0, 1].plot(decode_results['time'],
                                   plot_y1, c='blue')
                # decodeFAPlot.ax[0, 1].plot(decode_results['time'],
                #                     plot_y1_1, c = 'red')
                # decodeFAPlot.ax[0, 1].plot(decode_results['time'],
                #                     plot_y1_2, c = 'black')
        # plot ctrl in dashed line
                decodeFAPlot.ax[0, 1].plot(decode_results['time'],plot_y1_ctrl,
                                    linewidth = 0.5,linestyle = 'dashed', c='blue')
                # decodeFAPlot.ax[0, 1].plot(decode_results['time'], plot_y1_1_ctrl,
                #                    linewidth = 0.5,linestyle = 'dashed', c = 'red')
                # decodeFAPlot.ax[0, 1].plot(decode_results['time'],plot_y1_2_ctrl,
                #                    linewidth = 0.5,linestyle = 'dashed', c = 'black')
                decodeFAPlot.ax[0,1].set_title('FA')
                decodeFAPlot.ax[0, 1].set_ylim([0, 1])
        # CorRej trials
                plot_y1 = decode_results[decode_var[n]][evalKey]['CorRej']
                # plot_y1_1 = decode_results[decode_var[n]][evalKey]['CorRej_R']
                # plot_y1_2 = decode_results[decode_var[n]][evalKey]['CorRej_NR']
                plot_y1_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['CorRej']
                # plot_y1_1_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['CorRej_R']
                # plot_y1_2_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['CorRej_NR']

                decodeFAPlot.ax[0, 2].plot(decode_results['time'],
                                   plot_y1, c='blue')
                # decodeFAPlot.ax[0, 2].plot(decode_results['time'],
                #                     plot_y1_1, c = 'red')
                # decodeFAPlot.ax[0, 2].plot(decode_results['time'],
                #                     plot_y1_2, c = 'black')
            # plot ctrl in dashed line
                decodeFAPlot.ax[0, 2].plot(decode_results['time'],plot_y1_ctrl,
                                    linewidth = 0.5,linestyle = 'dashed', c='blue')
                # decodeFAPlot.ax[0, 2].plot(decode_results['time'], plot_y1_1_ctrl,
                #                    linewidth = 0.5,linestyle = 'dashed', c = 'red')
                # decodeFAPlot.ax[0, 2].plot(decode_results['time'],plot_y1_2_ctrl,
                #                    linewidth = 0.5,linestyle = 'dashed', c = 'black')
                decodeFAPlot.ax[0,2].set_title('CorRej')
                decodeFAPlot.ax[0,2].set_xlabel('Time from cue(s)')
                decodeFAPlot.ax[0,2].set_ylim([0, 1])
        # probe trials
                plot_y1 = decode_results[decode_var[n]][evalKey]['Probe']
                plot_y1_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['Probe']
                decodeFAPlot.ax[1, 0].plot(decode_results['time'],
                                   plot_y1, c='blue')
            # plot ctrl in dashed line
                decodeFAPlot.ax[1, 0].plot(decode_results['time'],plot_y1_ctrl,
                                    linewidth = 0.5,linestyle = 'dashed', c='blue')
                decodeFAPlot.ax[1, 0].set_title('Probe')
                decodeFAPlot.ax[1, 0].set_xlabel('Time from cue(s)')
                decodeFAPlot.ax[1, 0].set_ylim([0, 1])
        # FA cue 7 trials
        #         plot_y1_1 = decode_results[decode_var[n]][evalKey]['FA_cue7_R']
        #         plot_y1_2 = decode_results[decode_var[n]][evalKey]['FA_cue7_NR']
        #         plot_y1_1_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['FA_cue7_R']
        #         plot_y1_2_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['FA_cue7_NR']
        #
        #         decodeFAPlot.ax[1, 1].plot(decode_results['time'],
        #                             plot_y1_1, c = 'red')
        #         decodeFAPlot.ax[1, 1].plot(decode_results['time'],
        #                             plot_y1_2, c = 'black')
        #     # plot ctrl in dashed line
        #         decodeFAPlot.ax[1, 1].plot(decode_results['time'],plot_y1_ctrl,
        #                             linewidth = 0.5,linestyle = 'dashed', c='blue')
        #         decodeFAPlot.ax[1, 1].plot(decode_results['time'], plot_y1_1_ctrl,
        #                            linewidth = 0.5,linestyle = 'dashed', c = 'red')
        #         decodeFAPlot.ax[1, 1].plot(decode_results['time'],plot_y1_2_ctrl,
        #                            linewidth = 0.5,linestyle = 'dashed', c = 'black')
        #         decodeFAPlot.ax[1,1].set_title('FA Cue 7')
        #         decodeFAPlot.ax[1, 1].set_xlabel('Time from cue(s)')
        #         decodeFAPlot.ax[1, 1].set_ylim([0.3, 1])

        # CorRej cue 7 trials
        #         plot_y1_1 = decode_results[decode_var[n]][evalKey]['CorRej_cue7_R']
        #         plot_y1_2 = decode_results[decode_var[n]][evalKey]['CorRej_cue7_NR']
        #         plot_y1_1_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['CorRej_cue7_R']
        #         plot_y1_2_ctrl = decode_results[decode_var[n]][evalKey_ctrl]['CorRej_cue7_NR']

                # decodeFAPlot.ax[1, 2].plot(decode_results['time'],
                #                            plot_y1_1, c='red')
                # decodeFAPlot.ax[1, 2].plot(decode_results['time'],
                #                            plot_y1_2, c='black')
                # # plot ctrl in dashed line
                # decodeFAPlot.ax[1, 2].plot(decode_results['time'], plot_y1_ctrl,
                #                            linewidth=0.5, linestyle='dashed', c='blue')
                # decodeFAPlot.ax[1, 2].plot(decode_results['time'], plot_y1_1_ctrl,
                #                            linewidth=0.5, linestyle='dashed', c='red')
                # decodeFAPlot.ax[1, 2].plot(decode_results['time'], plot_y1_2_ctrl,
                #                            linewidth=0.5, linestyle='dashed', c='black')
                # decodeFAPlot.ax[1, 2].set_title('CorRej Cue 7')
                # decodeFAPlot.ax[1, 2].set_xlabel('Time from cue(s)')
                # decodeFAPlot.ax[1, 2].set_ylim([0.3, 1])
                #
                decodeFAPlot.fig.suptitle('Decoding_' +eval+ '_'+decode_var[n])
                decodeFAPlot.save_plot('Decoding_' +eval+ '_'+decode_var[n]+'(Trial types).tiff',
                                        'tiff', saveFigPath)
                plt.close()

        # plot decoding accuracy as a function of ensemble size
        ensemblePlot = StartPlots()
        startTime = 1
        endTime = 2 # time period to compute average decoding rate
        timeMask = np.logical_and(decode_results['time'] >= startTime,
                                decode_results['time'] < endTime)


        for n in range(nVars):
            if decode_var[n] == 'stimulus':
                # calculate average decoding accuracy
                aveAccuracy = np.nanmean(
                    decode_results[decode_var[n]]['decode_realSize']['accuracy'][timeMask,:],
                    0)
                ctrlAveAccuracy =  np.nanmean(
                    decode_results[decode_var[n]]['decode_realSize']['ctrl_accuracy'][timeMask,:],
                    0)
                ensemblePlot.ax.plot(
                    decode_results[decode_var[n]]['decode_realSize']['nNeurons'],
                                      aveAccuracy, c=(1, 0, 0))
                ensemblePlot.ax.set_title(decode_var[n])

                if n == 0:
                    ensemblePlot.ax.set_ylabel('Average decoding accuracy')
                    ensemblePlot.ax.set_xlabel('Ensemble size')
                # plot null control
                #bootCtrl = bootstrap(ctrl_results.transpose(),1, 0, n_sample=50)
                #decodePlot.ax[n].fill_between(decode_results['time'], bootCtrl['bootLow'],
                #                           bootCtrl['bootHigh'],color=(0.7, 0.7, 0.7))
                ensemblePlot.ax.plot(decode_results[decode_var[n]]['decode_realSize']['nNeurons'],
                         ctrlAveAccuracy, c=(0.5,0.5,0.5))
                ensemblePlot.fig.suptitle('Decoding_ensemble_size' + '_' + decode_var[n])
                ensemblePlot.save_plot('Decoding_ensemble_size' + '_' + decode_var[n] + '(Trial types).tiff',
                                       'tiff', saveFigPath)
                plt.close()
        ''' examine the important neurons'''
        # determine the relative importance by median importance between 0-3 s from cue
        # importantNeurons = {}
        # regr_time = decode_results['time']
        # win = np.logical_and(regr_time > 0, regr_time < 3)
        # p_thresh = 0.01
        # for var in decode_var:
        #     # get the neurons that is significantly more than population median
        #     if var == 'outcome':
        #         popMedianImportance = np.median(decode_results[var]['importance'][:, :, win])
        #         nFeatures = decode_results[var]['importance'].shape[1]
        #     else:
        #         popMedianImportance = np.median(decode_results[var]['importance'][:,win])
        #         nFeatures = decode_results[var]['importance'].shape[0]
        #     importantNeurons[var] = []
        #
        #     for n in range(nFeatures):
        #         #if var == 'outcome':
        #         _, p_val = wilcoxon(decode_results[var]['importance'][n,win]-popMedianImportance)
        #         if p_val < p_thresh:
        #             importantNeurons[var].append(n)
        #
        # # plot neuron position
        # cellstat = []
        # for cell in range(neuronRaw.Fraw.shape[0]):
        #     if neuronRaw.cells[cell, 0] > 0:
        #         cellstat.append(neuronRaw.stat[cell])
        #
        # fluoCellPlot = StartPlots()
        # im = np.zeros((256, 256,3))
        #
        # #for cell in range(decode_results[var]['importance'].shape[0]):
        # for cell in range(len(cellstat)):
        #     xs = cellstat[cell]['xpix']
        #     ys = cellstat[cell]['ypix']
        #     if cell not in \
        #             set(importantNeurons['action'])|set( importantNeurons['outcome'])|set( importantNeurons['stimulus']):
        #         im[ys, xs] = [0.7, 0.7, 0.7]
        #
        # for cell in importantNeurons['action']:
        #     xs = cellstat[cell]['xpix']
        #     ys = cellstat[cell]['ypix']
        #         #im[ys,xs] = [0,0,0]
        #     im[ys, xs] = np.add(im[ys, xs], [1.0, 0.0, 0.0])
        # for cell in importantNeurons['outcome']:
        #     xs = cellstat[cell]['xpix']
        #     ys = cellstat[cell]['ypix']
        #         #im[ys, xs] = [0, 0, 0]
        #     im[ys,xs] = np.add(im[ys,xs],[0.0,1.0,0.0])
        # for cell in importantNeurons['stimulus']:
        #     xs = cellstat[cell]['xpix']
        #     ys = cellstat[cell]['ypix']
        #         #im[ys, xs] = [0, 0, 0]
        #     im[ys,xs] = np.add(im[ys,xs],[0.0,0.0,1.0])
        # action_patch = mpatches.Patch(color=(1,0,0), label='Action')
        # outcome_patch = mpatches.Patch(color=(0,1,0), label = 'Outcome')
        # stimulus_patch = mpatches.Patch(color=(0, 0, 1), label='Stimulus')
        # # Create a custom legend with the green patch
        # plt.legend(handles=[action_patch, outcome_patch, stimulus_patch],loc='center left',bbox_to_anchor=(1, 0.5))
        # fluoCellPlot.ax.imshow(im, cmap='CMRmap')
        # #plt.show()
        #
        # fluoCellPlot.save_plot('Decoding neuron coordinates.tiff', 'tiff', saveFigPath)
        plt.close('all')

    def run_decoder(self, input_x, input_y, notnan_trials,trialMask, classifier, rand_seed, decoding_type):
        # hand-made decoder
        # return a trained decoder that can be used to decode subset of signals in a manner of testing set
        # need to make a balanced training set
        # if we want to balance the dataset
        # for each trial_type, use SMOTE to balance them, then combine every group - if we want to balance the dataset
        # smote = SMOTE(random_state=42)
        # Fit and transform the data
        #X_resampled, y_resampled = smote.fit_resample(x, y)
        #
        #X_test, y_test, X_train, y_train = self.get_train_test(x, y, test_size = 0.5, random_state = 66)

        # split the trials based on Hit (1), FA(2), CorRej(3), Miss(4), and preiously rewarded (1), previously unrewarded(-1)
        tempTrial = self.beh['trialType']
        trialType = np.zeros((len(self.beh['trialType'])))

        if decoding_type=='reg':
            for ii in range(len(self.beh['trialType'])):
                if tempTrial[ii] >= 1:  # hit
                    trialType[ii] = 1
                elif tempTrial[ii] == -1: # FA
                    trialType[ii] = 2
                elif tempTrial[ii]==0: # correct rejection
                        trialType[ii] = 3
                elif tempTrial[ii] == -2: # too few missed trials, combine with CorRej for simplicity
                    trialType[ii] = 3

            trialType = trialType[notnan_trials]


            stratefy_var = trialType[trialMask]
        elif decoding_type=='hardeasy':
            stratefy_var = input_y
            # # probably gonna delete this later - no effect of previous outcome on decoding accuracy detected
            # # exclude probe trials
            # Outcome = [r if r == 1 else -1 for r in self.beh['reward']]
            # preOutcome = [np.nan] + Outcome[0:-1]
            # # exclude the first trial to avoid nans
            # if trialMask[0]:
            #     trialMask[0] = False
            # preOutcome = np.array(preOutcome)[notnan_trials]
            # stratefy_var = trialType[trialMask] * preOutcome[trialMask]

        # get the value of most frequent trials
        counter = Counter(stratefy_var)
        for ii in np.unique(stratefy_var):
            if sum(stratefy_var==ii) == 1: # too few trials
                idx = np.where(stratefy_var == ii)[0]
                stratefy_var[idx] = counter.most_common(1)[0][0]
        # check if there is too few trials, if yes, make it to be a close trial type

        x = input_x[trialMask,:]
        y = input_y[trialMask]

        # repeat 20 times to get an average

        trial_identifiers = np.arange(len(y))
        X_train, X_test, y_train, y_test,trial_train, trial_test = train_test_split(x, y, trial_identifiers,
                                                            test_size=0.5, random_state=rand_seed, stratify=stratefy_var)

        # get the original trial Index that are split into test trial, will be used later to apply trained decoder to differnet trial types
        ori_trial = np.arange(len(trialMask))
        used_trial = ori_trial[trialMask]
        testTrials = [used_trial[ii] for ii in trial_test]

        if classifier == 'RandomForest':

            rfc = RFC(class_weight='balanced')
            n_estimators = [int(w) for w in np.linspace(start = 10, stop = 500, num=10)]
            max_depth = [int(w) for w in np.linspace(5, 20, num=10)] # from sqrt(n) - n/2
            min_samples_leaf = [0.1]
            max_depth.append(None)

            # create random grid
            random_grid = {
                'n_estimators': n_estimators,
                'min_samples_leaf': min_samples_leaf,
                'max_depth': max_depth
            }
            rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,
                                            n_iter=100, cv=3, verbose=False,
                                            random_state=42, n_jobs=-1)
            #rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,
            #                                n_iter=100, cv=5, verbose=False,
            #                                random_state=42)

            # Fit the model
            rfc_random.fit(X_train, y_train)
            # print results
            #print(rfc_random.best_params_)

            best_n_estimators = rfc_random.best_params_['n_estimators']
            #best_n_estimators = 10
            #best_max_features = rfc_random.best_params_['max_features']
            best_max_depth = rfc_random.best_params_['max_depth']
            best_min_samples_leaf = rfc_random.best_params_['min_samples_leaf']
            model = RFC(n_estimators = best_n_estimators,
                           max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf, class_weight='balanced')

            # get control value by shuffling trials

            model.fit(X_train, y_train)

            # best_cv_score = cross_val_score(best_rfc,x,y,cv=10,scoring='roc_auc')
            #from sklearn.metrics import balanced_accuracy_score
            # print(balanced_accuracy_score(y_train,best_rfc.predict(X_train)))
            # calculate decoding accuracy based on confusion matrix
            best_predict = model.predict(X_test)
            proba_estimates = model.predict_proba(X_test)
            pred = confusion_matrix(y_test, best_predict)
            pred_accuracy = np.trace(pred)/np.sum(pred)

            # feature importance
            # F1-score? in model?
            importance = model.feature_importances_
            precision = pred[1,1]/(pred[1,1]+pred[0,1])
            recall = pred[1,1]/(pred[1,1]+pred[1,0])
            f1_score = 2*(precision*recall)/(precision+recall)
            # need to return: classfier (to decode specific trial type later)
            #                 shuffled accuracy (control)
            #                 accuracy (decoding results)
            #                 importance
            #                 best parameters of the randomforest decoder


            # control
            # shuffle training model, retrain a new model
            # calculate predict accuracy on test set

            xInd = np.arange(len(y_train))
            X_train_shuffle = np.zeros((X_train.shape))
            for cc in range(X_train.shape[1]):
                np.random.shuffle(xInd)
                X_train_shuffle[:,cc] = X_train[xInd,cc]

            rfc_shuffle = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid,
                                                n_iter=100, cv=3, verbose=False,
                                                random_state=rand_seed, n_jobs=-1)
            rfc_shuffle.fit(X_train_shuffle, y_train)
            best_n_estimators = rfc_shuffle.best_params_['n_estimators']
            best_max_depth = rfc_shuffle.best_params_['max_depth']
            best_min_samples_leaf = rfc_shuffle.best_params_['min_samples_leaf']
            model_shuffle = RFC(n_estimators=best_n_estimators,
                            max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, class_weight='balanced')
            model_shuffle.fit(X_train_shuffle, y_train)
            predict_shuffle = model_shuffle.predict(X_test)
            pred = confusion_matrix(y_test, predict_shuffle)
            pred_shuffle= np.trace(pred) / np.sum(pred)
            precision_shuffle = pred[1, 1] / (pred[1, 1] + pred[0, 1])
            recall_shuffle = pred[1, 1] / (pred[1, 1] + pred[1, 0])
            f1_score_shuffle = 2 * (precision_shuffle* recall_shuffle) / (precision_shuffle + recall_shuffle)

        elif classifier == 'SVC':
            model = SVC(kernel='linear',class_weight='balanced')

            # best_cv_score = cross_val_score(best_rfc,x,y,cv=10,scoring='roc_auc')
            # from sklearn.metrics import balanced_accuracy_score
            # print(balanced_accuracy_score(y_train,best_rfc.predict(X_train)))
            # calculate decoding accuracy based on confusion matrix
            model.fit(X_train,y_train)
            best_predict = model.predict(X_test)
            pred = confusion_matrix(y_test, best_predict)
            pred_accuracy = np.trace(pred) / np.sum(pred)

            # feature importance
            importance = model.coef_
            #if pred_var == 'outcome'
            # precision = pred[1,1]/(pred[1,1]+pred[0,1])
            # recall = pred[1,1]/(pred[1,1]+pred[1,0])
            # f1_score = 2*(precision*recall)/(precision+recall)
            class_report = classification_report(y_test, best_predict, output_dict=True)
            f1_score = class_report['macro avg']['f1-score']
            # shuffled control

            xInd = np.arange(len(y_train))
            X_train_shuffle = np.zeros((X_train.shape))
            for cc in range(X_train.shape[1]):
                np.random.shuffle(xInd)
                X_train_shuffle[:,cc] = X_train[xInd,cc]

            model_shuffle = SVC(kernel='linear',class_weight='balanced')
            model_shuffle.fit(X_train_shuffle, y_train)
            predict_shuffle = model_shuffle.predict(X_test)
            pred = confusion_matrix(y_test, predict_shuffle)
            pred_shuffle = np.trace(pred) / np.sum(pred)
            #precision_shuffle = pred[1, 1] / (pred[1, 1] + pred[0, 1])
            #recall_shuffle = pred[1, 1] / (pred[1, 1] + pred[1, 0])
            #f1_score_shuffle = 2 * (precision_shuffle * recall_shuffle) / (precision_shuffle + recall_shuffle)
            class_report = classification_report(y_test, predict_shuffle, output_dict=True)
            f1_score_shuffle = class_report['macro avg']['f1-score']

        decoder = {}
        decoder['classifier'] = model
        decoder['classifier_shuffle'] = model_shuffle
        decoder['ctrl_accuracy'] = pred_shuffle
        #decoder['ctrl_precision'] = precision_shuffle
        #decoder['ctrl_recall'] = recall_shuffle
        decoder['ctrl_f1_score'] = f1_score_shuffle
        decoder['accuracy'] = pred_accuracy
        decoder['importance'] = importance
        #decoder['precision'] = precision
        #decoder['recall'] = recall
        decoder['f1_score'] = f1_score
        decoder['test_trials'] = testTrials

        if classifier == 'RandomForest':
            decoder['params'] = rfc_random.best_params_
            decoder['confidence'] = np.mean(proba_estimates,0)
        return decoder

    def run_decoder_trained_model(self, decoder, decoder_shuffle, input_x, input_y, test_trial):
        # decode subset of trials with already trained model
        if np.sum(test_trial)>0:
            x = input_x[test_trial,:]
            y = input_y[test_trial]


        # calculate decoding accuracy based on confusion matrix
            predict = decoder.predict(x)

            pred = confusion_matrix(y, predict)
            pred_accuracy = np.trace(pred) / np.sum(pred)

        # feature importance
        # F1-score? in model?
            class_report = classification_report(y, predict, output_dict=True)
            f1_score = class_report['macro avg']['f1-score']
        #     if len(pred) > 1:
        #         if pred[1,1] == 0: # when only trials with a single label is predicted
        #             precision = pred[0, 0] / (pred[0, 0] + pred[1, 0])
        #             recall = pred[0, 0] / (pred[0, 0] + pred[0, 1])
        #         else:
        #             precision = pred[1, 1] / (pred[1, 1] + pred[0, 1])
        #             recall = pred[1, 1] / (pred[1, 1] + pred[1, 0])
        #     else:
        #         precision = pred/ pred
        #         recall = pred / pred
        #     f1_score = 2 * (precision * recall) / (precision + recall)

            # control
            predict_shuffle = decoder_shuffle.predict(x)

            pred_shuffle = confusion_matrix(y, predict_shuffle)
            pred_accuracy_shuffle = np.trace(pred_shuffle) / np.sum(pred_shuffle)

            # feature importance
            # F1-score? in model?
            class_report = classification_report(y, predict_shuffle, output_dict=True)
            f1_score_shuffle = class_report['macro avg']['f1-score']
            # if len(pred_shuffle) > 1:
            #     precision = pred_shuffle[1, 1] / (pred_shuffle[1, 1] + pred_shuffle[0, 1])
            #     recall = pred_shuffle[1, 1] / (pred_shuffle[1, 1] + pred_shuffle[1, 0])
            # else:
            #     precision = pred_shuffle / pred_shuffle
            #     recall = pred_shuffle / pred_shuffle
            # f1_score_shuffle = 2 * (precision * recall) / (precision + recall)

            decode_results = {}
            decode_results['accuracy'] = pred_accuracy
            decode_results['f1_score'] = f1_score
            decode_results['accuracy_ctrl'] = pred_accuracy_shuffle
            decode_results['f1_score_ctrl'] = f1_score_shuffle
        else:
            decode_results = {}
            decode_results['accuracy'] = np.nan
            decode_results['f1_score'] = np.nan
            decode_results['accuracy_ctrl'] = np.nan
            decode_results['f1_score_ctrl'] = np.nan
        return decode_results

    def get_train_test(self, X, y, test_size, random_state):
        # check number of classes
        random.seed(random_state)
        classes = np.unique(y)
        nClass = len(np.unique(y))

        instance_class = np.zeros(nClass)
        for cc in range(nClass):
            instance_class[cc] = np.sum(y==classes[cc])

        minIns = np.min(instance_class)
        minInd = np.unravel_index(np.argmin(instance_class),instance_class.shape)
        minClass = classes[minInd]

        # split the trials based on test_size and the class with minimum instances
        classCountTest = np.sum(y==minClass)*test_size
        trainInd = []
        testInd = []
        for nn in range(nClass):
            tempClassInd = np.arange(len(y))[y==classes[nn]]
            tempTestInd = random.choices(tempClassInd,
                                          k=int(classCountTest))
            IndRemain = np.setdiff1d(tempClassInd,tempTestInd)
            tempTrainInd = random.choices(IndRemain,
                                          k=int(classCountTest))
            testInd = np.concatenate([testInd,tempTestInd])
            trainInd = np.concatenate([trainInd,tempTrainInd])

        trainInd = trainInd.astype(int)
        testInd = testInd.astype(int)

        X_train = X[trainInd,:]
        X_test = X[testInd,:]
        y_train = y[trainInd]
        y_test = y[testInd]

        return X_train,y_train, X_test, y_test

    def get_dpca(self, signal, var, regr_time, save_data_path):
        """
        PCA analysis on calcium data
        demixed PCA
        signal: shape [numCells, time_bin, stim, action]
        # python code not working, output the data for dPCA in matlab?
        referece: https://pietromarchesi.net/pca-neural-data.html#theforms
        PCA averaged-concatenated (hybrid) form
        """


        # prepare the signal for dpca input
        nVars = var.keys()
        nCells = signal.shape[0]
        nTimeBins = signal.shape[2]
        nTrials = signal.shape[1]
        trialInd = np.arange(nTrials)

        # reorganize signal into trial
        trials = [signal[:,i,:] for i in range(nTrials)]

        signal_pca = []
        # for the last two dimensions
        # (0, 0): nogo+nolick; (0, 1): nogo+lick
        # (1, 0): go+miss; (1,1): go+lick
        hitTrials = trialInd[np.logical_and(var['stim']==1, var['action']==1)]
        FATrials = trialInd[np.logical_and(var['stim']==0, var['action']==1)]
        CorRejTrials = trialInd[np.logical_and(var['stim']==0, var['action']==0)]
        #missTrials = trialInd[np.logical_and(var['stim']==1, var['action']==0)]
        signal_pca.append(np.mean(signal[:,hitTrials,:],1))
        signal_pca.append(np.mean(signal[:, FATrials, :], 1))
        signal_pca.append(np.mean(signal[:, CorRejTrials, :], 1))
        #signal_pca.append(np.mean(signal[:, missTrials, :], 1))

        signal_ave = np.hstack(signal_pca)

        # z-score the average df/f
        ss = StandardScaler(with_mean=True, with_std=True)
        signal_ave_sc = ss.fit_transform(signal_ave.T).T

        n_components = 15
        pca = PCA(n_components=n_components)
        pca.fit(signal_ave_sc.T)

        # plot amount of variance explained?
        varPlot = StartPlots()
        varPlot.ax.plot(np.cumsum(pca.explained_variance_ratio_),'o-')
        varPlot.ax.set_title('Variance explained')
        varPlot.save_plot('Variance explained.tiff', 'tif', save_data_path)

        # project individual trials to fitted PCA
        projected_trials = []
        signal_sc = np.zeros((signal.shape[0], signal.shape[1], signal.shape[2]))
        for trial in trials:
            trial = ss.transform(trial.T).T
            proj_trial = pca.transform(trial.T).T
            projected_trials.append(proj_trial)
        trial_types = ['Hit', 'FA', 'CorRej']
        gt = {comp: {t_type: [] for t_type in trial_types}
              for comp in range(n_components)}

        for comp in range(n_components):
            for t in hitTrials:
                tt = projected_trials[t][comp,:]
                gt[comp]['Hit'].append(tt)
            for t in FATrials:
                tt = projected_trials[t][comp,:]
                gt[comp]['FA'].append(tt)
            for t in CorRejTrials:
                tt = projected_trials[t][comp,:]
                gt[comp]['CorRej'].append(tt)

        save_file_path = os.path.join(save_data_path, 'dPCA_results.pickle')
        with open(save_file_path, 'wb') as file:
            pickle.dump(gt, file)
            '''use own plot function, include bootstrap CIs'''
        #matplotlib.use('QtAgg')
        pal = sns.color_palette('husl', 4)
        PCPlot = StartSubplots(1,4,figsize=(20, 6), ifSharex = True, ifSharey=True)

        for comp in range(4):
            ax = PCPlot.ax[comp]
            for t, t_type in enumerate(trial_types):
                    #sns.tsplot(gt[comp][t_type], time=regr_time, ax=ax,
                    #           err_style='ci_band',
                    #           ci=95,
                    #           color=pal[t])
                #ax.plot(regr_time,np.mean(gt[comp][t_type],0), label=t_type)
                # bootstrap
                tempdata = np.array(gt[comp][t_type])
                tempBoot = bootstrap(tempdata.T,1,1000)
                ax.plot(regr_time, tempBoot['bootAve'],label=t_type)
                ax.fill_between(regr_time, tempBoot['bootLow'],
                                        tempBoot['bootHigh'],
                                        alpha=0.2)

            #add_stim_to_plot(ax)
            ax.set_ylabel('PC {}'.format(comp + 1))

        PCPlot.ax[1].set_xlabel('Time (s)')
        PCPlot.ax[2].legend(['Hit', 'FA', 'CorRej'])
        #sns.despine(right=True, top=True)
        #add_orientation_legend(axes[2])
        PCPlot.save_plot('Principle components.tiff', 'tif', save_data_path)

        '''3d PCA projections for trial-average trajetory'''
        ave_pca = PCA(n_components = 3)
        signal_ave_p = ave_pca.fit_transform(signal_ave_sc.T).T

        component_x = 0
        component_y = 1
        component_z = 2

        sigma = 3 # smoothing param

        fig = plt.figure(figsize=[12,12])
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

        trial_size = len(regr_time)
        for t, t_type in enumerate(trial_types[0:3]):
            x = signal_ave_p[component_x, t * trial_size :(t+1) * trial_size]
            y = signal_ave_p[component_y, t * trial_size :(t+1) * trial_size]
            z = signal_ave_p[component_z, t * trial_size :(t+1) * trial_size]

                # apply some smoothing to the trajectories
            x = gaussian_filter1d(x, sigma=sigma)
            y = gaussian_filter1d(y, sigma=sigma)
            z = gaussian_filter1d(z, sigma=sigma)

                # use the mask to plot stimulus and pre/post stimulus separately

            ax.plot(x, y, z, c=pal[t], label=t_type)

            # plot t = -1, t= 0, t=2
            ax.scatter(x[0], y[0],z[0], c=pal[t], s= 50, marker='o', label='-1 s')
            ax.scatter(x[10], y[10], z[10], c=pal[t], s=50, marker='x', label='0 s')
            ax.scatter(x[-1], y[-1],z[-1], c=pal[t], s=50, marker='^', label='2 s')
        ax.legend(loc='right', bbox_to_anchor=(2, 1))

        output_file = os.path.join(save_data_path, "dPCA_projection.png")
        plt.savefig(output_file)

        output_pickle = os.path.join(save_data_path, "dPCA_projection.fig.pickle")
        pickle.dump(fig,open(output_pickle, 'wb'))


        # load the figure
        #figx = pickle.load(open(output_pickle, 'rb'))
        #plt.show()  # Show the figure, edit it, etc.!

        '''3d PCA projections for each individual trial'''

        sigma = 3 # smoothing param

        fig = plt.figure(figsize=[12,12])
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

        trial_size = len(regr_time)
        for t,trial in enumerate(projected_trials[0:40]):
            if t in hitTrials or t in FATrials or t in CorRejTrials:
                x = trial[0,:]
                y = trial[1,:]
                z = trial[2,:]

                    # apply some smoothing to the trajectories
                x = gaussian_filter1d(x, sigma=sigma)
                y = gaussian_filter1d(y, sigma=sigma)
                z = gaussian_filter1d(z, sigma=sigma)

                    # use the mask to plot stimulus and pre/post stimulus separately
                if t in hitTrials:
                    label = 'hit'
                    colorInd=0
                elif t in FATrials:
                    label='FA'
                    colorInd = 1
                elif t in CorRejTrials:
                    label = 'CorRej'
                    colorInd = 2

                ax.plot(x, y, z, c=pal[colorInd], linewidth = 0.5, label=label)

                # plot t = -1, t= 0, t=2
                ax.scatter(x[0], y[0],z[0], c=pal[colorInd], s= 50, marker='o', label='-1 s')
                ax.scatter(x[10], y[10], z[10], c=pal[colorInd], s=50, marker='x', label='0 s')
                ax.scatter(x[-1], y[-1],z[-1], c=pal[colorInd], s=50, marker='^', label='2 s')
            ax.legend(loc='right', bbox_to_anchor=(2, 1))

            output_file = os.path.join(save_data_path, "dPCA_projection_trial.png")
            plt.savefig(output_file)

            output_pickle = os.path.join(save_data_path, "dPCA_projection_trial.fig.pickle")
            pickle.dump(fig,open(output_pickle, 'wb'))

        plt.close('all')

        # animated trajectory plots
        # set up a dictionary to color each line
        # col = {trial_types[i]: pal[i] for i in range(len(trial_types))}
        #
        # fig = plt.figure(figsize=[9, 9]);
        # plt.close()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        #
        # def style_3d_ax(ax):
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_zticks([])
        #     ax.xaxis.pane.fill = False
        #     ax.yaxis.pane.fill = False
        #     ax.zaxis.pane.fill = False
        #     ax.xaxis.pane.set_edgecolor('w')
        #     ax.yaxis.pane.set_edgecolor('w')
        #     ax.zaxis.pane.set_edgecolor('w')
        #     ax.set_xlabel('PC 1')
        #     ax.set_ylabel('PC 2')
        #     ax.set_zlabel('PC 3')
        #
        # def animate(i):
        #
        #     ax.clear()
        #     style_3d_ax(ax)
        #     ax.view_init(elev=22, azim=30)
        #     for t, trial in enumerate(zip(projected_trials[0:40]):
        #         x = trial[0, :][0:i]
        #         y = trial[1, :][0:i]
        #         z = trial[2, :][0:i]
        #
        #         if t in hitTrials:
        #             label = 'hit'
        #             colorInd = 0
        #         elif t in FATrials:
        #             label = 'FA'
        #             colorInd = 1
        #         elif t in CorRejTrials:
        #             label = 'CorRej'
        #             colorInd = 2
        #
        #     ax.plot(x, y, z, c=pal[colorInd], linewidth=0.5, label=label)
        #
        #     ax.set_xlim((-12, 12))
        #     ax.set_ylim((-12, 12))
        #     ax.set_zlim((-13, 13))
        #     ax.view_init(elev=22, azim=30)
        #
        #     return []
        #
        # anim = animation.FuncAnimation(fig, animate, frames=len(pca_frame), interval=50, blit=True)
        #
        # plt.show()
    def noise_analysis(self, subTrialMask,save_data_path):
        # analysis of noise in the intertrial interval
        # reference: Rowland, 2023, Nature Neuroscience

        # 1. P(Hit) as a function of pre-stimulus variance
        # 2. latent factor analysis and covariance structure of pre-trial activity

        # calculate prestimulus (-0.5s - 0s) population mean and variance
        # for every trial, then take z-score
        saveData = {}

        """ calculate the population mean and variance"""
        # arrange original dFF and t into matrix, add nan for missing value

        # find the maximum length
        nTrials = self.dFF_aligned.shape[1]
        nCells = self.dFF_aligned.shape[2]
        maxLength = 0
        for trial in self.t_original.keys():
            if len(self.t_original[trial]) > maxLength:
                maxLength = len(self.t_original[trial])

        signal = np.full((maxLength,nTrials, nCells), np.nan)
        time = np.full((maxLength,nTrials), np.nan)
        startTime = -0.5
        endTime = 0
        timeMask = np.full((maxLength,nTrials), np.nan)
        for trial in self.dFF_original.keys():
            signal[:len(self.dFF_original[trial]), int(trial),:] = self.dFF_original[trial]
            time[:len(self.dFF_original[trial]), int(trial)] = self.t_original[trial]
            timeMask[:len(self.dFF_original[trial]), int(trial)] = np.logical_and(self.t_original[trial]>=startTime,
                                                                                  self.t_original[trial] < -endTime)
        timeMask[np.isnan(timeMask)] = False

        # get the max length of precue period data
        maxLength = int(max(np.sum(timeMask,0)))

        # merge dFF by trial types
        # calculate the mean and variance by trial type, plot P(Hit) as a function
        # of population variance


        dFF_byTrial = np.full((maxLength,nCells,nTrials),np.nan)
        mean_byTrial = np.full((nTrials), np.nan)
        var_byTrial = np.full((nTrials),np.nan)
        for trial in range(nTrials):
            tempSig = signal[:,trial,:]
            tempTime = timeMask[:,trial].astype(bool)
            if np.sum(tempTime) > 0:
                dFF_byTrial[:np.sum(tempTime),:,trial] = tempSig[tempTime,:]
                mean_byTrial[trial] = np.nanmean(tempSig[tempTime,:])
                var_byTrial[trial] = np.nanvar(tempSig[tempTime,:])

        # z-score pop mean and variance
        zmean_byTrial = (mean_byTrial - np.nanmean(mean_byTrial))/np.nanstd(mean_byTrial)
        zvar_byTrial = (var_byTrial - np.nanmean(var_byTrial)) / np.nanstd(var_byTrial)

        maxZ = 1
        minZ = -1
        # bin zmean and zvar and plot P(Hit) as a function of zmean and zvar
        binsize = 0.1
        larger_binsize = 0.2
        binX = np.arange(minZ+binsize/2, maxZ-binsize/2,binsize)
        pFA_binMean = np.full((len(binX)),np.nan)
        pFA_binVar = np.full((len(binX)),np.nan)
        trialCounter = np.arange(nTrials)
        nTrial_binMean = np.full((len(binX)),np.nan)
        nTrial_binVar = np.full((len(binX)),np.nan)
        for idx,x in enumerate(binX):
            lLimit = x-larger_binsize/2
            uLimit = x+larger_binsize/2
            trials = np.logical_and(zmean_byTrial>=lLimit,
                                  zmean_byTrial<uLimit)
            nTrial_binMean[idx] = np.sum(np.logical_or(self.beh['trialType'][trials]==-1,
                                                       self.beh['trialType'][trials]==0))
            pFA_binMean[idx] = np.sum(self.beh['trialType'][trials] ==-1)/nTrial_binMean[idx]

            trials = np.logical_and(zvar_byTrial >= lLimit,
                                    zvar_byTrial < uLimit)
            nTrial_binVar[idx] = np.sum(np.logical_or(self.beh['trialType'][trials]==-1,
                                                       self.beh['trialType'][trials]==0))
            pFA_binVar[idx] = np.sum(self.beh['trialType'][trials] ==-1) / nTrial_binVar[idx]

            # plot false alarm rate as a function of
            # binned population mean and var
        CorPlot = StartPlots()
        CorPlot.ax.plot(binX, pFA_binVar)
        CorPlot.ax.plot(binX,pFA_binMean)
        CorPlot.ax.legend(['Variance', 'Mean'])
        CorPlot.ax.set_xlabel('Z-score')
        CorPlot.ax.set_ylabel('False alarm rate')
        CorPlot.ax.set_title('False alarm rate - population var & mean')
        CorPlot.save_plot('False alarm rate vs population var and mean.tiff',
                          'tiff', save_data_path)
        plt.close()

        saveData['mean_byTrial'] = zmean_byTrial
        saveData['var_byTrial'] = zvar_byTrial
        saveData['pFA_Mean'] = pFA_binMean
        saveData['pFA_Var'] = pFA_binVar
        saveData['binX'] = binX

        """calculate signal correlation and noise correlation, precue and postcue"""
        # reference: Montjin, 2014
        # need to do interpolate to compute pearson correlation
        timeStart = -2
        timeEnd = 2
        interpT = np.arange(timeStart, timeEnd,0.05)
        noiseInterp = np.full((len(interpT), nTrials, nCells), np.nan)
        for tt in range(nTrials):
            t_dFF = self.t_original[str(tt)]
            for cc in range(nCells):
                tempT = t_dFF[np.logical_and(t_dFF > timeStart, t_dFF <= timeEnd)]
                tempdFF = self.dFF_original[str(tt)][np.logical_and(t_dFF > timeStart, t_dFF <= timeEnd),cc]
                if len(tempT)>0:
                    noiseInterp[:, tt, cc] = np.interp(interpT, tempT, tempdFF)
        label = 'real'
        self.signal_noise_correlation(noiseInterp, interpT,
                                     subTrialMask, save_data_path, label)

            # calculate pearson correlation for every time point, obtaining a
            # continuous correlation matrix for a trial
        noise_corr_trial = np.full((nCells, nCells, len(interpT)),np.nan)
        signal_corr_trial = np.full((nCells,nCells, len(interpT)), np.nan)
        for ii in tqdm(range(nCells))                                 :
            for jj in range(ii, nCells):
                # calculate noise correlations per time point
                matrix1 = noiseInterp[:,subTrialMask['Hit'],ii].T
                matrix2 = noiseInterp[:, subTrialMask['Hit'],jj].T
                valid_rows = ~np.isnan(matrix1).any(axis=1) & ~np.isnan(matrix2).any(axis=1)
                matrix1 = matrix1[valid_rows, :]
                matrix2 = matrix2[valid_rows, :]
                Hitcoeff = np.diag(np.corrcoef(
                        matrix1, matrix2, rowvar=False)[:len(interpT),
                                       len(interpT):])
                    # FA
                matrix1 = noiseInterp[:,subTrialMask['FA'],ii].T
                matrix2 = noiseInterp[:, subTrialMask['FA'],jj].T
                valid_rows = ~np.isnan(matrix1).any(axis=1) & ~np.isnan(matrix2).any(axis=1)
                matrix1 = matrix1[valid_rows, :]
                matrix2 = matrix2[valid_rows, :]
                nFA = np.sum(subTrialMask['FA'])
                FAcoeff = np.diag(np.corrcoef(
                        matrix1, matrix2, rowvar=False)[:len(interpT),
                                       len(interpT):])
                    # CR
                matrix1 = noiseInterp[:,subTrialMask['CorRej'],ii].T
                matrix2 = noiseInterp[:, subTrialMask['CorRej'],jj].T
                valid_rows = ~np.isnan(matrix1).any(axis=1) & ~np.isnan(matrix2).any(axis=1)
                matrix1 = matrix1[valid_rows, :]
                matrix2 = matrix2[valid_rows, :]
                nCR = np.sum(subTrialMask['CorRej'])
                CRcoeff = np.diag(np.corrcoef(
                        matrix1, matrix2, rowvar=False)[:len(interpT),
                                       len(interpT):])
                noise_corr_trial[ii,jj,:] = np.nanmean(
                        [Hitcoeff,FAcoeff,CRcoeff],0)

                # calculate signal correlation per time point
                HitAve1 = np.nanmean(noiseInterp[:,subTrialMask['Hit'],ii],1)
                HitAve2 = np.nanmean(noiseInterp[:,subTrialMask['Hit'],jj],1)
                FAAve1 = np.nanmean(noiseInterp[:,subTrialMask['FA'],ii],1)
                FAAve2 = np.nanmean(noiseInterp[:, subTrialMask['FA'], jj],1)
                CRAve1 = np.nanmean(noiseInterp[:, subTrialMask['CorRej'], ii],1)
                CRAve2 = np.nanmean(noiseInterp[:, subTrialMask['CorRej'], jj],1)

                matrix1 = np.concatenate((HitAve1[:,np.newaxis],
                                          FAAve1[:,np.newaxis],
                                          CRAve1[:,np.newaxis]),1).T
                matrix2 = np.concatenate((HitAve2[:,np.newaxis],
                                          FAAve2[:,np.newaxis],
                                          CRAve2[:,np.newaxis]),1).T
                signal_corr_trial[ii,jj,:] = np.diag(np.corrcoef(
                        matrix1, matrix2, rowvar=False)[:len(interpT),
                                       len(interpT):])
        # calculate second order correlation - correlation of correlation matrix in
        # neighbering time points
        noise_corr2_trial = np.zeros(len(interpT)-1)
        signal_corr2_trial = np.zeros(len(interpT)-1)
        for idx in range(len(interpT)-1):
            matrix1 = noise_corr_trial[:,:,idx]
            matrix2 = noise_corr_trial[:,:,idx+1]
            valid_rows = ~np.isnan(matrix1).any(axis=1) & ~np.isnan(matrix2).any(axis=1)
            matrix1 = matrix1[valid_rows, :]
            matrix2 = matrix2[valid_rows, :]
            noise_corr2_trial[idx] = np.corrcoef(matrix1.flatten(),
                                                 matrix2.flatten())[0, 1]
            matrix1 = signal_corr_trial[:,:,idx]
            matrix2 = signal_corr_trial[:,:,idx+1]
            valid_rows = ~np.isnan(matrix1).any(axis=1) & ~np.isnan(matrix2).any(axis=1)
            matrix1 = matrix1[valid_rows, :]
            matrix2 = matrix2[valid_rows, :]
            signal_corr2_trial[idx] = np.corrcoef(matrix1.flatten(),
                                                 matrix2.flatten())[0, 1]
        # make some plots
        corrPlots = StartPlots()
        corrPlots.ax.plot(interpT[:-1], noise_corr2_trial)
        corrPlots.ax.plot(interpT[:-1], signal_corr2_trial)
        corrPlots.ax.set_xlabel('Time from cue (s)')
        corrPlots.ax.set_ylabel('Pearson correlation')
        corrPlots.legend(['Noise','Signal'])
        corrPlots.ax.set_title('Second order correlation')
        corrPlots.save_plot('Second order correlation.tiff', 'tiff',save_data_path)
        plt.close()
        # plot the average traces
        saveData['noise_corr_trial'] = noise_corr_trial
        saveData['signal_corr_trial'] = signal_corr_trial
        saveData['noise_corr2_trial'] = noise_corr2_trial
        saveData['signal_corr2_trial'] = signal_corr2_trial
        saveData['interp_time'] = interpT
        # save the data
        saveDataPath = os.path.join(save_data_path,'noiseResults.pickle')
        with open(saveDataPath, 'wb') as pf:
            pickle.dump(saveData, pf, protocol=pickle.HIGHEST_PROTOCOL)
            pf.close()

    def pseudoensemble_analysis(self, decodeSig, decodeVar, trialMask,
                                subTrialMask, classifier, regr_time, save_data_path):
        """ shuffle the cell-trial contigency for every individual neurons
        to build a pseudo-ensemble data that
        maintains the signal correlation, but disrupts the noise correlation
        1. running decoding to check the effect of noise correlation and the influence of
        ensemble size
        2. calculate noise correlation to validate the shuffling methods
        """
        # for noise related analysis: within each trialtype (Hit, FA, CR), shuffle the neuron trial labels independently
        # for each neuron to disrupt

        saveDataPath = os.path.join(save_data_path, 'decoding_pseudo_stimulus.pickle')
        # if not os.path.exists(saveDataPath):
        saveData = {}
        nRepeats = 10
        ## decoding
        decode_results_pseudo = {}
        decode_results_pseudo['accuracy'] = np.zeros((len(regr_time),nRepeats))
        decode_results_pseudo['ctrl_accuracy'] = np.zeros((len(regr_time), nRepeats))
        decode_results_pseudo['classifier'] = [[] for xx in range(nRepeats)]
        decode_results_pseudo['classifier_shuffle'] = [[] for xx in range(nRepeats)]
        decode_results_pseudo['prediction_accuracy'] = {}
        decode_results_pseudo['prediction_accuracy_ctrl'] = {}
        nCells = decodeSig.shape[0]
        notnany_where = np.argwhere(~np.isnan(decodeSig[0, :, 0]))
        nanx_where = np.argwhere(np.isnan(decodeVar['stimulus']))
        trialTypes = ['Hit', 'FA', 'CorRej','Miss','probe']  # decode for stimulus only (consider decode for trial types later)
        trialCounter = np.arange(decodeSig.shape[1])
        if not len(nanx_where) == 0:
            notnan_trials = int(np.setdiff1d(np.unique(notnany_where), np.unique(nanx_where)))
        else:
            notnan_trials = np.unique(notnany_where)
        # shuffle the neurons-trial pair for each neuron within trial type
        tempSig = decodeSig[:, notnan_trials, :]
        decodeSig_noiseShuffle = np.full((tempSig.shape), np.nan)
        tempCounter = np.arange(len(notnan_trials))

        for tt in trialTypes:
            if tt in subTrialMask.keys(): # Hit, FA, and CR trials
                tempMask = subTrialMask[tt][notnan_trials]
            else: # miss trials
                tempMask = ~np.logical_or.reduce([subTrialMask['Hit'],subTrialMask['FA'],
                                         subTrialMask['CorRej'],
                                          subTrialMask['probe']])[notnan_trials]
            trialIdx = tempCounter[tempMask]
            for cc in range(nCells):
                # shuffle the index of trials
                np.random.seed(cc)
                shuffledIdx = np.random.permutation(trialIdx)
                decodeSig_noiseShuffle[cc, trialIdx, :] = tempSig[cc, shuffledIdx, :]

        # decoding stimulus for pseudo ensemble data

        if classifier == "RandomForest":
            pass
        elif classifier == 'SVC':
            for repeat in range(nRepeats):
                for rr in range(len(regr_time)):
                    result = self.run_decoder(
                        decodeSig_noiseShuffle[:, :, rr].transpose(),
                        decodeVar['stimulus'][notnan_trials], notnan_trials,
                        trialMask[notnan_trials],
                        classifier, rand_seed = repeat, decoding_type='reg')
                    # decode_results[varname]['ctrl_precision'][rr] = result['ctrl_precision']
                    # decode_results[varname]['ctrl_recall'][rr] = result['ctrl_recall']
                    decode_results_pseudo['ctrl_accuracy'][rr, repeat] = result['ctrl_accuracy']
                    decode_results_pseudo['accuracy'][rr, repeat] = result['accuracy']
                    decode_results_pseudo['classifier'][repeat].append(result['classifier'])
                    decode_results_pseudo['classifier_shuffle'][repeat].append(result['classifier_shuffle'])

                    if rr == 0 and repeat == 0:
                        decode_results_pseudo['test_trials'] = np.zeros(
                            (len(result['test_trials']), len(regr_time), nRepeats))
                    decode_results_pseudo['test_trials'][:, rr, repeat] = result['test_trials']


                trial_types = ['Hit',
                       'FA',
                       'CorRej',
                       'Probe']
                for key in trial_types:
                    if repeat == 0:
                        decode_results_pseudo['prediction_accuracy'][key] = np.zeros((len(regr_time),
                                                                                nRepeats))
                # decode_results[varname]['prediction_f1_score'][key] = np.zeros(len(regr_time))
                        decode_results_pseudo['prediction_accuracy_ctrl'][key] = np.zeros((len(regr_time),
                                                                                     nRepeats))
            # decode_results[varname]['prediction_f1_score_ctrl'][key] = np.zeros(len(regr_time))
            # in probe trials, 2 -> 0; 3 -> 1

            # from test trials, determine there trial type
            # based on trial types, determine the trial type mask
            # looking at FA_cue7 with prior reward/noreward specificly
                    test_trials_ori = [notnan_trials[int(tt)] for tt in
                               decode_results_pseudo['test_trials'][:, 0, repeat]]  # test trials in original index

                    if key == 'Hit':
                        test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == 2]
                    elif key == 'FA':
                        test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == -1]
                    elif key == 'CorRej':
                        test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == 0]
                    elif key == 'Probe':
                        test_trial = [tt for tt in range(len(self.beh['trialType'])) if
                              (self.beh['trialType'][tt] == -3 or self.beh['trialType'][tt] == -4)]
                    testTrialMask = np.array([False for tt in range(len(self.beh['trialType']))])
                    testTrialMask[test_trial] = True

            # don't do parallel
                    for rr in range(len(regr_time)):
                        result_trialType = self.run_decoder_trained_model(decode_results_pseudo['classifier'][repeat][rr],
                                                        decode_results_pseudo['classifier_shuffle'][repeat][rr],
                                                        decodeSig_noiseShuffle[:, :, rr].transpose(),
                                                        decodeVar['stimulus'][notnan_trials],
                                                        testTrialMask[notnan_trials])
                        decode_results_pseudo['prediction_accuracy'][key][rr, repeat] = result_trialType['accuracy']
                        decode_results_pseudo['prediction_accuracy_ctrl'][key][rr, repeat] = result_trialType['accuracy_ctrl']

        decode_results_pseudo['ctrl_accuracy'] = np.nanmean(decode_results_pseudo['ctrl_accuracy'], 1)
        decode_results_pseudo['accuracy'] = np.nanmean(decode_results_pseudo['accuracy'], 1)
        for key in trial_types:
            decode_results_pseudo['prediction_accuracy'][key] = np.nanmean(decode_results_pseudo['prediction_accuracy'][key], 1)
            decode_results_pseudo['prediction_accuracy_ctrl'][key] = np.nanmean(
                decode_results_pseudo['prediction_accuracy_ctrl'][key], 1)
        saveData['decode_pseudo'] = decode_results_pseudo

        # examine the effect of ensemble size on decoding accuracy
        nNeurons = np.arange(10,nCells,10)
        cellCounter = np.arange(nCells)
        nRepeats = 20 # repeat 20 times to get an average decoding accuracy
        decode_ensembleSize = {}
        decode_ensembleSize['accuracy'] = np.zeros((len(regr_time),len(nNeurons)))
        decode_ensembleSize['ctrl_accuracy'] = np.zeros((len(regr_time),
                                                         len(nNeurons)))
        decode_ensembleSize['prediction_accuracy'] = {}
        decode_ensembleSize['prediction_accuracy_ctrl'] = {}
        for key in trial_types:
            decode_ensembleSize['prediction_accuracy'][key] = np.zeros((len(regr_time),len(nNeurons)))
            decode_ensembleSize['prediction_accuracy_ctrl'][key] = np.zeros((len(regr_time), len(nNeurons)))
        for idx,nN in tqdm(enumerate(nNeurons)):
            tempDecode = {}
            tempDecode['accuracy'] = np.zeros((len(regr_time), nRepeats))
            tempDecode['ctrl_accuracy'] = np.zeros((len(regr_time), nRepeats))
            tempDecode['prediction_accuracy'] = {}
            tempDecode['prediction_accuracy_ctrl'] = {}
            for nR in range(nRepeats):
                # randomly picking neurons
                nPick = np.random.choice(cellCounter, size = nN, replace = False)
                for rr in range(len(regr_time)):
                    result = self.run_decoder(
                        decodeSig_noiseShuffle[nPick, :, rr].transpose(),
                        decodeVar['stimulus'][notnan_trials], notnan_trials,
                        trialMask[notnan_trials],
                        classifier, rand_seed = nR, decoding_type = 'reg')
                    tempDecode['ctrl_accuracy'][rr,nR] = result['ctrl_accuracy']
                    tempDecode['accuracy'][rr,nR] = result['accuracy']
                      # get the decoding accuracy for different trialtypes

                    for key in trial_types:
                        if nR == 0:
                            tempDecode['prediction_accuracy'][key] = np.zeros((len(regr_time),
                                                                                            nRepeats))
                            tempDecode['prediction_accuracy_ctrl'][key] = np.zeros((len(regr_time),
                                                                                                 nRepeats))
                        test_trials_ori = [notnan_trials[int(tt)] for tt in result['test_trials']]  # test trials in original index

                        if key == 'Hit':
                            test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == 2]
                        elif key == 'FA':
                            test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == -1]
                        elif key == 'CorRej':
                            test_trial = [tt for tt in test_trials_ori if self.beh['trialType'][tt] == 0]
                        elif key == 'Probe':
                            test_trial = [tt for tt in range(len(self.beh['trialType'])) if
                                          (self.beh['trialType'][tt] == -3 or self.beh['trialType'][tt] == -4)]
                        testTrialMask = np.array([False for tt in range(len(self.beh['trialType']))])
                        testTrialMask[test_trial] = True
                        # don't do parallel

                        result_trial = self.run_decoder_trained_model(result['classifier'],
                                                                    result['classifier_shuffle'],
                                                                    decodeSig_noiseShuffle[nPick, :, rr].transpose(),
                                                                    decodeVar['stimulus'][notnan_trials],
                                                                    testTrialMask[notnan_trials])
                        tempDecode['prediction_accuracy'][key][rr, nR] = result_trial['accuracy']
                            # decode_results[varname]['prediction_f1_score'][key][rr] = result['f1_score']
                        tempDecode['prediction_accuracy_ctrl'][key][rr, nR] = result_trial[
                                    'accuracy_ctrl']

            decode_ensembleSize['accuracy'][:, idx] = np.nanmean(
                        tempDecode['accuracy'], 1)
            decode_ensembleSize['ctrl_accuracy'][:, idx] = np.nanmean(
                        tempDecode['ctrl_accuracy'], 1)
            for key in trial_types:
                decode_ensembleSize['prediction_accuracy'][key][:,idx] = np.nanmean(
                            tempDecode['prediction_accuracy'][key],1)
                decode_ensembleSize['prediction_accuracy_ctrl'][key][:,idx] = np.nanmean(
                            tempDecode['prediction_accuracy_ctrl'][key],1)

        decode_ensembleSize['nNeurons'] = nNeurons
        decode_ensembleSize['regr_time'] = regr_time
        saveData['decode_pseudoSize'] = decode_ensembleSize



        with open(saveDataPath, 'wb') as pf:
            pickle.dump(saveData, pf, protocol=pickle.HIGHEST_PROTOCOL)
            pf.close()

    #calculate the signal and noice correlation
    #interpolate the data then shuffle to create a pseudo ensemble
        nTrials = self.dFF_aligned.shape[1]
        timeStart = -2
        timeEnd = 2
        interpT = np.arange(timeStart, timeEnd,0.05)
        noiseInterp = np.full((len(interpT), nTrials, nCells), np.nan)
        for tt in range(nTrials):
            t_dFF = self.t_original[str(tt)]
            for cc in range(nCells):
                tempT = t_dFF[np.logical_and(t_dFF > timeStart, t_dFF <= timeEnd)]
                tempdFF = self.dFF_original[str(tt)][np.logical_and(t_dFF > timeStart, t_dFF <= timeEnd),cc]
                if len(tempT)>0:
                    noiseInterp[:, tt, cc] = np.interp(interpT, tempT, tempdFF)
        # shuffle
        tempSig = noiseInterp
        noiseInterp_Shuffle = np.full((tempSig.shape), np.nan)
        tempCounter = trialCounter

        for tt in trialTypes:
            if tt in subTrialMask.keys(): # Hit, FA, and CR trials
                tempMask = subTrialMask[tt]
            else: # miss trials
                tempMask = ~np.logical_or.reduce([subTrialMask['Hit'],subTrialMask['FA'],
                                         subTrialMask['CorRej'],
                                          subTrialMask['probe']])
            trialIdx = tempCounter[tempMask]
            for cc in range(nCells):
                # shuffle the index of trials
                np.random.seed(cc)
                shuffledIdx = np.random.permutation(trialIdx)
                noiseInterp_Shuffle[:, trialIdx, cc] = tempSig[:, shuffledIdx,cc]

        self.signal_noise_correlation(noiseInterp_Shuffle, interpT,
                                      subTrialMask, save_data_path, label='pseudo')

    def signal_noise_correlation(self, noiseInterp, interpT,
                                 subTrialMask, save_data_path, label):
        # function to calculate signal and noise correlation
        # noiseInterp: interpolated signal
        # interpT: interpolate time
        # 1. signal correlation
        # 2. noise correlation
        # 3. signal-noise correlation
        """calculate signal correlation and noise correlation, precue and postcue"""
        # reference: Montjin, 2014
        # interpolate to compute pearson correlation
        saveDataPath = os.path.join(save_data_path, 'signal_noise ' + label + '.pickle')

        if not os.path.exists(saveDataPath):
            nTrials = self.dFF_aligned.shape[1]
            nCells = self.dFF_aligned.shape[2]
            saveData = {}
        # 1. precue signal correlation
        # p_i,j = corr(R_i,R_j), R_i = [R_Hit, R_FA, R_CR]
        # modifaction: we calculate signal correlation by trial type
        # then take the average
            precue_start = -0.5
            precue_end = 0
            aftercue_end = 2
            trialType = ['Hit', 'FA', 'CorRej']
            precueMask = np.logical_and(interpT>=precue_start, interpT<precue_end)
            aftercueMask = np.logical_and(interpT>=precue_end, interpT<aftercue_end)
            precueLength = np.sum(precueMask)
            aftercueLength = np.sum(aftercueMask)
            average_resp_preCue = {}
            average_resp_afterCue = {}# 3 types of trials type

            for tt in trialType:
                average_resp_preCue[tt] = np.zeros((nCells,precueLength))
                average_resp_afterCue[tt] = np.zeros((nCells, aftercueLength))
                for cc in range(nCells):
                    signalMean = np.nanmean(
                        noiseInterp[:,subTrialMask[tt],cc],1)
                    average_resp_preCue[tt][cc,:] = signalMean[precueMask]
                    average_resp_afterCue[tt][cc, :] = signalMean[aftercueMask]
        # calculate pearson correlation of signals
            preCueCorr_signal = np.full((nCells,nCells), np.nan)
            afterCueCorr_signal = np.full((nCells, nCells), np.nan)
            preCueCorr_signal_trialType = np.full((nCells,nCells,3), np.nan)
            afterCueCorr_signal_trialType = np.full((nCells, nCells,3), np.nan)
            for ii in range(nCells):
                for jj in range(ii,nCells):
                    precue_tempCorr = np.zeros(len(trialType))
                    aftercue_tempCorr = np.zeros(len(trialType))
                    for idx,tt in enumerate(trialType):
                        precue_tempCorr[idx] = np.corrcoef(average_resp_preCue[tt][ii],
                                                       average_resp_preCue[tt][jj])[0,1]
                        aftercue_tempCorr[idx] = np.corrcoef(average_resp_afterCue[tt][ii],
                                                 average_resp_afterCue[tt][jj])[0,1]
                        preCueCorr_signal_trialType[ii,jj,idx] = precue_tempCorr[idx]
                        afterCueCorr_signal_trialType[ii, jj, idx] = aftercue_tempCorr[idx]
                    preCueCorr_signal[ii,jj] = np.mean(precue_tempCorr)
                    afterCueCorr_signal[ii,jj] = np.mean(aftercue_tempCorr)

            sigCorrPlot = StartSubplots(1,2)
            sigCorrPlot.ax[0].matshow(preCueCorr_signal, cmap='coolwarm',
                                  vmin=-1,vmax=1)
            sigCorrPlot.ax[0].set_title('Signal correlation, pre cue')
            cax = sigCorrPlot.ax[1].matshow(afterCueCorr_signal, cmap='coolwarm',
                                        vmin=-1,vmax=1)
            sigCorrPlot.ax[1].set_title('Signal correlation, post cue')
            cbar = sigCorrPlot.fig.colorbar(cax)
            sigCorrPlot.save_plot('Signal correlation.tiff', 'tiff', save_data_path)
            plt.close()

            subSigPlot = StartSubplots(3,2)
            for idx,tt in enumerate(trialType):
                subSigPlot.ax[idx,0].set_title('Signal ,'+tt+' pre cue')
                cax = subSigPlot.ax[idx,0].matshow(preCueCorr_signal_trialType[:,:,idx],
                                            cmap='coolwarm',
                                            vmin=-1, vmax=1)
                subSigPlot.ax[idx,1].set_title('Signal ,'+tt+' post cue')
                cax = subSigPlot.ax[idx,1].matshow(afterCueCorr_signal_trialType[:,:,idx],
                                            cmap='coolwarm',
                                            vmin=-1, vmax=1)
            cbar = subSigPlot.fig.colorbar(cax)
            subSigPlot.save_plot('Signal correlation by trial type.tiff', 'tiff', save_data_path)
            plt.close()

        # make some plots
            saveData['preCueCorr_signal'] = preCueCorr_signal
            saveData['afterCueCorr_signal'] = afterCueCorr_signal
            saveData['preCueCorr_signal_trialType'] = preCueCorr_signal_trialType
            saveData['afterCueCorr_signal_trialType'] = afterCueCorr_signal_trialType
        # noise correlation,
        # for every neuron, concatenate neural activity in for a given trial type
        # maybe also try latent factor analysis later
            preCueCorr_noise = np.full((nCells,nCells), np.nan)
            afterCueCorr_noise = np.full((nCells, nCells), np.nan)
            preCueCorr_noise_trialType = np.full((nCells,nCells,3), np.nan)
            afterCueCorr_noise_trialType = np.full((nCells, nCells,3), np.nan)
            for ii in tqdm(range(nCells)):
                for jj in range(ii,nCells):
                    matrix1 = noiseInterp[precueMask, :, ii]
                    matrix2 = noiseInterp[precueMask, :, jj]
                    coeff = np.diag(np.corrcoef(
                        matrix1, matrix2, rowvar=False)[:nTrials, nTrials:])
                    preCueCorr_noise[ii, jj] = np.nanmean([np.nanmean(coeff[subTrialMask['Hit']]),
                                              np.nanmean(coeff[subTrialMask['FA']]),
                                              np.nanmean(coeff[subTrialMask['CorRej']])])
                    for idx,tt in enumerate(trialType):
                        preCueCorr_noise_trialType[ii,jj,idx] = np.nanmean(coeff[subTrialMask[tt]])
                    matrix1 = noiseInterp[aftercueMask, :, ii]
                    matrix2 = noiseInterp[aftercueMask, :, jj]
                    coeff = np.diag(np.corrcoef(
                        matrix1, matrix2, rowvar=False)[:nTrials, nTrials:])
                    afterCueCorr_noise[ii, jj] = np.nanmean([np.nanmean(coeff[subTrialMask['Hit']]),
                                              np.nanmean(coeff[subTrialMask['FA']]),
                                              np.nanmean(coeff[subTrialMask['CorRej']])])
                    for idx, tt in enumerate(trialType):
                        afterCueCorr_noise_trialType[ii, jj, idx] = np.nanmean(coeff[subTrialMask[tt]])

            subNoisePlot = StartSubplots(3,2)
            for idx,tt in enumerate(trialType):
                subNoisePlot.ax[idx,0].set_title('Signal ,'+tt+' pre cue')
                cax = subNoisePlot.ax[idx,0].matshow(preCueCorr_noise_trialType[:,:,idx],
                                            cmap='coolwarm',
                                            vmin=-1, vmax=1)
                subNoisePlot.ax[idx,1].set_title('Signal ,'+tt+' post cue')
                cax = subNoisePlot.ax[idx,1].matshow(afterCueCorr_noise_trialType[:,:,idx],
                                            cmap='coolwarm',
                                            vmin=-1, vmax=1)
            cbar = subNoisePlot.fig.colorbar(cax)
            subNoisePlot.save_plot('Noise correlation by trial type.tiff', 'tiff', save_data_path)
            plt.close()

        # make some plots
            noiseCorrPlot = StartSubplots(1,2)
            noiseCorrPlot.ax[0].matshow(preCueCorr_noise, cmap='coolwarm',
                                    vmin=-1,vmax=1)
            noiseCorrPlot.ax[0].set_title('Noise correlation, pre cue')
            cax = noiseCorrPlot.ax[1].matshow(afterCueCorr_noise, cmap='coolwarm',
                                          vmin=-1,vmax=1)
            noiseCorrPlot.ax[1].set_title('Noise correlation, post cue')
            cbar = noiseCorrPlot.fig.colorbar(cax)
            noiseCorrPlot.save_plot('Noise correlation.tiff', 'tiff', save_data_path)
            plt.close()


        # scatter plot of noise correlation and signal correlation and fit for linear regression  - get slope
            scatterPlot = StartSubplots(1,2)
            scatterPlot.ax[0].scatter(preCueCorr_signal, preCueCorr_noise)
            scatterPlot.ax[0].set_title('Signal-noise correlation, pre cue')
            scatterPlot.ax[0].set_xlabel('Signal correlation')
            scatterPlot.ax[0].set_ylabel('Noise correlation')
            scatterPlot.ax[0].spines['left'].set_position('zero')
            scatterPlot.ax[0].spines['bottom'].set_position('zero')
            scatterPlot.ax[0].set_ylim([-0.3, 0.3])
            row, column = np.triu_indices(preCueCorr_signal.shape[0], k=1)
            preCueCorr_signal1d = preCueCorr_signal[row,column].flatten()
            preCueCorr_noise1d = preCueCorr_noise[row,column].flatten()
        #remove nans
            valid_idx=~np.isnan(preCueCorr_signal1d) & ~np.isnan(preCueCorr_noise1d)
            X = sm.add_constant(preCueCorr_signal1d[valid_idx])
            model_pre = sm.OLS(preCueCorr_noise1d[valid_idx],X)
            result_pre = model_pre.fit()
            coefficients_pre = result_pre.params
            x = np.arange(-1,1,0.1)
            scatterPlot.ax[0].plot(x, coefficients_pre[0] + coefficients_pre[1] * x, color='red', label='Linear Regression Line')

            scatterPlot.ax[1].scatter(afterCueCorr_signal, afterCueCorr_noise)
            scatterPlot.ax[1].set_title('Signal-noise correlation, post cue')
            scatterPlot.ax[1].set_xlabel('Signal correlation')
            scatterPlot.ax[1].set_ylabel('Noise correlation')
            scatterPlot.ax[1].spines['left'].set_position('zero')
            scatterPlot.ax[1].spines['bottom'].set_position('zero')
            scatterPlot.ax[1].set_ylim([-0.3, 0.3])
            row, column =  np.triu_indices(afterCueCorr_signal.shape[0], k=1)
            afterCueCorr_signal1d = afterCueCorr_signal[row,column].flatten()
            afterCueCorr_noise1d = afterCueCorr_noise[row,column].flatten()
        #remove nans
            valid_idx=~np.isnan(afterCueCorr_signal1d) & ~np.isnan(afterCueCorr_noise1d)
            X = sm.add_constant(afterCueCorr_signal1d[valid_idx])
            model_after = sm.OLS(afterCueCorr_noise1d[valid_idx],X)
            result_after = model_after.fit()
            coefficients_after = result_after.params
            x = np.arange(-1,1,0.1)
            scatterPlot.ax[1].plot(x, coefficients_after[0] + coefficients_after[1] * x, color='red', label='Linear Regression Line')
            scatterPlot.save_plot('Signal-noise correlation.tiff', 'tiff', save_data_path)
            plt.close()

            scatterPlot = StartSubplots(1,2)
            scatterPlot.ax[0].scatter(preCueCorr_signal, preCueCorr_noise)
            scatterPlot.ax[0].set_title('Signal-noise correlation, pre cue')
            scatterPlot.ax[0].set_xlabel('Signal correlation')
            scatterPlot.ax[0].set_ylabel('Noise correlation')
            scatterPlot.ax[0].spines['left'].set_position('zero')
            scatterPlot.ax[0].spines['bottom'].set_position('zero')
            scatterPlot.ax[0].set_ylim([-0.3, 0.3])
            row, column =  np.triu_indices(preCueCorr_signal.shape[0], k=1)
            preCueCorr_signal1d = preCueCorr_signal[row,column].flatten()
            preCueCorr_noise1d = preCueCorr_noise[row,column].flatten()
        #remove nans
            valid_idx=~np.isnan(preCueCorr_signal1d) & ~np.isnan(preCueCorr_noise1d)
            X = sm.add_constant(preCueCorr_signal1d[valid_idx])
            model_pre = sm.OLS(preCueCorr_noise1d[valid_idx],X)
            result_pre = model_pre.fit()
            coefficients_pre = result_pre.params
            x = np.arange(-1,1,0.1)
            scatterPlot.ax[0].plot(x, coefficients_pre[0] + coefficients_pre[1] * x, color='red', label='Linear Regression Line')

            scatterPlot.ax[1].scatter(afterCueCorr_signal, afterCueCorr_noise)
            scatterPlot.ax[1].set_title('Signal-noise correlation, post cue')
            scatterPlot.ax[1].set_xlabel('Signal correlation')
            scatterPlot.ax[1].set_ylabel('Noise correlation')
            scatterPlot.ax[1].spines['left'].set_position('zero')
            scatterPlot.ax[1].spines['bottom'].set_position('zero')
            scatterPlot.ax[1].set_ylim([-0.3, 0.3])
            row, column =  np.triu_indices(afterCueCorr_signal.shape[0], k=1)
            afterCueCorr_signal1d = afterCueCorr_signal[row,column].flatten()
            afterCueCorr_noise1d = afterCueCorr_noise[row,column].flatten()
        #remove nans
            valid_idx=~np.isnan(afterCueCorr_signal1d) & ~np.isnan(afterCueCorr_noise1d)
            X = sm.add_constant(afterCueCorr_signal1d[valid_idx])
            model_after = sm.OLS(afterCueCorr_noise1d[valid_idx],X)
            result_after = model_after.fit()
            coefficients_after = result_after.params
            x = np.arange(-1,1,0.1)
            scatterPlot.ax[1].plot(x, coefficients_after[0] + coefficients_after[1] * x, color='red', label='Linear Regression Line')
            scatterPlot.save_plot('Signal-noise correlation.tiff', 'tiff', save_data_path)
            plt.close()

            saveData['preCueCorr_noise'] = preCueCorr_noise
            saveData['afterCueCorr_noise'] = afterCueCorr_noise
            saveData['preCueCorr_noise_trialType'] = preCueCorr_noise_trialType
            saveData['afterCueCorr_noise_trialType'] = afterCueCorr_noise_trialType
            saveData['SNCorr_slope_pre'] = coefficients_pre[1]
            saveData['SNCorr_slope_after'] = coefficients_after[1]

            saveDataPath = os.path.join(save_data_path,'signal_noise '+label+'.pickle')
            with open(saveDataPath, 'wb') as pf:
                pickle.dump(saveData, pf, protocol=pickle.HIGHEST_PROTOCOL)
                pf.close()

    def dpca_elife(self, signal, var, regr_time, save_data_path):

        nVars = var.keys()
        nCells = signal.shape[0]
        nTimeBins = signal.shape[2]
        nTrials = signal.shape[1]
        trialInd = np.arange(nTrials)
        signal_pca = []
        # for the last two dimensions
        # (0, 0): nogo+nolick; (0, 1): nogo+lick
        # (1, 0): go+miss; (1,1): go+lick
        hitTrials = trialInd[np.logical_and(var['stim']==1, var['action']==1)]
        FATrials = trialInd[np.logical_and(var['stim']==0, var['action']==1)]
        CorRejTrials = trialInd[np.logical_and(var['stim']==0, var['action']==0)]

        R = np.nanmean(signal,0)
        N = signal.shape[0]
        centered_signal = R - np.nanmean(R.reshape((N,-1)),1)[:,None,None]
    def clusering(self):
        # hierachical clustering
        # cluster pure neural activity? cluster average neural activity in different trials?
        # cluster correlation coefficient?
        pass

class fluoSum:

    def __init__(self,root_dir):
        # start with the root directory, taking the dataframe of behObject

        # specify saved files
        self.root_dir = root_dir
        self.data_dir = 'Data'
        self.analysis_dir = 'Analysis'
        self.summary_dir = 'Summary'

        animals = os.listdir(os.path.join(self.root_dir, self.data_dir))
        if '.DS_Store' in animals:
            animals.remove('.DS_Store') # fxck OSX
        # initialize the dataframe
        columns = ['subject', 'age','date',
                   'beh_data_path', 'beh_analysis_dir', 'fluo_data_dir','fluo_analysis_dir',
                   'dprime']
        data_df = pd.DataFrame(columns=columns)

        # go through the files to update the dataframe
        Ind = 0
        for animal in animals:
            animal_path = os.path.join(self.root_dir, self.analysis_dir, animal)
            sessions = []
            for item in os.listdir(animal_path):
                item_path = os.path.join(animal_path, item)
                if os.path.isdir(item_path):
                    sessions.append(item)

            for session in sessions:
                matFile = glob.glob(os.path.join(animal_path, session, '*.mat'))
                separated = matFile[0].split(os.sep)
                data = pd.DataFrame({
                    'subject': animal,
                    'date': separated[-2],
                    'age': animal[0:3],
                    'beh_data_path': matFile,
                    'beh_analysis_dir': os.path.join(root_dir, self.analysis_dir,
                                                     animal, separated[-2], 'behavior'),
                    'fluo_data_dir': os.path.join(root_dir, self.data_dir,
                                                  animal, separated[-2],'suite2p','plane0'),
                    'fluo_analysis_dir': os.path.join(root_dir,  self.analysis_dir,
                                              animal, separated[-2],'fluo'),
                    'dprime': np.nan
                }, index=[Ind])


                data_df = pd.concat([data_df, data])
                Ind = Ind + 1

        self.data_df = data_df
        self.data_dict = dict()

    def process_single_session(self):
        nFiles = self.data_df.shape[0]
        self.data_df['with_imaging'] = np.zeros((nFiles))
        for f in tqdm(range(nFiles)):
            animal = self.data_df.iloc[f]['subject']
            session = self.data_df.iloc[f]['date']
            input_path = self.data_df.iloc[f]['beh_data_path']

            # get behavior data
            x = GoNogoBehaviorMat(animal, session, input_path)
            self.data_df.at[f, 'dprime'] = x.dprime
            self.data_df.at[f,'beh_object'] = x
            self.data_df.at[f, 'ifCut'] = x.ifCut
            self.data_df.at[f, 'cutoff'] = x.cutoff
            self.data_df.at[f, 'ifprobe'] = x.ifprobe
            # get fluorescent data, check for sessions without imaging

            input_folder = self.data_df.iloc[f]['fluo_data_dir']

            if not os.path.exists(input_folder):
                self.data_df.at[f,'with_imaging'] = False
            else:
                self.data_df.at[f, 'with_imaging'] =True
                if not os.path.exists(self.data_df.iloc[f]['fluo_analysis_dir']):
                    os.makedirs(self.data_df.iloc[f]['fluo_analysis_dir'])
                fluo_file = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'], animal + session + '_dff_df_file.csv')

                gn_series = Suite2pSeries(input_folder)
                gn_series.get_dFF(x, fluo_file)

            # plot the cells in field of view
            #     savefigpath = self.data_df.iloc[f]['fluo_analysis_dir']
            #     gn_series.plot_cell_location_dFF(np.arange(gn_series.neural_df.shape[1] - 1), savefigpath)
            #
            #     time_range = [0,600]
            #     gn_series.plot_cell_dFF(time_range, savefigpath)
            # create a fluorescent analysis object
                beh_file = os.path.join(self.data_df.iloc[f]['beh_analysis_dir'], 'behAnalysis.pickle')
                analysis = fluoAnalysis(beh_file, fluo_file)
                analysis.align_fluo_beh(self.data_df.iloc[f]['fluo_analysis_dir'])

                self.data_df.at[f, 'fluo_analysis'] = analysis
                self.data_df.at[f, 'fluo_raw'] = gn_series
                self.data_df.at[f, 'n_cells'] = gn_series.neural_df.shape[1]-1
                self.data_df.at[f, 'numStim'] = len(np.unique(analysis.beh['sound_num']))

    def cell_plots(self):
        # plot cell location and PSTH of every cell
        nFiles = self.data_df.shape[0]
        for f in tqdm(range(nFiles)):
            if self.data_df['with_imaging'][f]:
                analysis = self.data_df['fluo_analysis'][f]
                savefigpath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],'cells-combined-cue')
                if not os.path.exists(savefigpath):
                    os.makedirs(savefigpath)
                analysis.plot_dFF(savefigpath)

    def MLR_session(self):
        # run multiple linear regression session by session
        n_predictors = 14
        labels = ['s(n+1)', 's(n)', 's(n-1)', 'c(n+1)', 'c(n)', 'c(n-1)',
                  'r(n+1)', 'r(n)', 'r(n-1)', 'x(n+1)', 'x(n)', 'x(n-1)', 'speed', 'lick']
        nFiles = self.data_df.shape[0]
        for f in tqdm(range(nFiles)):
            analysis = self.data_df['fluo_analysis'][f]
            gn_series = self.data_df['fluo_raw'][f]
            saveDataPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],'MLR')
            if not os.path.exists(saveDataPath):
                os.makedirs(saveDataPath)

            saveTbTFile = os.path.join(saveDataPath, 'trialbytrialVar.pickle')
            saveMatFile = os.path.join(saveDataPath, 'trialbytrialVar.mat')
            import scipy.io

            if not os.path.exists(saveMatFile):
                X, y, regr_time = analysis.linear_model(n_predictors)
                tbtVar = {}
                tbtVar['X'] = X
                tbtVar['y'] = y
                tbtVar['regr_time'] = regr_time

                scipy.io.savemat(saveMatFile, tbtVar)
                #
            # save X and y
                with open(saveTbTFile, 'wb') as pf:
                    pickle.dump(tbtVar, pf, protocol=pickle.HIGHEST_PROTOCOL)
                    pf.close()
            else:
                # load the saved results
                with open(saveTbTFile, 'rb') as f:
                    tbtVar = pickle.load(f)
                    X = tbtVar['X']
                    y = tbtVar['y']
                    regr_time = tbtVar['regr_time']
            # run MLR; omit trials without fluorescent data

            saveDataFile = os.path.join(saveDataPath, 'MLRResult.pickle')
            if not os.path.exists(saveDataFile):
                MLRResult = analysis.linear_regr(X[:, 1:-2, :], y[:, 1:-2, :], regr_time, saveDataFile)
                analysis.plotMLRResult(MLRResult, labels, gn_series, saveDataPath)
            else:

                print('MLR done!')

    def MLR_summary(self, group_adt, group_juv, group_label):
        # data structure of MLR results
        # MLR['coeff'] npred * t * nCell coefficient
        # ['pval']  npred * t * nCell p-value
        # ['r2']  t * nCell  r-square
        # ['regr_time']  t: time step
        # ['sigCells'] a dictionary of cell ID that are significant for 'stimulus', 'action', 'reward'

        """
        what to do with sigCells?
        """
        n_predictors = 14
        labels = ['s(n+1)', 's(n)', 's(n-1)', 'c(n+1)', 'c(n)', 'c(n-1)',
                  'r(n+1)', 'r(n)', 'r(n-1)', 'x(n+1)', 'x(n)', 'x(n-1)', 'speed', 'lick']

        # load adult group
        nFiles = group_adt.shape[0]
        # read coefficient, sig cells, neurons with
        for f in tqdm(range(nFiles)):
            saveDataPath = os.path.join(group_adt.iloc[f]['fluo_analysis_dir'], 'MLR')
            saveDataFile = os.path.join(saveDataPath, 'MLRResult.pickle')
            # load the pickle file
            with open(saveDataFile, 'rb') as file:
                MLRResult = pickle.load(file)

            # initialize variables
            if f==0:
                regr_time = MLRResult['regr_time']
                MLRSummary_ADT = {}
                MLRSummary_ADT['coeff'] = np.empty((n_predictors,len(regr_time),
                                                    0))
                MLRSummary_ADT['sig'] = np.empty((n_predictors,len(regr_time),
                                                    0))
            # calculate fraction of neurons that are significant
            pThresh = 0.01
            nCell = MLRResult['coeff'].shape[2]
            fracSig = np.sum(MLRResult['pval'][:, :, :] < pThresh, 2) / nCell
            MLRSummary_ADT['coeff'] = np.concatenate((MLRSummary_ADT['coeff'],
                                                        MLRResult['coeff']), 2)
            MLRSummary_ADT['sig'] = np.concatenate((MLRSummary_ADT['sig'],
                                                        fracSig[:,:,np.newaxis]), 2)

        # load juv group
        nFiles = group_juv.shape[0]
        # read coefficient, sig cells, neurons with
        for f in tqdm(range(nFiles)):
            saveDataPath = os.path.join(group_juv.iloc[f]['fluo_analysis_dir'], 'MLR')
            saveDataFile = os.path.join(saveDataPath, 'MLRResult.pickle')
            # load the pickle file
            with open(saveDataFile, 'rb') as file:
                MLRResult = pickle.load(file)

            if f==0:
                MLRSummary_JUV = {}
                MLRSummary_JUV['coeff'] = np.empty((n_predictors, len(regr_time),
                                            0))
                MLRSummary_JUV['sig'] = np.empty((n_predictors, len(regr_time),
                                          0))
            pThresh = 0.01
            nCell = MLRResult['coeff'].shape[2]
            fracSig = np.sum(MLRResult['pval'][:, :, :] < pThresh, 2) / nCell
            MLRSummary_JUV['coeff'] = np.concatenate((MLRSummary_JUV['coeff'],
                                                  MLRResult['coeff']), 2)
            MLRSummary_JUV['sig'] = np.concatenate((MLRSummary_JUV['sig'],
                                                fracSig[:, :, np.newaxis]), 2)

            # determine p-value
            # rank-sum test
        # plot the summary results
        savefigpath = os.path.join(self.root_dir, self.summary_dir, 'fluo')
        if not os.path.exists(savefigpath):
            os.makedirs(savefigpath)
        self.plot_MLR_summary(MLRSummary_ADT, MLRSummary_JUV, regr_time, labels, savefigpath, group_label)

    def plot_MLR_summary(self, MLR_ADT, MLR_JUV, regr_time, labels, saveFigPath, group_label):
        # get the average coefficient plot and fraction of significant neurons
        # separatte ADT and JUV
        # separate number of exposed cues (2-4 / 8-16
        matplotlib.use('Qt5Agg')
        varList = labels
            # average coefficient
        nPredictors = len(varList)

        #adtColor = (255 / 255, 189 / 255, 53 / 255)
        #juvColor = (63 / 255, 167 / 255, 150 / 255)
        adtColor = 'blue'
        juvColor = 'red'
        coeffPlot = StartSubplots(4, 4, ifSharey=True)

        maxY = 0
        minY = 0
        for n in range(nPredictors-2):
            ## plot ADT animals
            tempBoot = bootstrap(MLR_ADT['coeff'][n, :, :], 1, 1000)
            tempMax = max(tempBoot['bootHigh'])
            tempMin = min(tempBoot['bootLow'])
            if tempMax > maxY:
                maxY = tempMax
            if tempMin < minY:
                minY = tempMin

            plotIdx = np.logical_and(regr_time>-2,regr_time<2)
            coeffPlot.ax[n // 4, n % 4].plot(regr_time[plotIdx], tempBoot['bootAve'][plotIdx], c=adtColor)
            coeffPlot.ax[n // 4, n % 4].fill_between(regr_time[plotIdx], tempBoot['bootLow'][plotIdx],
                                                         tempBoot['bootHigh'][plotIdx],
                                                         alpha=0.2, color=adtColor)

            tempBoot = bootstrap(MLR_JUV['coeff'][n, :, :], 1, 1000)
            tempMax = max(tempBoot['bootHigh'])
            tempMin = min(tempBoot['bootLow'])
            if tempMax > maxY:
                maxY = tempMax
            if tempMin < minY:
                minY = tempMin
            coeffPlot.ax[n // 4, n % 4].plot(regr_time[plotIdx], tempBoot['bootAve'][plotIdx], c=juvColor)
            coeffPlot.ax[n // 4, n % 4].fill_between(regr_time[plotIdx], tempBoot['bootLow'][plotIdx],
                                                         tempBoot['bootHigh'][plotIdx],
                                                         alpha=0.2, color=juvColor)

            coeffPlot.ax[n // 4, n % 4].set_title(varList[n])

            # test significance
        coeffPlot.fig.suptitle('Average coefficient group '+ group_label)
        plt.ylim((minY, maxY))
        plt.show()
        coeffPlot.save_plot('Average coefficient group ' + group_label + '.tif', 'tiff', saveFigPath)
        coeffPlot.save_plot('Average coefficient group ' + group_label + '.svg', 'svg', saveFigPath)

        # fraction of significant neurons
        sigPlot = StartSubplots(3, 4, ifSharey=True)

        # binomial test to determine signficance

        for n in range(nPredictors-2):

            tempBoot = bootstrap(MLR_ADT['sig'][n, :, :], 1, 1000)

            sigPlot.ax[n // 4, n % 4].plot(regr_time[plotIdx], tempBoot['bootAve'][plotIdx],
                                           c=adtColor)
            sigPlot.ax[n // 4, n % 4].fill_between(regr_time[plotIdx], tempBoot['bootLow'][plotIdx],
                                                         tempBoot['bootHigh'][plotIdx],
                                                         alpha=0.2, color=adtColor)

            tempBoot = bootstrap(MLR_JUV['sig'][n, :, :], 1, 1000)

            sigPlot.ax[n // 4, n % 4].plot(regr_time[plotIdx], tempBoot['bootAve'][plotIdx],
                                           c=juvColor)
            sigPlot.ax[n // 4, n % 4].fill_between(regr_time[plotIdx], tempBoot['bootLow'][plotIdx],
                                                   tempBoot['bootHigh'][plotIdx],
                                                   alpha=0.2, color=juvColor)

            sigPlot.ax[n // 4, n % 4].set_title(varList[n])

            if n // 4 == 0:
                sigPlot.ax[n // 4, n % 4].set_ylabel('Fraction of sig')
            if n > 8:
                sigPlot.ax[n // 4, n % 4].set_xlabel('Time from cue (s)')
                # plot the signifcance bar
 #           dt = np.mean(np.diff(regr_time))
 #           newTime = regr_time[plotIdx]
 #           for tt in range(len(newTime)):
 #               statistic, p_MWtest = mannwhitneyu(MLR_JUV['sig'][n, tt, :],
 #                                                  MLR_ADT['sig'][n, tt, :])
 #               if p_MWtest < 0.01:
 #                   sigPlot.ax[n // 4, n % 4].plot(newTime[tt] + dt * np.array([-0.5, 0.5]),
 #                                                      [0.5, 0.5], color=(255 / 255, 189 / 255, 53 / 255), linewidth=5)
        sigPlot.fig.suptitle('Fraction of significant neurons group ' + group_label)
        plt.ylim((0, 0.2))
        plt.show()
        sigPlot.save_plot('Fraction of significant neurons group ' + group_label + '.tif', 'tiff', saveFigPath)
        sigPlot.save_plot('Fraction of significant neurons group ' + group_label + '.svg', 'svg', saveFigPath)

    def decoding_session(self, n_predictors):
        nFiles = self.data_df.shape[0]
        for f in tqdm(range(nFiles)):
            if self.data_df['with_imaging'][f]:
                analysis = self.data_df['fluo_analysis'][f]
                gn_series = self.data_df['fluo_raw'][f]
                saveDataPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],
                                        'MLR')
                if not os.path.exists(saveDataPath):
                    os.makedirs(saveDataPath)

                saveTbTFile = os.path.join(saveDataPath, 'trialbytrialVar.pickle')
                if not os.path.exists(saveTbTFile):
                    X, y, regr_time = analysis.linear_model(n_predictors)
                    tbtVar = {}
                    tbtVar['X'] = X
                    tbtVar['y'] = y
                    tbtVar['regr_time'] = regr_time

            # save X and y
                    with open(saveTbTFile, 'wb') as pf:
                        pickle.dump(tbtVar, pf, protocol=pickle.HIGHEST_PROTOCOL)
                        pf.close()
                else:
                # load the saved results
                    with open(saveTbTFile, 'rb') as pf:
                        tbtVar = pickle.load(pf)
                        X = tbtVar['X']
                        y = tbtVar['y']
                        regr_time = tbtVar['regr_time']
            # run decoding; omit trials without fluorescent data
                saveDataPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],
                                        'decoding')
                if not os.path.exists(saveDataPath):
                    os.makedirs(saveDataPath)
                saveDataFile = os.path.join(saveDataPath, 'decodingResult.pickle')

                if not os.path.exists(saveDataFile):
                    decodeVar = {}

                    decodeVar['action'] = X[4, :, 0]
    #                decodeVar['outcome'] = X[7,:,0]
                    # dummy-code: 1: hit, 0: Correj; -1: FA

                    decodeVar['stimulus'] = np.array(
                            [np.nan if np.isnan(x)
                            else np.int(x)
                            for x in analysis.beh['sound_num']])
                    decodeVar['trialType'] = np.array(
                            [np.nan if np.isnan(x)
                            else np.int(x)
                            for x in analysis.beh['trialType']])
                    decodeSig = y

                    trialMask = decodeVar['stimulus'] <=8
                    # check false alarm trials, and probe trials
                    subTrialMask = {}
                    subTrialMask['FA'] = analysis.beh['trialType'] == -1
                    subTrialMask['probe'] = decodeVar['stimulus'] > 8
                    subTrialMask['Hit'] = analysis.beh['trialType'] == 2
                    subTrialMask['CorRej'] = analysis.beh['trialType'] == 0
                    # stimulus 1-4: 1
                    # stimulus 5-8: 0
                    # stimulus 9-12；2
                    # stimulus 13-16: 3
                    tempSti = np.zeros(len(decodeVar['stimulus']))
                    for ss in range(len(decodeVar['stimulus'])):
                        if decodeVar['stimulus'][ss] <= 4:
                            tempSti[ss] = 1
                        elif decodeVar['stimulus'][ss] > 4 and decodeVar['stimulus'][ss] <= 8:
                            tempSti[ss] = 0
                        elif decodeVar['stimulus'][ss] > 8 and decodeVar['stimulus'][ss] <= 12:
                            tempSti[ss] = 1
                        elif decodeVar['stimulus'][ss] > 12:
                            tempSti[ss] = 0
                        # trialType
                    decodeVar['stimulus'] = tempSti
                    decodeVar['trialType'][decodeVar['trialType']==2] = 1

                    classifier = "SVC"
                    varList = ['stimulus', 'action', 'trialType']

                    analysis.decoding(decodeSig, decodeVar, varList,
                                      trialMask,subTrialMask, classifier,
                                      regr_time, saveDataFile)
    #                else:
    #                    print('Decoding done!')
                    # ensemble size analysis

                    analysis.decode_analysis(gn_series, saveDataFile, saveDataPath)

                    # removing trials that has too large running speed
                saveDataFile = os.path.join(saveDataPath, 'decodingResult_running_1.5.pickle')

                if not os.path.exists(saveDataFile):
                    decodeVar = {}

                    decodeVar['stimulus'] = np.array(
                            [np.nan if np.isnan(x)
                             else np.int(x)
                             for x in analysis.beh['sound_num']])
                    decodeSig = y

                    trialMask = decodeVar['stimulus'] <= 8
                        # check false alarm trials, and probe trials
                    subTrialMask = {}
                    subTrialMask['FA'] = analysis.beh['trialType'] == -1
                    subTrialMask['probe'] = decodeVar['stimulus'] > 8
                    subTrialMask['Hit'] = analysis.beh['trialType'] == 2
                    subTrialMask['CorRej'] = analysis.beh['trialType'] == 0
                        # stimulus 1-4: 1
                        # stimulus 5-8: 0
                        # stimulus 9-12；2
                        # stimulus 13-16: 3
                    tempSti = np.zeros(len(decodeVar['stimulus']))
                    for ss in range(len(decodeVar['stimulus'])):
                        if decodeVar['stimulus'][ss] <= 4:
                            tempSti[ss] = 1
                        elif decodeVar['stimulus'][ss] > 4 and decodeVar['stimulus'][ss] <= 8:
                            tempSti[ss] = 0
                        elif decodeVar['stimulus'][ss] > 8 and decodeVar['stimulus'][ss] <= 12:
                            tempSti[ss] = 1
                        elif decodeVar['stimulus'][ss] > 12:
                            tempSti[ss] = 0
                            # trialType
                    decodeVar['stimulus'] = tempSti

                    classifier = "SVC"
                    varList = ['stimulus']

                    # examine the average running speed
                    ave_running = [np.nanmean(analysis.beh['running_speed'][i]) for i in range(len(trialMask))]
                    run_threshold = 1.5
                    trials_include = np.array(ave_running)<run_threshold

                    #combine with trialMask and subtrial Mask
                    new_trialMask = np.logical_and(trials_include,trialMask)
                    new_subTrialMask ={}
                    for key in subTrialMask.keys():
                        new_subTrialMask[key] = np.logical_and(subTrialMask[key],trials_include)

                    analysis.decoding(decodeSig, decodeVar, varList,
                                          new_trialMask, new_subTrialMask, classifier,
                                          regr_time, saveDataFile)
                        #                else:
                        #                    print('Decoding done!')
                        # ensemble size analysis

                    #analysis.decode_analysis(gn_series, saveDataFile, saveDataPath)

    def decoding_summary(self, group_adt, group_juv, group_label):

        # load data from adult animal
        nFiles = group_adt.shape[0]
        #n_ctrl = 20
        # read coefficient, sig cells, neurons with
        for f in tqdm(range(nFiles)):
            saveDataPath = os.path.join(group_adt.iloc[f]['fluo_analysis_dir'], 'decoding')
            saveDataFile = os.path.join(saveDataPath, 'decodingResult.pickle')
            # load the pickle file
            with open(saveDataFile, 'rb') as file:
                decodingResult = pickle.load(file)

            # initialize variables
            if f==0:
                decodingVars = decodingResult['var']
                regr_time = decodingResult['time']
                decodingSummary_ADT = {}
                for var in decodingVars:
                    decodingSummary_ADT[var] = {}
                    decodingSummary_ADT[var]['accuracy'] = np.empty((len(regr_time),
                                                    0))
                    decodingSummary_ADT[var]['ctrl_accuracy'] = np.empty((len(regr_time),
                                                    0))
                # load predictions based on different trial types
                    decodingSummary_ADT[var]['prediction_accuracy'] = {}
                    decodingSummary_ADT[var]['prediction_accuracy_ctrl'] = {}
                    for key in decodingResult[var]['prediction_accuracy'].keys():
                        decodingSummary_ADT[var]['prediction_accuracy'][key] = np.empty((len(regr_time), 0))
                        decodingSummary_ADT[var]['prediction_accuracy_ctrl'][key] = np.empty((len(regr_time), 0))

                    # load ensemble size result
                    if var=='stimulus':
                        maxNeurons = 300
                        nNeurons = np.arange(0,300,10)
                        decodingSummary_ADT[var]['ensemble_accuracy'] = np.full((len(nNeurons),nFiles), np.nan)
                        decodingSummary_ADT[var]['ensemble_size'] = decodingResult[var]['decode_realSize']['nNeurons']
                        decodingSummary_ADT[var]['ensemble_ctrl'] = np.full((len(nNeurons),nFiles), np.nan)


            for var in decodingVars:
            # calculate fraction of neurons that are significant
                decodingSummary_ADT[var]['accuracy'] = np.concatenate((decodingSummary_ADT[var]['accuracy'],
                                                        decodingResult[var]['accuracy'][:, np.newaxis]), 1)
                decodingSummary_ADT[var]['ctrl_accuracy'] = np.concatenate((decodingSummary_ADT[var]['ctrl_accuracy'],
                                                        decodingResult[var]['ctrl_accuracy'][:, np.newaxis]), 1)
                #get the average decoding accuracy between 1-2 second after cue
                if var == "stimulus":
                    timeMask = np.logical_and(regr_time>=1, regr_time<2)
                    ave_accuracy = np.nanmean(decodingResult[var]['decode_realSize']['accuracy'][timeMask,:],
                                              0)
                    ave_ctrl = np.nanmean(decodingResult[var]['decode_realSize']['ctrl_accuracy'][timeMask,:],
                                          0)
                    decodingSummary_ADT[var]['ensemble_accuracy'][:len(ave_accuracy),f] = ave_accuracy
                    decodingSummary_ADT[var]['ensemble_ctrl'][:len(ave_accuracy), f] = ave_ctrl

                for key in decodingResult[var]['prediction_accuracy'].keys():
                    decodingSummary_ADT[var]['prediction_accuracy'][key] = np.concatenate(
                            (decodingSummary_ADT[var]['prediction_accuracy'][key],
                             decodingResult[var]['prediction_accuracy'][key][:,np.newaxis]), 1)
                    decodingSummary_ADT[var]['prediction_accuracy_ctrl'][key] = np.concatenate(
                            (decodingSummary_ADT[var]['prediction_accuracy_ctrl'][key],
                             decodingResult[var]['prediction_accuracy_ctrl'][key][:, np.newaxis]), 1)

        # load juv group
        nFiles = group_juv.shape[0]
        # read coefficient, sig cells, neurons with
        for f in tqdm(range(nFiles)):
            saveDataPath = os.path.join(group_juv.iloc[f]['fluo_analysis_dir'], 'decoding')
            saveDataFile = os.path.join(saveDataPath, 'decodingResult.pickle')
            # load the pickle file
            with open(saveDataFile, 'rb') as file:
                decodingResult = pickle.load(file)

            # initialize variables
            if f==0:
                decodingVars = decodingResult['var']
                regr_time = decodingResult['time']
                decodingSummary_JUV = {}
                for var in decodingVars:
                    decodingSummary_JUV[var] = {}
                    decodingSummary_JUV[var]['accuracy'] = np.empty((len(regr_time),
                                                    0))
                    decodingSummary_JUV[var]['ctrl_accuracy'] = np.empty((len(regr_time),
                                                    0))

                    if var == "stimulus":
                        maxNeurons = 300
                        nNeurons = np.arange(0,300,10)
                        decodingSummary_JUV[var]['ensemble_accuracy'] = np.full((len(nNeurons),nFiles), np.nan)
                        decodingSummary_JUV[var]['ensemble_size'] = decodingResult[var]['decode_realSize']['nNeurons']
                        decodingSummary_JUV[var]['ensemble_ctrl'] = np.full((len(nNeurons),nFiles), np.nan)

                # load predictions based on different trial types
                    decodingSummary_JUV[var]['prediction_accuracy'] = {}
                    decodingSummary_JUV[var]['prediction_accuracy_ctrl'] = {}
                    for key in decodingResult[var]['prediction_accuracy'].keys():

                        decodingSummary_JUV[var]['prediction_accuracy'][key] = np.empty((len(regr_time), 0))
                        decodingSummary_JUV[var]['prediction_accuracy_ctrl'][key] = np.empty((len(regr_time), 0))

            for var in decodingVars:
            # calculate fraction of neurons that are significant
                decodingSummary_JUV[var]['accuracy'] = np.concatenate((decodingSummary_JUV[var]['accuracy'],
                                                        decodingResult[var]['accuracy'][:, np.newaxis]), 1)
                decodingSummary_JUV[var]['ctrl_accuracy'] = np.concatenate((decodingSummary_JUV[var]['ctrl_accuracy'],
                                                        decodingResult[var]['ctrl_accuracy'][:, np.newaxis]), 1)
                if var == "stimulus":
                    timeMask = np.logical_and(regr_time >= 1, regr_time < 2)
                    ave_accuracy = np.nanmean(decodingResult[var]['decode_realSize']['accuracy'][timeMask, :],
                                          0)
                    ave_ctrl = np.nanmean(decodingResult[var]['decode_realSize']['ctrl_accuracy'][timeMask, :],
                                      0)
                    decodingSummary_JUV[var]['ensemble_accuracy'][:len(ave_accuracy), f] = ave_accuracy
                    decodingSummary_JUV[var]['ensemble_ctrl'][:len(ave_accuracy), f] = ave_ctrl

                for key in decodingResult[var]['prediction_accuracy'].keys():
                    decodingSummary_JUV[var]['prediction_accuracy'][key] = np.concatenate(
                        (decodingSummary_JUV[var]['prediction_accuracy'][key],
                         decodingResult[var]['prediction_accuracy'][key][:,np.newaxis]), 1)
                    decodingSummary_JUV[var]['prediction_accuracy_ctrl'][key] = np.concatenate(
                        (decodingSummary_JUV[var]['prediction_accuracy_ctrl'][key],
                         decodingResult[var]['prediction_accuracy_ctrl'][key][:, np.newaxis]), 1)

        # plot the summary results
        savefigpath = os.path.join(self.root_dir, self.summary_dir, 'fluo')
        if not os.path.exists(savefigpath):
            os.makedirs(savefigpath)

        group_label = 'late'
        savefigpath = os.path.join(self.root_dir, self.summary_dir, 'fluo')
        self.plot_decoding_summary(decodingSummary_ADT, decodingSummary_JUV, regr_time, savefigpath, group_label)

    def decoding_summary_running(self, group_adt, group_juv,group_label):

        # load data from adult animal
        nFiles = group_adt.shape[0]
        #n_ctrl = 20
        # read coefficient, sig cells, neurons with
        for f in tqdm(range(nFiles)):
            saveDataPath = os.path.join(group_adt.iloc[f]['fluo_analysis_dir'], 'decoding')
            saveDataFile = os.path.join(saveDataPath, 'decodingResult_running.pickle')
            # load the pickle file
            if os.path.exists(saveDataFile):
                with open(saveDataFile, 'rb') as file:
                    decodingResult = pickle.load(file)

            # initialize variables
                if f==0:
                    decodingVars = decodingResult['var']
                    regr_time = decodingResult['time']
                    decodingSummary_ADT = {}
                    for var in decodingVars:
                        decodingSummary_ADT[var] = {}
                        decodingSummary_ADT[var]['accuracy'] = np.empty((len(regr_time),
                                                    0))
                        decodingSummary_ADT[var]['ctrl_accuracy'] = np.empty((len(regr_time),
                                                    0))
                # load predictions based on different trial types
                        decodingSummary_ADT[var]['prediction_accuracy'] = {}
                        decodingSummary_ADT[var]['prediction_accuracy_ctrl'] = {}
                        for key in decodingResult[var]['prediction_accuracy'].keys():
                            decodingSummary_ADT[var]['prediction_accuracy'][key] = np.empty((len(regr_time), 0))
                            decodingSummary_ADT[var]['prediction_accuracy_ctrl'][key] = np.empty((len(regr_time), 0))

                    # load ensemble size result
                        if var=='stimulus':
                            maxNeurons = 300
                            nNeurons = np.arange(0,300,10)
                            decodingSummary_ADT[var]['ensemble_accuracy'] = np.full((len(nNeurons),nFiles), np.nan)
                            decodingSummary_ADT[var]['ensemble_size'] = decodingResult[var]['decode_realSize']['nNeurons']
                            decodingSummary_ADT[var]['ensemble_ctrl'] = np.full((len(nNeurons),nFiles), np.nan)


                for var in decodingVars:
            # calculate fraction of neurons that are significant
                    decodingSummary_ADT[var]['accuracy'] = np.concatenate((decodingSummary_ADT[var]['accuracy'],
                                                        decodingResult[var]['accuracy'][:, np.newaxis]), 1)
                    decodingSummary_ADT[var]['ctrl_accuracy'] = np.concatenate((decodingSummary_ADT[var]['ctrl_accuracy'],
                                                        decodingResult[var]['ctrl_accuracy'][:, np.newaxis]), 1)
                #get the average decoding accuracy between 1-2 second after cue
                    if var == "stimulus":
                        timeMask = np.logical_and(regr_time>=1, regr_time<2)
                        ave_accuracy = np.nanmean(decodingResult[var]['decode_realSize']['accuracy'][timeMask,:],
                                              0)
                        ave_ctrl = np.nanmean(decodingResult[var]['decode_realSize']['ctrl_accuracy'][timeMask,:],
                                          0)
                        decodingSummary_ADT[var]['ensemble_accuracy'][:len(ave_accuracy),f] = ave_accuracy
                        decodingSummary_ADT[var]['ensemble_ctrl'][:len(ave_accuracy), f] = ave_ctrl

                    for key in decodingResult[var]['prediction_accuracy'].keys():
                        decodingSummary_ADT[var]['prediction_accuracy'][key] = np.concatenate(
                            (decodingSummary_ADT[var]['prediction_accuracy'][key],
                             decodingResult[var]['prediction_accuracy'][key][:,np.newaxis]), 1)
                        decodingSummary_ADT[var]['prediction_accuracy_ctrl'][key] = np.concatenate(
                            (decodingSummary_ADT[var]['prediction_accuracy_ctrl'][key],
                             decodingResult[var]['prediction_accuracy_ctrl'][key][:, np.newaxis]), 1)

        # load juv group
        nFiles = group_juv.shape[0]
        # read coefficient, sig cells, neurons with
        for f in tqdm(range(nFiles)):
            saveDataPath = os.path.join(group_juv.iloc[f]['fluo_analysis_dir'], 'decoding')
            saveDataFile = os.path.join(saveDataPath, 'decodingResult_running.pickle')
            # load the pickle file
            if os.path.exists(saveDataFile):
                with open(saveDataFile, 'rb') as file:
                    decodingResult = pickle.load(file)

                # initialize variables
                if f==0:
                    decodingVars = decodingResult['var']
                    regr_time = decodingResult['time']
                    decodingSummary_JUV = {}
                    for var in decodingVars:
                        decodingSummary_JUV[var] = {}
                        decodingSummary_JUV[var]['accuracy'] = np.empty((len(regr_time),
                                                        0))
                        decodingSummary_JUV[var]['ctrl_accuracy'] = np.empty((len(regr_time),
                                                        0))

                        if var == "stimulus":
                            maxNeurons = 300
                            nNeurons = np.arange(0,300,10)
                            decodingSummary_JUV[var]['ensemble_accuracy'] = np.full((len(nNeurons),nFiles), np.nan)
                            decodingSummary_JUV[var]['ensemble_size'] = decodingResult[var]['decode_realSize']['nNeurons']
                            decodingSummary_JUV[var]['ensemble_ctrl'] = np.full((len(nNeurons),nFiles), np.nan)

                    # load predictions based on different trial types
                        decodingSummary_JUV[var]['prediction_accuracy'] = {}
                        decodingSummary_JUV[var]['prediction_accuracy_ctrl'] = {}
                        for key in decodingResult[var]['prediction_accuracy'].keys():

                            decodingSummary_JUV[var]['prediction_accuracy'][key] = np.empty((len(regr_time), 0))
                            decodingSummary_JUV[var]['prediction_accuracy_ctrl'][key] = np.empty((len(regr_time), 0))

                for var in decodingVars:
                # calculate fraction of neurons that are significant
                    decodingSummary_JUV[var]['accuracy'] = np.concatenate((decodingSummary_JUV[var]['accuracy'],
                                                            decodingResult[var]['accuracy'][:, np.newaxis]), 1)
                    decodingSummary_JUV[var]['ctrl_accuracy'] = np.concatenate((decodingSummary_JUV[var]['ctrl_accuracy'],
                                                            decodingResult[var]['ctrl_accuracy'][:, np.newaxis]), 1)
                    if var == "stimulus":
                        timeMask = np.logical_and(regr_time >= 1, regr_time < 2)
                        ave_accuracy = np.nanmean(decodingResult[var]['decode_realSize']['accuracy'][timeMask, :],
                                              0)
                        ave_ctrl = np.nanmean(decodingResult[var]['decode_realSize']['ctrl_accuracy'][timeMask, :],
                                          0)
                        decodingSummary_JUV[var]['ensemble_accuracy'][:len(ave_accuracy), f] = ave_accuracy
                        decodingSummary_JUV[var]['ensemble_ctrl'][:len(ave_accuracy), f] = ave_ctrl

                    for key in decodingResult[var]['prediction_accuracy'].keys():
                        decodingSummary_JUV[var]['prediction_accuracy'][key] = np.concatenate(
                            (decodingSummary_JUV[var]['prediction_accuracy'][key],
                             decodingResult[var]['prediction_accuracy'][key][:,np.newaxis]), 1)
                        decodingSummary_JUV[var]['prediction_accuracy_ctrl'][key] = np.concatenate(
                            (decodingSummary_JUV[var]['prediction_accuracy_ctrl'][key],
                             decodingResult[var]['prediction_accuracy_ctrl'][key][:, np.newaxis]), 1)

        # plot the summary results
        savefigpath = os.path.join(self.root_dir, self.summary_dir, 'fluo')
        if not os.path.exists(savefigpath):
            os.makedirs(savefigpath)

        group_label = 'late_running'
        savefigpath = os.path.join(self.root_dir, self.summary_dir, 'fluo')
        self.plot_decoding_summary(decodingSummary_ADT, decodingSummary_JUV, regr_time, savefigpath, group_label)

    def plot_decoding_summary(self, decoding_ADT, decoding_JUV, regr_time, saveFigPath, group_label):
        matplotlib.use('Qt5Agg')

        #adtColor = (255 / 255, 189 / 255, 53 / 255)
        #juvColor = (63 / 255, 167 / 255, 150 / 255)
        adtColor = (83/255,187/255,244/255)
        juvColor = (255/255,67/255,46/255)

        varList = decoding_ADT.keys()

        """ plot average decoding and by trial types"""
        subplot_label = ['accuracy', 'Hit', 'FA',  'CorRej','Probe']
        for var in varList:
            decodingPlot = StartSubplots(2, 3, ifSharey=True)
            plt.ylim((0.3, 1.05))
            decodingPlot.fig.suptitle('Average average decoding accuracy ' + var + '_' + group_label)
            for idx, l in enumerate(subplot_label):
                ## plot ADT animals
                decodingPlot.ax[idx // 3, idx % 3].set_title(l)

                if l == 'accuracy':
                    tempBoot = bootstrap(decoding_ADT[var][l], 1, 1000)

                    tempCtrlBoot = bootstrap(decoding_ADT[var]['ctrl_accuracy'], 1, 1000)

                elif l == 'Probe':
                    tempBoot = bootstrap(decoding_ADT[var]['prediction_accuracy'][l], 1, 1000)
                    tempCtrlBoot = bootstrap(decoding_ADT[var]['prediction_accuracy_ctrl'][l], 1, 1000)

                # for Hit, FA, CorRej trials
                else:
                    tempBoot = bootstrap(decoding_ADT[var]['prediction_accuracy'][l], 1, 1000)
                    tempCtrlBoot = bootstrap(decoding_ADT[var]['prediction_accuracy_ctrl'][l], 1, 1000)

                # plot data and ctrl
                plotIdx = np.logical_and(regr_time > -2, regr_time < 2)
                decodingPlot.ax[idx // 3, idx % 3].plot(regr_time[plotIdx], tempBoot['bootAve'][plotIdx],
                                                            c=adtColor)
                decodingPlot.ax[idx // 3, idx % 3].fill_between(regr_time[plotIdx], tempBoot['bootLow'][plotIdx],
                                                                    tempBoot['bootHigh'][plotIdx],
                                                                    alpha=0.2, color=adtColor)

                decodingPlot.ax[idx // 3, idx % 3].plot(regr_time[plotIdx], tempCtrlBoot['bootAve'][plotIdx],
                                                        c=(0.7, 0.7, 0.7))
                decodingPlot.ax[idx // 3, idx % 3].fill_between(regr_time[plotIdx], tempCtrlBoot['bootLow'][plotIdx],
                                                                tempCtrlBoot['bootHigh'][plotIdx],
                                                                alpha=0.2, color=(0.7, 0.7, 0.7))

                # plot JUV
                decodingPlot.ax[idx // 3, idx % 3].set_title(l)

                if l == 'accuracy':
                    tempBoot = bootstrap(decoding_JUV[var][l], 1, 1000)

                    tempCtrlBoot = bootstrap(decoding_JUV[var]['ctrl_accuracy'], 1, 1000)

                    # determine signficant level
                    # bonferroni test or FDR correction
                    p_list = []
                    dt = np.mean(np.diff(regr_time))
                    for tt in range(len(regr_time)):
                        statistic, p_MWtest = mannwhitneyu(decoding_JUV[var][l][tt,:],
                                                           decoding_ADT[var][l][tt,:])
                        p_list.append(p_MWtest)
                    q_values = multipletests(p_list, method='fdr_bh')[1]
                        # Set your desired FDR level (e.g., 0.05)
                    FDR_threshold = 0.05

                        # Identify significant results
                    significant_results = [p < FDR_threshold for p in q_values]
                    for tt in range(len(regr_time)):
                        if significant_results[tt]:
                            decodingPlot.ax[idx // 3, idx % 3].plot(regr_time[tt] + dt * np.array([-0.5, 0.5]),
                                                                    [1.0, 1.0], color=(139/255, 137 / 255, 184/255), linewidth=5)

                elif l == 'Probe':
                    tempBoot = bootstrap(decoding_JUV[var]['prediction_accuracy'][l], 1, 1000)
                    tempCtrlBoot = bootstrap(decoding_JUV[var]['prediction_accuracy_ctrl'][l], 1, 1000)

                # for Hit, FA, CorRej trials
                else:
                    tempBoot = bootstrap(decoding_JUV[var]['prediction_accuracy'][l], 1, 1000)
                    tempCtrlBoot = bootstrap(decoding_JUV[var]['prediction_accuracy_ctrl'][l], 1, 1000)

                # plot data and ctrl
                plotIdx = np.logical_and(regr_time > -2, regr_time < 2)
                decodingPlot.ax[idx // 3, idx % 3].plot(regr_time[plotIdx], tempBoot['bootAve'][plotIdx],
                                                        c=juvColor)
                decodingPlot.ax[idx // 3, idx % 3].fill_between(regr_time[plotIdx], tempBoot['bootLow'][plotIdx],
                                                                tempBoot['bootHigh'][plotIdx],
                                                                alpha=0.2, color=juvColor)

                decodingPlot.ax[idx // 3, idx % 3].plot(regr_time[plotIdx], tempCtrlBoot['bootAve'][plotIdx],
                                                        c=(0.7, 0.7, 0.7))
                decodingPlot.ax[idx // 3, idx % 3].fill_between(regr_time[plotIdx], tempCtrlBoot['bootLow'][plotIdx],
                                                                tempCtrlBoot['bootHigh'][plotIdx],
                                                                alpha=0.2, color=(0.7, 0.7, 0.7))


            decodingPlot.save_plot('Decoding accuracy for variable ' + var + '_' + group_label+'.tif',
                                   'tiff', saveFigPath)
            decodingPlot.save_plot('Decoding accuracy for variable ' + var + '_' + group_label+'.svg',
                                   'svg', saveFigPath)
            plt.close()

        """plot decoding rate separted by prior reward"""
        # subplot_label = ['Hit', 'FA', 'CorRej']
        # for var in varList:
        #     decodingPlot = StartSubplots(1, 3, ifSharey=True)
        #     plt.ylim((0.2, 1))
        #     decodingPlot.fig.suptitle('Average average decoding accuracy ' + var)
        #     for idx, l in enumerate(subplot_label):
        #         ## plot ADT animals
        #         decodingPlot.ax[idx % 3].set_title(l)
        #
        #         key_R = l + '_R'
        #         key_NR = l + '_NR'
        #         tempBootR = bootstrap(decoding_ADT[var]['prediction_accuracy'][key_R], 1, 1000)
        #         tempBootR_ctrl = bootstrap(decoding_ADT[var]['prediction_accuracy_ctrl'][key_R], 1, 1000)
        #         tempBootNR = bootstrap(decoding_ADT[var]['prediction_accuracy'][key_NR], 1, 1000)
        #         tempBootNR_ctrl = bootstrap(decoding_ADT[var]['prediction_accuracy_ctrl'][key_NR], 1, 1000)
        #
        #         # plot data and ctrl
        #         plotIdx = np.logical_and(regr_time > -2, regr_time < 2)
        #         decodingPlot.ax[idx % 3].plot(regr_time[plotIdx], tempBootR['bootAve'][plotIdx],
        #                                                 c=adtColor)
        #         decodingPlot.ax[idx % 3].fill_between(regr_time[plotIdx], tempBootR['bootLow'][plotIdx],
        #                                                         tempBootR['bootHigh'][plotIdx],
        #                                                         alpha=0.2, color=adtColor)
        #
        #         decodingPlot.ax[idx % 3].plot(regr_time[plotIdx], tempBootR_ctrl['bootAve'][plotIdx],
        #                                                 c=(0.7, 0.7, 0.7))
        #         decodingPlot.ax[idx % 3].fill_between(regr_time[plotIdx], tempBootR_ctrl['bootLow'][plotIdx],
        #                                                         tempBootR_ctrl['bootHigh'][plotIdx],
        #                                                         alpha=0.2, color=(0.7, 0.7, 0.7))
        #
        #         decodingPlot.ax[idx % 3].plot(regr_time[plotIdx], tempBootNR['bootAve'][plotIdx],
        #                                          c=adtColor,linestyle = '--')
        #         decodingPlot.ax[idx % 3].fill_between(regr_time[plotIdx], tempBootNR['bootLow'][plotIdx],
        #                                                  tempBootNR['bootHigh'][plotIdx],
        #                                                  alpha=0.2, color=adtColor)
        #
        #         decodingPlot.ax[idx % 3].plot(regr_time[plotIdx], tempBootNR_ctrl['bootAve'][plotIdx],
        #                                          c=(0.7, 0.7, 0.7))
        #         decodingPlot.ax[idx % 3].fill_between(regr_time[plotIdx], tempBootNR_ctrl['bootLow'][plotIdx],
        #                                                  tempBootNR_ctrl['bootHigh'][plotIdx],
        #                                                  alpha=0.2, color=(0.7, 0.7, 0.7))
        #         # plot JUV
        #         tempBootR = bootstrap(decoding_JUV[var]['prediction_accuracy'][key_R], 1, 1000)
        #         tempBootR_ctrl = bootstrap(decoding_JUV[var]['prediction_accuracy_ctrl'][key_R], 1, 1000)
        #         tempBootNR = bootstrap(decoding_JUV[var]['prediction_accuracy'][key_NR], 1, 1000)
        #         tempBootNR_ctrl = bootstrap(decoding_JUV[var]['prediction_accuracy_ctrl'][key_NR], 1, 1000)
        #
        #         plotIdx = np.logical_and(regr_time > -2, regr_time < 2)
        #         decodingPlot.ax[idx % 3].plot(regr_time[plotIdx], tempBootR['bootAve'][plotIdx],
        #                                          c=juvColor)
        #         decodingPlot.ax[idx % 3].fill_between(regr_time[plotIdx], tempBootR['bootLow'][plotIdx],
        #                                                  tempBootR['bootHigh'][plotIdx],
        #                                                  alpha=0.2, color=juvColor)
        #
        #         decodingPlot.ax[idx % 3].plot(regr_time[plotIdx], tempBootR_ctrl['bootAve'][plotIdx],
        #                                          c=(0.7, 0.7, 0.7))
        #         decodingPlot.ax[idx % 3].fill_between(regr_time[plotIdx], tempBootR_ctrl['bootLow'][plotIdx],
        #                                                  tempBootR_ctrl['bootHigh'][plotIdx],
        #                                                  alpha=0.2, color=(0.7, 0.7, 0.7))
        #
        #         decodingPlot.ax[idx % 3].plot(regr_time[plotIdx], tempBootNR['bootAve'][plotIdx],
        #                                          c=juvColor, linestyle='--')
        #         decodingPlot.ax[idx % 3].fill_between(regr_time[plotIdx], tempBootNR['bootLow'][plotIdx],
        #                                                  tempBootNR['bootHigh'][plotIdx],
        #                                                  alpha=0.2, color=juvColor)
        #
        #         decodingPlot.ax[idx % 3].plot(regr_time[plotIdx], tempBootNR_ctrl['bootAve'][plotIdx],
        #                                          c=(0.7, 0.7, 0.7))
        #         decodingPlot.ax[idx % 3].fill_between(regr_time[plotIdx], tempBootNR_ctrl['bootLow'][plotIdx],
        #                                                  tempBootNR_ctrl['bootHigh'][plotIdx],
        #                                                  alpha=0.2, color=(0.7, 0.7, 0.7))
        #
        #     decodingPlot.save_plot('Decoding accuracy for variable R and NR ' + var + '.tif',
        #                            'tiff', saveFigPath)
        #     decodingPlot.save_plot('Decoding accuracy for variable R and NR ' + var + '.svg',
        #                            'svg', saveFigPath)
        #     plt.close()

        # plot decoding accuracy as a function of ensemble size
        #maxNeurons = 300
        nNeurons = np.arange(10, 310, 10)
        decodingPlot = StartPlots()

        # get bootstrap of ADT and JUV
        ADTBoot = bootstrap(decoding_ADT['stimulus']['ensemble_accuracy'],1,1000)
        JUVBoot = bootstrap(decoding_JUV['stimulus']['ensemble_accuracy'],1,1000)

        ADTCtrlBoot = bootstrap(decoding_ADT['stimulus']['ensemble_ctrl'],1,1000)
        JUVCtrlBoot = bootstrap(decoding_JUV['stimulus']['ensemble_ctrl'], 1, 1000)

        decodingPlot.ax.plot(nNeurons, ADTBoot['bootAve'],
                                                c=adtColor, label = 'ADT')
        decodingPlot.ax.plot(nNeurons, JUVBoot['bootAve'],
                             c=juvColor, label='JUV')
        decodingPlot.ax.fill_between(nNeurons, ADTBoot['bootLow'],
                                                ADTBoot['bootHigh'],
                                                alpha=0.2, color=adtColor)

        decodingPlot.ax.fill_between(nNeurons, JUVBoot['bootLow'],
                                            JUVBoot['bootHigh'],
                                            alpha=0.2, color=juvColor)
        decodingPlot.ax.plot(nNeurons, ADTCtrlBoot['bootAve'],'--',
                             c = adtColor)
        decodingPlot.ax.plot(nNeurons, JUVCtrlBoot['bootAve'],'--',
                             c = juvColor)
        decodingPlot.legend(['ADT', 'JUV'])
        decodingPlot.ax.set_xlim([0,100])
        decodingPlot.ax.set_ylim([0.2,1.05])
        decodingPlot.ax.set_xlabel('Number of Neurons')
        decodingPlot.ax.set_ylabel('Average decoding accuracy')
       # stats
        p_list = []
        dt = np.mean(np.diff(nNeurons[:10]))
        for nn in range(len(nNeurons[:10])):
            ADT = np.array(
                [value for value in decoding_ADT['stimulus']['ensemble_accuracy'][nn, :] if not np.isnan(value)])
            JUV = np.array(
                [value for value in decoding_JUV['stimulus']['ensemble_accuracy'][nn, :] if not np.isnan(value)])

            #statistic, p_MWtest = mannwhitneyu(ADT, JUV)
            stats,p = ttest_ind(ADT, JUV)
            #results = bs.bootstrap_ab(ADT, JUV, bs_stats.median, bs_compare.difference)

            # Get the p-value
            #p_value = results.p_value
            p_list.append(p)
        q_values = multipletests(p_list, method='fdr_bh')[1]
        reject, p_adjusted, _, _ = multipletests(p_list, method='bonferroni')
        # Set your desired FDR level (e.g., 0.05)
        FDR_threshold = 0.05

        # Identify significant results
        significant_results = [p < FDR_threshold for p in q_values]
        for tt in range(len(nNeurons[:10])):
            if significant_results[tt]:
                decodingPlot.ax.plot(nNeurons[tt] + dt * np.array([-0.5, 0.5]),
                                                        [1.0, 1.0], color=(139/255, 137 / 255, 184/255), linewidth=5)

        decodingPlot.save_plot('Decoding accuracy ensemble size_' + group_label + '.tif',
                               'tiff', saveFigPath)
        decodingPlot.save_plot('Decoding accuracy ensemble size_'
                                + group_label + '.svg',
                               'svg', saveFigPath)
        plt.close()

    def decoding_hardeasy_session(self, n_predictors):
        nFiles = self.data_df.shape[0]
        for f in tqdm(range(nFiles)):
            if self.data_df['with_imaging'][f]:
                analysis = self.data_df['fluo_analysis'][f]
                gn_series = self.data_df['fluo_raw'][f]
                saveDataPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],
                                            'MLR')
                if not os.path.exists(saveDataPath):
                    os.makedirs(saveDataPath)

                saveTbTFile = os.path.join(saveDataPath, 'trialbytrialVar.pickle')
                if not os.path.exists(saveTbTFile):
                    X, y, regr_time = analysis.linear_model(n_predictors)
                    tbtVar = {}
                    tbtVar['X'] = X
                    tbtVar['y'] = y
                    tbtVar['regr_time'] = regr_time

                    # save X and y
                    with open(saveTbTFile, 'wb') as pf:
                        pickle.dump(tbtVar, pf, protocol=pickle.HIGHEST_PROTOCOL)
                        pf.close()
                else:
                    # load the saved results
                    with open(saveTbTFile, 'rb') as pf:
                        tbtVar = pickle.load(pf)
                        X = tbtVar['X']
                        y = tbtVar['y']
                        regr_time = tbtVar['regr_time']
                # run decoding; omit trials without fluorescent data
                saveDataPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],
                                            'decoding')
                if not os.path.exists(saveDataPath):
                    os.makedirs(saveDataPath)
                saveDataFile = os.path.join(saveDataPath, 'decodingResult_hardeasy.pickle')

                #if not os.path.exists(saveDataFile):
                decodeVar = {}

                decodeVar['stimulus'] = np.array(
                    [np.nan if np.isnan(x)
                     else np.int(x)
                     for x in analysis.beh['sound_num']])
                decodeSig = y

                trialMask = decodeVar['stimulus'] <= 8
                # check false alarm trials, and probe trials

                # stimulus 1-4: 1
                # stimulus 5-8: 0
                # stimulus 9-12；2
                # stimulus 13-16: 3


                classifier = "SVC"
                varList = ['stimulus', 'action', 'trialType']

                analysis.decoding_hardeasy(decodeSig, decodeVar,
                                  trialMask,  classifier,
                                  regr_time, saveDataFile)
                    #                else:
                    #                    print('Decoding done!')
                    # ensemble size analysis

                    #analysis.decode_analysis(gn_series, saveDataFile, saveDataPath)

    def decoding_hardeasy_summary(self, group_adt, group_juv, group_label):
        # load data from adult animal
        nFiles = group_adt.shape[0]
        #n_ctrl = 20
        # read coefficient, sig cells, neurons with
        for f in tqdm(range(nFiles)):
            saveDataPath = os.path.join(group_adt.iloc[f]['fluo_analysis_dir'], 'decoding')
            saveDataFile = os.path.join(saveDataPath, 'decodingResult_hardeasy.pickle')
            # load the pickle file
            with open(saveDataFile, 'rb') as file:
                decodingResult = pickle.load(file)

            # initialize variables
            if f==0:
                decodingVars = ['go', 'nogo']
                regr_time = decodingResult['time']
                decodingSummary_ADT = {}
                for var in decodingVars:
                    decodingSummary_ADT[var] = {}
                    decodingSummary_ADT[var]['accuracy'] = np.empty((len(regr_time),
                                                    0))
                    decodingSummary_ADT[var]['ctrl_accuracy'] = np.empty((len(regr_time),
                                                    0))

            for var in decodingVars:
            # calculate fraction of neurons that are significant
                decodingSummary_ADT[var]['accuracy'] = np.concatenate((decodingSummary_ADT[var]['accuracy'],
                                                        decodingResult[var]['accuracy'][:, np.newaxis]), 1)
                decodingSummary_ADT[var]['ctrl_accuracy'] = np.concatenate((decodingSummary_ADT[var]['ctrl_accuracy'],
                                                        decodingResult[var]['ctrl_accuracy'][:, np.newaxis]), 1)
                #get the average decoding accuracy between 1-2 second after cue


        # load juv group
        nFiles = group_juv.shape[0]
        # read coefficient, sig cells, neurons with
        for f in tqdm(range(nFiles)):
            saveDataPath = os.path.join(group_juv.iloc[f]['fluo_analysis_dir'], 'decoding')
            saveDataFile = os.path.join(saveDataPath, 'decodingResult_hardeasy.pickle')
            # load the pickle file
            with open(saveDataFile, 'rb') as file:
                decodingResult = pickle.load(file)

            # initialize variables
            if f==0:
                decodingVars =['go', 'nogo']
                regr_time = decodingResult['time']
                decodingSummary_JUV = {}
                for var in decodingVars:
                    decodingSummary_JUV[var] = {}
                    decodingSummary_JUV[var]['accuracy'] = np.empty((len(regr_time),
                                                    0))
                    decodingSummary_JUV[var]['ctrl_accuracy'] = np.empty((len(regr_time),
                                                    0))


            for var in decodingVars:
            # calculate fraction of neurons that are significant
                decodingSummary_JUV[var]['accuracy'] = np.concatenate((decodingSummary_JUV[var]['accuracy'],
                                                        decodingResult[var]['accuracy'][:, np.newaxis]), 1)
                decodingSummary_JUV[var]['ctrl_accuracy'] = np.concatenate((decodingSummary_JUV[var]['ctrl_accuracy'],
                                                        decodingResult[var]['ctrl_accuracy'][:, np.newaxis]), 1)


        # plot the summary results
        savefigpath = os.path.join(self.root_dir, self.summary_dir, 'fluo')
        if not os.path.exists(savefigpath):
            os.makedirs(savefigpath)

        group_label = 'hardeasy'
        savefigpath = os.path.join(self.root_dir, self.summary_dir, 'fluo')

        # plot
        adtColor = (83/255,187/255,244/255)
        juvColor = (255/255,67/255,46/255)

        varList = decodingSummary_ADT.keys()

        """ plot average decoding and by trial types"""
        decodingPlot = StartSubplots(1, 2, ifSharey=True)
        for idx,var in enumerate(varList):

            plt.ylim((0.4, 0.65))
            decodingPlot.fig.suptitle('Average decoding accuracy hardeasy')
                ## plot ADT animals
            decodingPlot.ax[idx].set_title(var)

            tempBoot = bootstrap(decodingSummary_ADT[var]['accuracy'], 1, 1000)

            tempCtrlBoot = bootstrap(decodingSummary_ADT[var]['ctrl_accuracy'], 1, 1000)

                # plot data and ctrl
            plotIdx = np.logical_and(regr_time > -2, regr_time < 2)
            decodingPlot.ax[idx].plot(regr_time[plotIdx], tempBoot['bootAve'][plotIdx],
                                                            c=adtColor)
            decodingPlot.ax[idx].fill_between(regr_time[plotIdx], tempBoot['bootLow'][plotIdx],
                                                                    tempBoot['bootHigh'][plotIdx],
                                                                    alpha=0.2, color=adtColor)

            decodingPlot.ax[idx].plot(regr_time[plotIdx], tempCtrlBoot['bootAve'][plotIdx],
                                                        c=(0.7, 0.7, 0.7))
            decodingPlot.ax[idx].fill_between(regr_time[plotIdx], tempCtrlBoot['bootLow'][plotIdx],
                                                                tempCtrlBoot['bootHigh'][plotIdx],
                                                                alpha=0.2, color=(0.7, 0.7, 0.7))

                # plot JUV
            tempBoot = bootstrap(decodingSummary_JUV[var]['accuracy'], 1, 1000)

            tempCtrlBoot = bootstrap(decodingSummary_JUV[var]['ctrl_accuracy'], 1, 1000)

            plotIdx = np.logical_and(regr_time > -2, regr_time < 2)
            decodingPlot.ax[idx].plot(regr_time[plotIdx], tempBoot['bootAve'][plotIdx],
                                                        c=juvColor)
            decodingPlot.ax[idx].fill_between(regr_time[plotIdx], tempBoot['bootLow'][plotIdx],
                                                                tempBoot['bootHigh'][plotIdx],
                                                                alpha=0.2, color=juvColor)

            decodingPlot.ax[idx].plot(regr_time[plotIdx], tempCtrlBoot['bootAve'][plotIdx],
                                                        c=(0.7, 0.7, 0.7))
            decodingPlot.ax[idx].fill_between(regr_time[plotIdx], tempCtrlBoot['bootLow'][plotIdx],
                                                                tempCtrlBoot['bootHigh'][plotIdx],
                                                                alpha=0.2, color=(0.7, 0.7, 0.7))
            p_list = []
            dt = np.mean(np.diff(regr_time))
            for tt in range(len(regr_time)):
                statistic, p_MWtest = mannwhitneyu(decodingSummary_JUV[var]['accuracy'][tt, :],
                                                   decodingSummary_ADT[var]['accuracy'][tt, :])
                p_list.append(p_MWtest)
            q_values = multipletests(p_list, method='fdr_bh')[1]
            # Set your desired FDR level (e.g., 0.05)
            FDR_threshold = 0.05

            # Identify significant results
            significant_results = [p < FDR_threshold for p in q_values]
            for tt in range(len(regr_time)):
                if significant_results[tt]:
                    decodingPlot.ax[idx ].plot(regr_time[tt] + dt * np.array([-0.5, 0.5]),
                                                            [0.6, 0.6], color=(139 / 255, 137 / 255, 184 / 255),
                                                            linewidth=5)

        decodingPlot.save_plot('Decoding accuracy for variable ' + group_label + '.tif',
                                   'tiff', savefigpath)
        decodingPlot.save_plot('Decoding accuracy for variable ' + group_label + '.svg',
                                   'svg', savefigpath)
        plt.close()

        # calculate average decoding accuracy between 1 - 2 seconds
        timeMask = np.logical_and(regr_time>=1,regr_time<2)
        ave_JUV = np.nanmean(decodingSummary_JUV['nogo']['accuracy'][timeMask, :], axis=0)
        ave_JUV_ctrl = np.nanmean(decodingSummary_JUV['nogo']['ctrl_accuracy'][timeMask, :], axis=0)
        ave_ADT = np.nanmean(decodingSummary_ADT['nogo']['accuracy'][timeMask, :], axis=0)
    def dpca_session(self):
        # clean up the plot function
        nFiles = self.data_df.shape[0]
        concated_y = []
        convated_Var = []
        for f in tqdm(range(nFiles)):
            if self.data_df['with_imaging'][f]:
                analysis = self.data_df['fluo_analysis'][f]
                gn_series = self.data_df['fluo_raw'][f]
                saveDataPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],
                                            'MLR')

                saveTbTFile = os.path.join(saveDataPath, 'trialbytrialVar.pickle')
                if not os.path.exists(saveTbTFile):
                    X, y, regr_time = analysis.linear_model(n_predictors)
                    tbtVar = {}
                    tbtVar['X'] = X
                    tbtVar['y'] = y
                    tbtVar['regr_time'] = regr_time

                    # save X and y
                    with open(saveTbTFile, 'wb') as pf:
                        pickle.dump(tbtVar, pf, protocol=pickle.HIGHEST_PROTOCOL)
                        pf.close()
                else:
                    # load the saved results
                    with open(saveTbTFile, 'rb') as pf:
                        tbtVar = pickle.load(pf)
                        X = tbtVar['X']
                        y = tbtVar['y']
                        regr_time = tbtVar['regr_time']


                stim = np.array([np.int(analysis.beh['sound_num'][x]) for x in range(len(analysis.beh['sound_num']))])
                tempStim = np.zeros(len(stim))
                for ss in range(len(stim)):
                    if stim[ss] <= 4:
                        tempStim[ss] = 1
                    elif stim[ss] > 4 and stim[ss] <= 8:
                        tempStim[ss] = 0
                    elif stim[ss] > 8 and stim[ss] <= 12:
                        tempStim[ss] = 2
                    elif stim[ss] > 12:
                        tempStim[ss] = 3

                first_not_nan_trial = np.min(np.where(~np.isnan(y[0, :, 0])))
                last_not_nan_trial = np.max(np.where(~np.isnan(y[0, :, 0])))

                pcaVar = {'stim':tempStim[first_not_nan_trial:last_not_nan_trial],
                          'action':X[4,first_not_nan_trial:last_not_nan_trial,0]}

                savePCAPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],
                                            'dPCA')
                if not os.path.exists(savePCAPath):
                    os.makedirs(savePCAPath)
                # signal:

                pca_time = regr_time[regr_time>-1]
                analysis.get_dpca(y[:,first_not_nan_trial:last_not_nan_trial,regr_time>-1], pcaVar, pca_time, savePCAPath)
                #analysis.dpca_elife(y[:,first_not_nan_trial:last_not_nan_trial,:], pcaVar, pca_time, savePCAPath)

                # concatenate y and PCA Var for every session, running a whole pca on the
                # concatenated data set

        # doing pca on the full dataset

    def noise_session(self):
        # running noise analysis for every session
        nFiles = self.data_df.shape[0]
        for f in tqdm(range(nFiles)):
            if self.data_df['with_imaging'][f]:
                analysis = self.data_df['fluo_analysis'][f]
                saveDataPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],
                                            'noise')
                if not os.path.exists(saveDataPath):
                    os.makedirs(saveDataPath)

                saveDataFile = os.path.join(saveDataPath, 'noiseResults.pickle')
                #if not os.path.exists(saveDataFile):
               # if not os.path.exists(saveDataFile):# check false alarm trials, and probe trials
                subTrialMask = {}
                subTrialMask['FA'] = analysis.beh['trialType'] == -1
                subTrialMask['Hit'] = analysis.beh['trialType'] == 2
                subTrialMask['CorRej'] = analysis.beh['trialType'] == 0
                analysis.noise_analysis(subTrialMask, saveDataPath)

    def noise_summary(self, group_adt, group_juv, group_label):
        # summary analysis for noise-related data
        # 1. relationship between FA rate and population mean and var
        # 2. pre-post cue signal/noise correlation comparison between ADT and JUV
        # 3. second order correlation of noise and signal correlation

        # initialize data matrix
        savefigpath = os.path.join(self.root_dir, self.summary_dir, 'fluo')
        trialTypes = ['Hit','FA','CR']

        nFiles = group_adt.shape[0]

        for f in tqdm(range(nFiles)):
            analysis = group_adt.iloc[f]['fluo_analysis']
            saveDataPath = os.path.join(group_adt.iloc[f]['fluo_analysis_dir'],
                                            'noise')
            saveDataFile = os.path.join(saveDataPath, 'noiseResults.pickle')
            with open(saveDataFile, 'rb') as file:
                    noiseResult = pickle.load(file)

            saveDataFile2 = os.path.join(saveDataPath, 'signal_noise real.pickle')
            with open(saveDataFile2, 'rb') as file:
                signalnoiseResult = pickle.load(file)

            if f==0:
                    # initialize data matrix
                binX = noiseResult['binX']
                mean_byTrial_ADT = {}
                var_byTrial_ADT = {}
                for tt in trialTypes:
                    mean_byTrial_ADT[tt] = []
                    var_byTrial_ADT[tt] = []
                pFA_Mean_ADT = noiseResult['pFA_Mean'][:,np.newaxis]
                pFA_Var_ADT = noiseResult['pFA_Var'][:,np.newaxis]
                preCue_signal_Ave_ADT = []
                postCue_signal_Ave_ADT = []
                preCue_signal_Ave_ADT_trialType = [[] for t in trialTypes]
                postCue_signal_Ave_ADT_trialType = [[] for t in trialTypes]
                preCue_noise_Ave_ADT = []
                postCue_noise_Ave_ADT = []
                preCue_noise_Ave_ADT_trialType = [[] for t in trialTypes]
                postCue_noise_Ave_ADT_trialType = [[] for t in trialTypes]
                preCue_SNSlope_ADT = []
                postCue_SNSlope_ADT = []
                noise_corr2_trial_ADT = noiseResult['noise_corr2_trial'][:,np.newaxis]
                signal_corr2_trial_ADT = noiseResult['signal_corr2_trial'][:,np.newaxis]
                corr2_t = noiseResult['interp_time'][0:-1]
                    # calcualte average population mean and var by different trialtype
            for idx,tt in enumerate(trialTypes):
                if tt=='Hit':
                    trialMask = analysis.beh['trialType'] == 2
                elif tt=='FA':
                    trialMask = analysis.beh['trialType'] == -1
                else:
                    trialMask = analysis.beh['trialType'] == 0
                mean_byTrial_ADT[tt].append(np.nanmean(noiseResult['mean_byTrial'][trialMask]))
                var_byTrial_ADT[tt].append(np.nanmean(noiseResult['var_byTrial'][trialMask]))
                preCue_signal_Ave_ADT_trialType[idx].append(
                    np.nanmean(signalnoiseResult['preCueCorr_signal_trialType'][:,:,idx]))
                postCue_signal_Ave_ADT_trialType[idx].append(
                    np.nanmean(signalnoiseResult['afterCueCorr_signal_trialType'][:, :, idx]))
                preCue_noise_Ave_ADT_trialType[idx].append(
                    np.nanmean(signalnoiseResult['preCueCorr_noise_trialType'][:, :, idx]))
                postCue_noise_Ave_ADT_trialType[idx].append(
                        np.nanmean(signalnoiseResult['afterCueCorr_noise_trialType'][:, :, idx]))
            pFA_Mean_ADT = np.concatenate((pFA_Mean_ADT,
                                 noiseResult['pFA_Mean'][:,np.newaxis]),1)
            pFA_Var_ADT = np.concatenate((pFA_Var_ADT,
                                 noiseResult['pFA_Var'][:,np.newaxis]),1)
            preCue_signal_Ave_ADT.append(np.nanmean(signalnoiseResult['preCueCorr_signal']))
            postCue_signal_Ave_ADT.append(np.nanmean(signalnoiseResult['afterCueCorr_signal']))
            preCue_noise_Ave_ADT.append(np.nanmean(signalnoiseResult['preCueCorr_noise']))
            postCue_noise_Ave_ADT.append(np.nanmean(signalnoiseResult['afterCueCorr_noise']))
            preCue_SNSlope_ADT.append(signalnoiseResult['SNCorr_slope_pre'])
            postCue_SNSlope_ADT.append(signalnoiseResult['SNCorr_slope_after'])
            noise_corr2_trial_ADT = np.concatenate((noise_corr2_trial_ADT,
                                 noiseResult['noise_corr2_trial'][:,np.newaxis]),1)
            signal_corr2_trial_ADT = np.concatenate((signal_corr2_trial_ADT,
                                        noiseResult['signal_corr2_trial'][:,np.newaxis]),1)

        # load JUV data
        nFiles = group_juv.shape[0]

        for f in tqdm(range(nFiles)):
            analysis = group_juv.iloc[f]['fluo_analysis']
            saveDataPath = os.path.join(group_juv.iloc[f]['fluo_analysis_dir'],
                                            'noise')
            saveDataFile = os.path.join(saveDataPath, 'noiseResults.pickle')
            with open(saveDataFile, 'rb') as file:
                    noiseResult = pickle.load(file)
            saveDataFile2 = os.path.join(saveDataPath, 'signal_noise real.pickle')
            with open(saveDataFile2, 'rb') as file:
                signalnoiseResult = pickle.load(file)

            if f==0:
                    # initialize data matrix
                binX = noiseResult['binX']
                mean_byTrial_JUV = {}
                var_byTrial_JUV = {}
                for tt in trialTypes:
                    mean_byTrial_JUV[tt] = []
                    var_byTrial_JUV[tt] = []
                pFA_Mean_JUV = noiseResult['pFA_Mean'][:,np.newaxis]
                pFA_Var_JUV= noiseResult['pFA_Var'][:,np.newaxis]
                preCue_signal_Ave_JUV = []
                postCue_signal_Ave_JUV = []
                preCue_noise_Ave_JUV = []
                postCue_noise_Ave_JUV = []
                preCue_signal_Ave_JUV_trialType = [[] for t in trialTypes]
                postCue_signal_Ave_JUV_trialType = [[] for t in trialTypes]
                preCue_noise_Ave_JUV_trialType = [[] for t in trialTypes]
                postCue_noise_Ave_JUV_trialType = [[] for t in trialTypes]
                preCue_SNSlope_JUV = []
                postCue_SNSlope_JUV = []
                noise_corr2_trial_JUV = noiseResult['noise_corr2_trial'][:,np.newaxis]
                signal_corr2_trial_JUV = noiseResult['signal_corr2_trial'][:,np.newaxis]

                    # calcualte average population mean and var by different trialtype
            for idx,tt in enumerate(trialTypes):
                if tt=='Hit':
                    trialMask = analysis.beh['trialType'] == 2
                elif tt=='FA':
                    trialMask = analysis.beh['trialType'] == -1
                else:
                    trialMask = analysis.beh['trialType'] == 0
                mean_byTrial_JUV[tt].append(np.nanmean(noiseResult['mean_byTrial'][trialMask]))
                var_byTrial_JUV[tt].append(np.nanmean(noiseResult['var_byTrial'][trialMask]))
                preCue_signal_Ave_JUV_trialType[idx].append(
                        np.nanmean(signalnoiseResult['preCueCorr_signal_trialType'][:,:,idx]))
                postCue_signal_Ave_JUV_trialType[idx].append(
                        np.nanmean(signalnoiseResult['afterCueCorr_signal_trialType'][:, :, idx]))
                preCue_noise_Ave_JUV_trialType[idx].append(
                        np.nanmean(signalnoiseResult['preCueCorr_noise_trialType'][:, :, idx]))
                postCue_noise_Ave_JUV_trialType[idx].append(
                        np.nanmean(signalnoiseResult['afterCueCorr_noise_trialType'][:, :, idx]))
            pFA_Mean_JUV = np.concatenate((pFA_Mean_JUV,
                                                   noiseResult['pFA_Mean'][:,np.newaxis]),1)
            pFA_Var_JUV = np.concatenate((pFA_Var_JUV,
                                                  noiseResult['pFA_Var'][:,np.newaxis]),1)
            preCue_signal_Ave_JUV.append(np.nanmean(signalnoiseResult['preCueCorr_signal']))
            postCue_signal_Ave_JUV.append(np.nanmean(signalnoiseResult['afterCueCorr_signal']))
            preCue_noise_Ave_JUV.append(np.nanmean(signalnoiseResult['preCueCorr_noise']))
            postCue_noise_Ave_JUV.append(np.nanmean(signalnoiseResult['afterCueCorr_noise']))
            preCue_SNSlope_JUV.append(signalnoiseResult['SNCorr_slope_pre'])
            postCue_SNSlope_JUV.append(signalnoiseResult['SNCorr_slope_after'])
            noise_corr2_trial_JUV = np.concatenate((noise_corr2_trial_JUV,
                                                       noiseResult['noise_corr2_trial'][:,np.newaxis]),1)
            signal_corr2_trial_JUV = np.concatenate((signal_corr2_trial_JUV,
                                                            noiseResult['signal_corr2_trial'][:,np.newaxis]),1)

        # make some plots!
        adtColor = (83/255,187/255,244/255)
        juvColor = (255/255,67/255,46/255)
        groups = ['Hit', 'FA', 'CR']
        mean_JUV = np.zeros((len(mean_byTrial_JUV['Hit']),3)) # mean and ste
        mean_ADT = np.zeros((len(mean_byTrial_ADT['Hit']),3))
        var_JUV = np.zeros((len(var_byTrial_JUV['Hit']),3))
        var_ADT  = np.zeros((len(var_byTrial_ADT['Hit']),3))
        for idx,tt in enumerate(groups):
            mean_JUV[:,idx] = (mean_byTrial_JUV[tt]-np.mean(mean_byTrial_JUV[tt]))/np.std(mean_byTrial_JUV[tt])
            mean_ADT[:,idx] = (mean_byTrial_ADT[tt]-np.mean(mean_byTrial_ADT[tt]))/np.std(mean_byTrial_ADT[tt])

            var_JUV[:,idx] = (var_byTrial_JUV[tt]-np.mean(var_byTrial_JUV[tt]))/np.std(var_byTrial_JUV[tt])
            var_ADT[:,idx] = (var_byTrial_ADT[tt]-np.mean(var_byTrial_ADT[tt]))/np.std(var_byTrial_ADT[tt])

        popStatsPlot = StartSubplots(1,2)
        bar_width = 0.35
        # Create an array of positions for the bars
        x = np.arange(len(groups))

        boxplot = popStatsPlot.ax[0].boxplot(mean_ADT, positions=x - 0.2, widths=0.4,
            patch_artist=True, showfliers=False)
        for box in boxplot['boxes']:
            box.set(facecolor=adtColor)

        boxplot = popStatsPlot.ax[0].boxplot(mean_JUV, positions=x + 0.2, widths=0.4,
                                             patch_artist=True, showfliers=False)
        for box in boxplot['boxes']:
            box.set(facecolor=juvColor)
        # Customize the plot
        popStatsPlot.ax[0].set_xlabel('Trial types')
        popStatsPlot.ax[0].set_ylabel('Population mean (precue)')
        popStatsPlot.ax[0].set_xticks(x, groups)
        # Add a legend

        boxplot = popStatsPlot.ax[1].boxplot(var_ADT, positions=x - 0.2, widths=0.4,
            patch_artist=True, showfliers=False)
        for box in boxplot['boxes']:
            box.set(facecolor=adtColor)

        boxplot = popStatsPlot.ax[1].boxplot(var_JUV, positions=x + 0.2, widths=0.4,
                                             patch_artist=True, showfliers=False)
        for box in boxplot['boxes']:
            box.set(facecolor=juvColor)
        # Customize the plot
        popStatsPlot.ax[1].set_xlabel('Trial types')
        popStatsPlot.ax[1].set_ylabel('Population variance (precue)')
        popStatsPlot.ax[1].set_xticks(x, groups)

        popStatsPlot.save_plot('Population stats and trial type.tiff','tiff',
                               savefigpath)
        plt.close()

        # plot the average pair-wise correlation of signal and noise, pre/post cue
        cueScatter = StartSubplots(2,2)
        pre = np.full((len(preCue_noise_Ave_ADT)),1)
        post = np.full((len(preCue_noise_Ave_ADT)), 2)
        cueScatter.ax[0,0].scatter(pre,preCue_signal_Ave_ADT,
                                 label = 'Paired Data', color=adtColor)
        cueScatter.ax[0,0].scatter(post,postCue_signal_Ave_ADT,
                                 label = 'Paired Data', color=adtColor)
        cueScatter.ax[0,1].scatter(pre,preCue_noise_Ave_ADT,
                                 label = 'Paired Data', color=adtColor)
        cueScatter.ax[0,1].scatter(post,postCue_noise_Ave_ADT,
                                 label = 'Paired Data', color=adtColor)
        pre = np.full((len(preCue_noise_Ave_JUV)),1)
        post = np.full((len(preCue_noise_Ave_JUV)), 2)
        cueScatter.ax[0,0].scatter(pre,preCue_signal_Ave_JUV,
                                 label = 'Paired Data', color=juvColor)
        cueScatter.ax[0,0].scatter(post,postCue_signal_Ave_JUV,
                                 label = 'Paired Data', color=juvColor)
        cueScatter.ax[0,1].scatter(pre,preCue_noise_Ave_JUV,
                                 label = 'Paired Data', color=juvColor)
        cueScatter.ax[0,1].scatter(post,postCue_noise_Ave_JUV,
                                 label = 'Paired Data', color=juvColor)
        for x,y in zip(preCue_signal_Ave_ADT, postCue_signal_Ave_ADT):
            cueScatter.ax[0,0].plot([1,2],[x,y], color = adtColor, linewidth=1)
        for x,y in zip(preCue_signal_Ave_JUV, postCue_signal_Ave_JUV):
            cueScatter.ax[0,0].plot([1,2],[x,y], color = juvColor, linewidth=1)
        for x, y in zip(preCue_noise_Ave_ADT, postCue_noise_Ave_ADT):
            cueScatter.ax[0,1].plot([1, 2], [x, y], color=adtColor, linewidth=1)
        for x, y in zip(preCue_noise_Ave_JUV, postCue_noise_Ave_JUV):
            cueScatter.ax[0,1].plot([1, 2], [x, y], color=juvColor, linewidth=1)
        cueScatter.ax[0,0].set_xticks([1,2], ['pre', 'post'])
        cueScatter.ax[0,0].set_ylabel('Pearson correlation')
        cueScatter.ax[0,0].set_title('Signal correlation')
        cueScatter.ax[0,1].set_xticks([1,2], ['pre', 'post'])
        cueScatter.ax[0,1].set_title('Noise correlation')

        # calculate percent change
        change_signal_ADT = (np.array(postCue_signal_Ave_ADT)-np.array(preCue_signal_Ave_ADT))/\
                            np.array(preCue_signal_Ave_ADT)
        change_signal_JUV = (np.array(postCue_signal_Ave_JUV)-np.array(preCue_signal_Ave_JUV))/\
                            np.array(preCue_signal_Ave_JUV)
        change_noise_ADT = (np.array(postCue_noise_Ave_ADT)-np.array(preCue_noise_Ave_ADT))/\
                           np.array(preCue_noise_Ave_ADT)
        change_noise_JUV = (np.array(postCue_noise_Ave_JUV)-np.array(preCue_noise_Ave_JUV))/\
                           np.array(preCue_noise_Ave_JUV)
        boxplot=cueScatter.ax[1,0].boxplot([change_signal_ADT,change_signal_JUV],
                                   positions=[1,2],
                                   patch_artist=True, showfliers=False)
        for median_line in boxplot['medians']:
            median_line.set_linewidth(4.0)
            median_line.set_color('black')
        cueScatter.ax[1,0].set_xticks([1,2], ['ADT', 'JUV'])
        boxplot['boxes'][0].set(facecolor=adtColor)
        boxplot['boxes'][1].set(facecolor=juvColor)
        cueScatter.ax[1, 0].set_ylabel('Change in average correlation')
        boxplot=cueScatter.ax[1,1].boxplot([change_noise_ADT,change_noise_JUV],
                                   positions=[1,2],
                                   patch_artist=True, showfliers=False)
        for median_line in boxplot['medians']:
            median_line.set_linewidth(3.0)
            median_line.set_color('black')
        cueScatter.ax[1,1].set_xticks([1,2], ['ADT', 'JUV'])
        boxplot['boxes'][0].set(facecolor=adtColor)
        boxplot['boxes'][1].set(facecolor=juvColor)

        cueScatter.save_plot('pre-post signal and noise correlation.tiff','tiff',
                               savefigpath)
        cueScatter.save_plot('pre-post signal and noise correlation.svg','svg',
                               savefigpath)
        plt.close()

    # plot the average pair-wise correlation of signal and noise, pre/post cue, for different trialtypes
    #todo (or not?) put this into a function
        for idx,tt in enumerate(trialTypes):

            cueScatter = StartSubplots(2, 2)
            pre = np.full((len(preCue_noise_Ave_ADT)), 1)
            post = np.full((len(preCue_noise_Ave_ADT)), 2)
            cueScatter.ax[0, 0].scatter(pre, preCue_signal_Ave_ADT_trialType[idx],
                                        label='Paired Data', color=adtColor)
            cueScatter.ax[0, 0].scatter(post, postCue_signal_Ave_ADT_trialType[idx],
                                        label='Paired Data', color=adtColor)
            cueScatter.ax[0, 1].scatter(pre, preCue_noise_Ave_ADT_trialType[idx],
                                        label='Paired Data', color=adtColor)
            cueScatter.ax[0, 1].scatter(post, postCue_noise_Ave_ADT_trialType[idx],
                                        label='Paired Data', color=adtColor)
            pre = np.full((len(preCue_noise_Ave_JUV)), 1)
            post = np.full((len(preCue_noise_Ave_JUV)), 2)
            cueScatter.ax[0, 0].scatter(pre, preCue_signal_Ave_JUV_trialType[idx],
                                        label='Paired Data', color=juvColor)
            cueScatter.ax[0, 0].scatter(post, postCue_signal_Ave_JUV_trialType[idx],
                                        label='Paired Data', color=juvColor)
            cueScatter.ax[0, 1].scatter(pre, preCue_noise_Ave_JUV_trialType[idx],
                                        label='Paired Data', color=juvColor)
            cueScatter.ax[0, 1].scatter(post, postCue_noise_Ave_JUV_trialType[idx],
                                        label='Paired Data', color=juvColor)
            for x, y in zip(preCue_signal_Ave_ADT_trialType[idx], postCue_signal_Ave_ADT_trialType[idx]):
                cueScatter.ax[0, 0].plot([1, 2], [x, y], color=adtColor, linewidth=1)
            for x, y in zip(preCue_signal_Ave_JUV_trialType[idx], postCue_signal_Ave_JUV_trialType[idx]):
                cueScatter.ax[0, 0].plot([1, 2], [x, y], color=juvColor, linewidth=1)
            for x, y in zip(preCue_noise_Ave_ADT_trialType[idx], postCue_noise_Ave_ADT_trialType[idx]):
                cueScatter.ax[0, 1].plot([1, 2], [x, y], color=adtColor, linewidth=1)
            for x, y in zip(preCue_noise_Ave_JUV_trialType[idx], postCue_noise_Ave_JUV_trialType[idx]):
                cueScatter.ax[0, 1].plot([1, 2], [x, y], color=juvColor, linewidth=1)
            cueScatter.ax[0, 0].set_xticks([1, 2], ['pre', 'post'])
            cueScatter.ax[0, 0].set_ylabel('Pearson correlation')
            cueScatter.ax[0, 0].set_title('Signal correlation '+tt+' trials')
            cueScatter.ax[0, 1].set_xticks([1, 2], ['pre', 'post'])
            cueScatter.ax[0, 1].set_title('Noise correlation '+tt+' trials')

            # calculate percent change
            change_signal_ADT = (np.array(postCue_signal_Ave_ADT_trialType[idx]) - np.array(preCue_signal_Ave_ADT_trialType[idx])) / \
                                np.array(preCue_signal_Ave_ADT_trialType[idx])
            change_signal_JUV = (np.array(postCue_signal_Ave_JUV_trialType[idx]) - np.array(preCue_signal_Ave_JUV_trialType[idx])) / \
                                np.array(preCue_signal_Ave_JUV_trialType[idx])
            change_noise_ADT = (np.array(postCue_noise_Ave_ADT_trialType[idx]) - np.array(preCue_noise_Ave_ADT_trialType[idx])) / \
                               np.array(preCue_noise_Ave_ADT_trialType[idx])
            change_noise_JUV = (np.array(postCue_noise_Ave_JUV_trialType[idx]) - np.array(preCue_noise_Ave_JUV_trialType[idx])) / \
                               np.array(preCue_noise_Ave_JUV_trialType[idx])
            boxplot = cueScatter.ax[1, 0].boxplot([change_signal_ADT, change_signal_JUV],
                                                  positions=[1, 2],
                                                  patch_artist=True, showfliers=False)
            cueScatter.ax[1, 0].set_xticks([1, 2], ['ADT', 'JUV'])
            boxplot['boxes'][0].set(facecolor=adtColor)
            boxplot['boxes'][1].set(facecolor=juvColor)
            cueScatter.ax[1, 0].set_ylabel('Change in average correlation')
            boxplot = cueScatter.ax[1, 1].boxplot([change_noise_ADT, change_noise_JUV],
                                                  positions=[1, 2],
                                                  patch_artist=True, showfliers=False)
            cueScatter.ax[1, 1].set_xticks([1, 2], ['ADT', 'JUV'])
            boxplot['boxes'][0].set(facecolor=adtColor)
            boxplot['boxes'][1].set(facecolor=juvColor)

            cueScatter.save_plot('pre-post signal and noise correlation '+tt+'.tiff', 'tiff',
                                 savefigpath)
            plt.close()

    # plot the scatter plot of pair-wise correlation - x: signal correlation, y: noise correlation
        SNSlopePlot = StartSubplots(1,2)
        boxplot=SNSlopePlot.ax[0].boxplot([preCue_SNSlope_ADT,preCue_SNSlope_JUV],
                                   positions=[1,2],
                                   patch_artist=True, showfliers=False)
        for median_line in boxplot['medians']:
            median_line.set_linewidth(3.0)
            median_line.set_color('black')
        boxplot['boxes'][0].set(facecolor=adtColor)
        boxplot['boxes'][1].set(facecolor=juvColor)
        SNSlopePlot.ax[0].set_xticks([1,2],['ADT', 'JUV'])
        SNSlopePlot.ax[0].set_ylabel('Average correlation coefficient')
        SNSlopePlot.ax[0].set_title('Signal noise slope pre-cue')

        boxplot=SNSlopePlot.ax[1].boxplot([postCue_SNSlope_ADT,postCue_SNSlope_JUV],
                                   positions=[1,2],
                                   patch_artist=True, showfliers=False)
        for median_line in boxplot['medians']:
            median_line.set_linewidth(3.0)
            median_line.set_color('black')
        boxplot['boxes'][0].set(facecolor=adtColor)
        boxplot['boxes'][1].set(facecolor=juvColor)
        SNSlopePlot.ax[1].set_xticks([1,2],['ADT', 'JUV'])
        SNSlopePlot.ax[1].set_title('Signal noise slope post-cue')
        SNSlopePlot.save_plot('Signal-noise correlation.tiff','tiff',
                               savefigpath)
        SNSlopePlot.save_plot('Signal-noise correlation.svg','svg',
                               savefigpath)
        # plot the average 2nd correlation
        # get bootstrap results
        bootSignal_ADT = bootstrap(signal_corr2_trial_ADT, 1, 0, 1000)
        bootSignal_JUV = bootstrap(signal_corr2_trial_JUV, 1, 0, 1000)
        bootNoise_ADT = bootstrap(noise_corr2_trial_ADT, 1, 0, 1000)
        bootNoise_JUV = bootstrap(noise_corr2_trial_JUV, 1, 0, 1000)

        secondCorrPlot = StartSubplots(1,2)
        secondCorrPlot.ax[0].plot(corr2_t, bootSignal_ADT['bootAve'], color=adtColor)
        secondCorrPlot.ax[0].fill_between(corr2_t, bootSignal_ADT['bootLow'],
                                                        bootSignal_ADT['bootHigh'],
                                                        alpha=0.2, color=adtColor)
        secondCorrPlot.ax[0].plot(corr2_t, bootSignal_JUV['bootAve'], color=adtColor)
        secondCorrPlot.ax[0].fill_between(corr2_t, bootSignal_JUV['bootLow'],
                                                        bootSignal_JUV['bootHigh'],
                                                        alpha=0.2, color=juvColor)
        secondCorrPlot.ax[0].set_xlabel('Time from cue(s)')
        secondCorrPlot.ax[0].set_ylabel('Second order correlation')
        secondCorrPlot.ax[0].set_title('Signal')

        secondCorrPlot.ax[1].plot(corr2_t, bootNoise_ADT['bootAve'], color=adtColor)
        secondCorrPlot.ax[1].fill_between(corr2_t, bootNoise_ADT['bootLow'],
                                                        bootNoise_ADT['bootHigh'],
                                                        alpha=0.2, color=adtColor)
        secondCorrPlot.ax[1].plot(corr2_t, bootNoise_JUV['bootAve'], color=juvColor)
        secondCorrPlot.ax[1].fill_between(corr2_t, bootNoise_JUV['bootLow'],
                                                        bootNoise_JUV['bootHigh'],
                                                        alpha=0.2, color=juvColor)
        secondCorrPlot.ax[1].set_xlabel('Time from cue(s)')
        secondCorrPlot.ax[1].set_ylabel('Second order correlation')
        secondCorrPlot.ax[1].set_title('Noise')
        secondCorrPlot.save_plot('second order signal and noise correlation.tiff','tiff',
                               savefigpath)
        plt.close()

    def pseudo_session(self):
        nFiles = self.data_df.shape[0]
        for f in tqdm(range(nFiles)):
            if self.data_df['with_imaging'][f]:
                analysis = self.data_df['fluo_analysis'][f]
                gn_series = self.data_df['fluo_raw'][f]
                saveDataPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],
                                            'MLR')
                if not os.path.exists(saveDataPath):
                    os.makedirs(saveDataPath)

                saveTbTFile = os.path.join(saveDataPath, 'trialbytrialVar.pickle')
                if not os.path.exists(saveTbTFile):
                    X, y, regr_time = analysis.linear_model(n_predictors)
                    tbtVar = {}
                    tbtVar['X'] = X
                    tbtVar['y'] = y
                    tbtVar['regr_time'] = regr_time

                    # save X and y
                    with open(saveTbTFile, 'wb') as pf:
                        pickle.dump(tbtVar, pf, protocol=pickle.HIGHEST_PROTOCOL)
                        pf.close()
                else:
                    # load the saved results
                    with open(saveTbTFile, 'rb') as pf:
                        tbtVar = pickle.load(pf)
                        X = tbtVar['X']
                        y = tbtVar['y']
                        regr_time = tbtVar['regr_time']
                # run decoding; omit trials without fluorescent data
                saveDataPath = os.path.join(self.data_df.iloc[f]['fluo_analysis_dir'],
                                            'pseudo')
                if not os.path.exists(saveDataPath):
                    os.makedirs(saveDataPath)

                saveDataFile = os.path.join(saveDataPath, 'decoding_pseudo_stimulus.pickle')
                #               if not os.path.exists(saveDataFile):
                #if not os.path.exists(saveDataFile):
                decodeVar = {}

                decodeVar['stimulus'] = np.array(
                    [np.nan if np.isnan(x)
                     else np.int(x)
                     for x in analysis.beh['sound_num']])
                decodeVar['trialType'] = np.array(
                    [np.nan if np.isnan(x)
                     else np.int(x)
                     for x in analysis.beh['trialType']])
                decodeSig = y

                trialMask = decodeVar['stimulus'] <= 8
                # check false alarm trials, and probe trials
                subTrialMask = {}
                subTrialMask['FA'] = analysis.beh['trialType'] == -1
                subTrialMask['probe'] = decodeVar['stimulus'] > 8
                subTrialMask['Hit'] = analysis.beh['trialType'] == 2
                subTrialMask['CorRej'] = analysis.beh['trialType'] == 0
                # stimulus 1-4: 1
                # stimulus 5-8: 0
                # stimulus 9-12；2
                # stimulus 13-16: 3
                tempSti = np.zeros(len(decodeVar['stimulus']))
                for ss in range(len(decodeVar['stimulus'])):
                    if decodeVar['stimulus'][ss] <= 4:
                        tempSti[ss] = 1
                    elif decodeVar['stimulus'][ss] > 4 and decodeVar['stimulus'][ss] <= 8:
                        tempSti[ss] = 0
                    elif decodeVar['stimulus'][ss] > 8 and decodeVar['stimulus'][ss] <= 12:
                        tempSti[ss] = 1
                    elif decodeVar['stimulus'][ss] > 12:
                        tempSti[ss] = 0
                    # trialType
                decodeVar['stimulus'] = tempSti
                decodeVar['trialType'][decodeVar['trialType'] == 2] = 1

                classifier = "SVC"
                varList = ['stimulus']

                analysis.pseudoensemble_analysis(decodeSig, decodeVar,
                                             trialMask,
                                             subTrialMask, classifier,
                                             regr_time, saveDataPath
                                             )

    def load_decode_results(self, group, nNeurons, trialTypes):
        # load decoding results
        nFiles = group.shape[0]
        nInd = int(nNeurons / 10) - 1
        for f in tqdm(range(nFiles)):
            #analysis = group_adt.iloc[f]['fluo_analysis']
            # load decoding
            decodingPath = os.path.join(group.iloc[f]['fluo_analysis_dir'], 'decoding')
            decodingFile= os.path.join(decodingPath, 'decodingResult.pickle')
            # load the pickle file
            with open(decodingFile, 'rb') as file:
                decodingResult = pickle.load(file)
            pseudoPath = os.path.join(group.iloc[f]['fluo_analysis_dir'],
                                            'pseudo')
            pseudoDecodingFile = os.path.join(pseudoPath, 'decoding_pseudo_stimulus.pickle')
            with open(pseudoDecodingFile, 'rb') as file:
                    pseudoDecodingResult = pickle.load(file)

            # load signal-noise angle
            noisePath = os.path.join(group.iloc[f]['fluo_analysis_dir'], 'noise')
            noiseFile= os.path.join(noisePath, 'signal_noise real.pickle')
            with open(noiseFile, 'rb') as file:
                    noiseResult = pickle.load(file)

            pseudoNoisePath = os.path.join(group.iloc[f]['fluo_analysis_dir'], 'pseudo')
            pseudoNoisefile= os.path.join(pseudoNoisePath, 'signal_noise pseudo.pickle')
            with open(pseudoNoisefile, 'rb') as file:
                    pseudoNoiseResult = pickle.load(file)

            if f==0:
                    # initialize data matrix
                decodingTime = decodingResult['time']
                decodingAccuracy_trialType={}
                decodingAccuracy_trialType_full = {}
                pseudoDecodingAccuracy_trialType = {}
                decodingAccuracy_ave_trialType = {}
                pseudoDecodingAccuracy_ave_trialType = {}

                for trial in trialTypes:
                    decodingAccuracy_trialType_full[trial] = decodingResult['stimulus']['prediction_accuracy'][trial][:,np.newaxis]
                if len(decodingResult['stimulus']['decode_realSize']['nNeurons'])>=nInd+1:
                    decodingAccuracy = decodingResult['stimulus']['decode_realSize']['accuracy'][:,nInd][:,np.newaxis]

                    # decoding accuracy in trial types
                    for trial in trialTypes:
                        decodingAccuracy_trialType[trial] = decodingResult['stimulus']['decode_realSize']['prediction_accuracy'][trial][:,nInd][:,np.newaxis]
                else:
                    decodingAccuracy = np.full((len(decodingResult['stimulus']['decode_realSize']['regr_time']),1),np.nan)
                    for trial in trialTypes:
                        decodingAccuracy_trialType[trial] = np.full((len(decodingResult['stimulus']['decode_realSize']['regr_time']),1),np.nan)
                decodingCtrl = decodingResult['stimulus']['ctrl_accuracy'][:,np.newaxis]
                #pseudoDecodingAccuracy_ADT = pseudoDecodingResult['decode_pseudo']['accuracy'][:,np.newaxis]
                if len(decodingResult['stimulus']['decode_realSize']['nNeurons'])>=nInd+1:
                    pseudoDecodingAccuracy = pseudoDecodingResult['decode_pseudoSize']['accuracy'][:,nInd][:,np.newaxis]
                    for trial in trialTypes:
                        pseudoDecodingAccuracy_trialType[trial] = \
                            pseudoDecodingResult['decode_pseudoSize']['prediction_accuracy'][trial][:, nInd][:,
                            np.newaxis]

                else:
                    pseudoDecodingAccuracy = np.full((len(decodingResult['stimulus']['decode_realSize']['regr_time']), 1),
                                                   np.nan)
                    for trial in trialTypes:
                        pseudoDecodingAccuracy_trialType[trial] = np.full((len(decodingResult['stimulus']['decode_realSize']['regr_time']),1),np.nan)

                pseudoDecodingCtrl = pseudoDecodingResult['decode_pseudo']['ctrl_accuracy'][:,np.newaxis]

                # calculate average decoding accuracy from 1-2 s
                timeStart = 1
                timeEnd = 2
                timeMask = np.logical_and(decodingTime >= timeStart,
                                          decodingTime < timeEnd)
                if len(decodingResult['stimulus']['decode_realSize']['nNeurons'])>=nInd+1:
                    decodingAccuracy_ave = [np.nanmean(decodingResult['stimulus']['decode_realSize']['accuracy'][timeMask,nInd], 0)]
                    pseudoDecodingAccuracy_ave = [
                        np.nanmean(pseudoDecodingResult['decode_pseudoSize']['accuracy'][timeMask, nInd], 0)]

                    for trial in trialTypes:
                        decodingAccuracy_ave_trialType[trial] = [np.nanmean(decodingResult['stimulus']['decode_realSize']['prediction_accuracy'][trial][timeMask,nInd], 0)]
                        pseudoDecodingAccuracy_ave_trialType[trial] = [np.nanmean(
                            pseudoDecodingResult['decode_pseudoSize']['prediction_accuracy'][trial][timeMask, nInd],
                            0)]
                else:
                    decodingAccuracy_ave_ADT = [np.nan]
                    for trial in trialTypes:
                        decodingAccuracy_ave_trialType[trial] = [np.nan]
                        pseudoDecodingAccuracy_ave_trialType[trial] = [np.nan]
                SNSlope_pre = [noiseResult['SNCorr_slope_pre']]
                SNSlope_post = [noiseResult['SNCorr_slope_after']]

                pseudoSNSlope_pre = [pseudoNoiseResult['SNCorr_slope_pre']]
                pseudoSNSlope_post= [pseudoNoiseResult['SNCorr_slope_after']]

            else:
                    # calcualte average population mean and var by different trialtype
                for trial in trialTypes:
                    decodingAccuracy_trialType_full[trial] = np.concatenate((decodingAccuracy_trialType_full[trial],
                                            decodingResult['stimulus']['prediction_accuracy'][
                                                                     trial][:, np.newaxis]),1)

                if len(decodingResult['stimulus']['decode_realSize']['nNeurons'])>=nInd+1:
                    decodingAccuracy = np.concatenate((decodingAccuracy,
                                        decodingResult['stimulus']['decode_realSize']['accuracy'][:, nInd][:,np.newaxis]) ,1)
                    pseudoDecodingAccuracy = np.concatenate((pseudoDecodingAccuracy,
                                        pseudoDecodingResult['decode_pseudoSize']['accuracy'][:, nInd][:,np.newaxis]), 1)
                    decodingAccuracy_ave.append(np.nanmean(decodingResult['stimulus']['decode_realSize']['accuracy'][timeMask,nInd],0))
                    pseudoDecodingAccuracy_ave.append(np.nanmean(pseudoDecodingResult['decode_pseudoSize']['accuracy'][timeMask,nInd],0))

                    for trial in trialTypes:
                        decodingAccuracy_trialType[trial] = np.concatenate((decodingAccuracy_trialType[trial],
                                        decodingResult['stimulus']['decode_realSize']['prediction_accuracy'][trial][:, nInd][:,np.newaxis]) ,1)
                        pseudoDecodingAccuracy_trialType[trial] = np.concatenate((pseudoDecodingAccuracy_trialType[trial],
                                                                                pseudoDecodingResult[
                                                                                    'decode_pseudoSize'][
                                                                                    'prediction_accuracy'][trial][:,
                                                                                nInd][:, np.newaxis]), 1)
                        decodingAccuracy_ave_trialType[trial].append(
                            np.nanmean(decodingResult['stimulus']['decode_realSize']['prediction_accuracy'][trial][timeMask, nInd], 0))
                        pseudoDecodingAccuracy_ave_trialType[trial].append(
                            np.nanmean(pseudoDecodingResult['decode_pseudoSize']['prediction_accuracy'][trial][timeMask, nInd], 0))

                decodingCtrl = np.concatenate((decodingCtrl,
                                    decodingResult['stimulus']['ctrl_accuracy'][:,np.newaxis]),1)
                pseudoDecodingCtrl = np.concatenate((pseudoDecodingCtrl,
                                    pseudoDecodingResult['decode_pseudo']['ctrl_accuracy'][:,np.newaxis]),1)

                SNSlope_pre.append(noiseResult['SNCorr_slope_pre'])
                SNSlope_post.append(noiseResult['SNCorr_slope_after'])

                pseudoSNSlope_pre.append(pseudoNoiseResult['SNCorr_slope_pre'])
                pseudoSNSlope_post.append(pseudoNoiseResult['SNCorr_slope_after'])

        result= {}
        result['accuracy_real'] = decodingAccuracy
        result['accuracy_ave_real'] = decodingAccuracy_ave
        result['accuracy_ave_trialType_real'] = decodingAccuracy_ave_trialType
        result['accuracy_pseudo'] = pseudoDecodingAccuracy
        result['accuracy_ave_pseudo'] = pseudoDecodingAccuracy_ave
        result['accuracy_ave_trialType_pseudo'] = pseudoDecodingAccuracy_ave_trialType
        result['accuracy_real_trialType'] = decodingAccuracy_trialType
        result['accuracy_pseudo_trialType'] = pseudoDecodingAccuracy_trialType
        result['accuracy_real_ctrl'] = decodingCtrl
        result['accuracy_pseudo_ctrl'] = pseudoDecodingCtrl
        result['SNSlope_pre_real'] = SNSlope_pre
        result['SNSlope_post_real'] = SNSlope_post
        result['SNSlope_pre_pseudo'] = pseudoSNSlope_pre
        result['SNSlope_post_pseudo'] = pseudoSNSlope_post
        result['time'] = decodingTime
        return result


    def pseudo_summray(self, group_adt, group_juv,group_label):
        savefigpath = os.path.join(self.root_dir, self.summary_dir, 'fluo')
        trialTypes = ['Hit','FA','CorRej']
        adtColor = (83/255,187/255,244/255)
        juvColor = (255/255,67/255,46/255)
        nNeurons = 80

        # load ADT groups
        result_ADT = self.load_decode_results(group_adt,nNeurons, trialTypes)

        result_JUV = self.load_decode_results(group_juv, nNeurons, trialTypes)

        # compare decoding results between pseudo - real ensemble
        decodingPlot = StartSubplots(1,2)
        bootADT_real = bootstrap(result_ADT['accuracy_real'],1, 1000)
        bootADT_pseudo = bootstrap(result_ADT['accuracy_pseudo'], 1, 1000)
        decodingPlot.ax[0].plot(result_ADT['time'], bootADT_real['bootAve'], color='blue', label = 'real')
        decodingPlot.ax[0].plot(result_ADT['time'], bootADT_pseudo['bootAve'], color = 'red', label = 'real')
        decodingPlot.ax[0].fill_between(result_ADT['time'], bootADT_real['bootLow'],
                                                        bootADT_real['bootHigh'],
                                                        alpha=0.2, color=adtColor)
        decodingPlot.ax[0].fill_between(result_ADT['time'], bootADT_pseudo['bootLow'],
                                                bootADT_pseudo['bootHigh'],
                                                alpha=0.2, color=adtColor)
        decodingPlot.ax[0].set_ylim([0,1])
        # pair-wise wilcoxon test
        # p_list = []
        # for tt in range(len(decodingTime)):
        #     statistic, p_wilcoxon = wilcoxon(decodingAccuracy_ADT[tt, :],
        #                                        pseudoDecodingAccuracy_ADT[tt, :])
        #     p_list.append(p_wilcoxon)
        # q_values = multipletests(p_list, method='fdr_bh')[1]
        # # Set your desired FDR level (e.g., 0.05)
        # FDR_threshold = 0.05
        #
        # # Identify significant results
        # significant_results = [p < FDR_threshold for p in q_values]
        # for tt in range(len(regr_time)):
        #     if significant_results[tt]:
        #         decodingPlot.ax[0].plot(regr_time[tt] + dt * np.array([-0.5, 0.5]),
        #                                                 [1, 1], color=(1, 69 / 255, 0), linewidth=5)
        bootJUV_real = bootstrap(result_JUV['accuracy_real'],1, 1000)
        bootJUV_pseudo = bootstrap(result_JUV['accuracy_pseudo'], 1, 1000)
        decodingPlot.ax[1].plot(result_JUV['time'], bootJUV_real['bootAve'], color='blue', label = 'real')
        decodingPlot.ax[1].plot(result_JUV['time'], bootJUV_pseudo['bootAve'], color = 'red', label = 'real')
        decodingPlot.ax[1].fill_between(result_JUV['time'], bootJUV_real['bootLow'],
                                                        bootJUV_real['bootHigh'],
                                                        alpha=0.2, color='blue')
        decodingPlot.ax[1].fill_between(result_JUV['time'], bootJUV_pseudo['bootLow'],
                                                        bootJUV_pseudo['bootHigh'],
                                                        alpha=0.2, color='red')
        decodingPlot.ax[1].set_ylim([0, 1])
        p_list = []
        for tt in range(len(result_JUV['time'])):
            statistic, p_wilcoxon = wilcoxon(result_JUV['accuracy_real'][tt, :],
                                               result_JUV['accuracy_pseudo'][tt, :])
            p_list.append(p_wilcoxon)
        q_values = multipletests(p_list, method='fdr_bh')[1]

        # plot decoding accuracy by trial type
        for trial in trialTypes:
            decodingPlot = StartSubplots(1,2)
            bootADT_real = bootstrap(result_ADT['accuracy_real_trialType'][trial],1, 1000)
            bootADT_pseudo = bootstrap(result_ADT['accuracy_pseudo_trialType'][trial], 1, 1000)
            decodingPlot.ax[0].plot(result_ADT['time'], bootADT_real['bootAve'], color='blue', label = 'real')
            decodingPlot.ax[0].plot(result_ADT['time'], bootADT_pseudo['bootAve'], color = 'red', label = 'real')
            decodingPlot.ax[0].fill_between(result_ADT['time'], bootADT_real['bootLow'],
                                                            bootADT_real['bootHigh'],
                                                            alpha=0.2, color=adtColor)
            decodingPlot.ax[0].fill_between(result_ADT['time'], bootADT_pseudo['bootLow'],
                                                    bootADT_pseudo['bootHigh'],
                                                    alpha=0.2, color=adtColor)
            decodingPlot.ax[0].set_ylim([0,1])
            decodingPlot.ax[0].set_title('ADT,'+trial)
            # pair-wise wilcoxon test
            # p_list = []
            # for tt in range(len(decodingTime)):
            #     statistic, p_wilcoxon = wilcoxon(decodingAccuracy_ADT[tt, :],
            #                                        pseudoDecodingAccuracy_ADT[tt, :])
            #     p_list.append(p_wilcoxon)
            # q_values = multipletests(p_list, method='fdr_bh')[1]
            # # Set your desired FDR level (e.g., 0.05)
            # FDR_threshold = 0.05
            #
            # # Identify significant results
            # significant_results = [p < FDR_threshold for p in q_values]
            # for tt in range(len(regr_time)):
            #     if significant_results[tt]:
            #         decodingPlot.ax[0].plot(regr_time[tt] + dt * np.array([-0.5, 0.5]),
            #                                                 [1, 1], color=(1, 69 / 255, 0), linewidth=5)
            bootJUV_real = bootstrap(result_JUV['accuracy_real_trialType'][trial],1, 1000)
            bootJUV_pseudo = bootstrap(result_JUV['accuracy_pseudo_trialType'][trial], 1, 1000)
            decodingPlot.ax[1].plot(result_JUV['time'], bootJUV_real['bootAve'], color='blue', label = 'real')
            decodingPlot.ax[1].plot(result_JUV['time'], bootJUV_pseudo['bootAve'], color = 'red', label = 'real')
            decodingPlot.ax[1].fill_between(result_JUV['time'], bootJUV_real['bootLow'],
                                                            bootJUV_real['bootHigh'],
                                                            alpha=0.2, color='blue')
            decodingPlot.ax[1].fill_between(result_JUV['time'], bootJUV_pseudo['bootLow'],
                                                            bootJUV_pseudo['bootHigh'],
                                                            alpha=0.2, color='red')
            decodingPlot.ax[1].set_ylim([0, 1])
            decodingPlot.ax[1].set_title('JUV,' + trial)
            p_list = []
            for tt in range(len(result_JUV['time'])):
                statistic, p_wilcoxon = wilcoxon(result_JUV['accuracy_real'][tt, :],
                                                   result_JUV['accuracy_pseudo'][tt, :])
                p_list.append(p_wilcoxon)
            q_values = multipletests(p_list, method='fdr_bh')[1]
            decodingPlot.save_plot('Average decoding accuracy real-pseudo'+trial+' .tiff', 'tiff',
                                      savefigpath)
            decodingPlot.save_plot('Average decoding accuracy real-pseudo'+trial+' .svg', 'svg',
                                      savefigpath)

        # test equal median for pseudo/real ensemble for ADT and JUV Hit trials
        starttime = 1
        endtime = 2
        #aveADTHitPseudo = np.nanmean(result_ADT['accuracy'])
        decodingAvePlot = StartSubplots(1,2)


        decodingAvePlot.ax[0].set_xticks([1,2],['Real', 'Pseudo'])
        decodingAvePlot.ax[0].set_title('ADT')
        decodingAvePlot.ax[0].set_ylim([0.3, 1])

        for x, y in zip(result_ADT['accuracy_ave_real'], result_ADT['accuracy_ave_pseudo']):
            decodingAvePlot.ax[0].plot([1, 2], [x, y], color=adtColor, linewidth=1)
        decodingAvePlot.ax[0].set_xticks([1, 2], ['Real', 'Pseudo'])
        decodingAvePlot.ax[0].set_title('ADT')
        decodingAvePlot.ax[0].set_ylim([0.5, 1])
        for x, y in zip(result_JUV['accuracy_ave_real'], result_JUV['accuracy_ave_pseudo']):
            decodingAvePlot.ax[1].plot([1, 2], [x, y], color=juvColor, linewidth=1)

        decodingAvePlot.ax[1].set_xticks([1,2],['Real', 'Pseudo'])
        decodingAvePlot.ax[1].set_title('JUV')
        decodingAvePlot.ax[1].set_ylim([0.3, 1])

        # sig test

        wilcoxon(result_JUV['accuracy_ave_real'], result_JUV['accuracy_ave_pseudo'])
        from scipy.stats import ttest_rel
        ttest_rel(result_ADT['accuracy_ave_real'], result_ADT['accuracy_ave_pseudo'])
        ttest_rel(result_JUV['accuracy_ave_real'], result_JUV['accuracy_ave_pseudo'])
        decodingAvePlot.save_plot('Average decoding accuracy real-pseudo.tiff', 'tiff',
                                  savefigpath)
        decodingAvePlot.save_plot('Average decoding accuracy real-pseudo.svg', 'svg',
                                  savefigpath)

        # sn slope
        SNSlopePlot = StartSubplots(1,2)
        SNSlopePlot.ax[0].boxplot([result_ADT['SNSlope_pre_real'],result_ADT['SNSlope_pre_pseudo']],
                                   positions=[1,2],
                                   patch_artist=True, showfliers=False)
        SNSlopePlot.ax[0].set_xticks([1,2],['ADT', 'JUV'])
        SNSlopePlot.ax[0].set_title('Signal noise slope pre-cue')

        SNSlopePlot.ax[1].boxplot([result_ADT['SNSlope_post_pseudo'],result_JUV['SNSlope_post_pseudo']],
                                   positions=[1,2],
                                   patch_artist=True, showfliers=False)
        SNSlopePlot.ax[1].set_xticks([1,2],['ADT', 'JUV'])
        SNSlopePlot.ax[1].set_title('Signal noise slope post-cue')
        SNSlopePlot.save_plot('Signal-noise correlation pseudo ensemble.tiff','tiff',
                               savefigpath)

if __name__ == "__main__":

    test_single_session = False
    if test_single_session:
        animal, session = 'JUV015', '220407'
        beh_folder = r"Z:\HongliWang\Madeline\LateLearning\Analysis\JUV015\220407"
        beh_file = "JUV015-220407-behaviorLOG.mat"
        # get behavior data
        x = GoNogoBehaviorMat(animal, session, os.path.join(beh_folder, beh_file))

        # get fluorescent data, check for sessions without imaging

        input_folder = r"Z:\HongliWang\Madeline\LateLearning\Data\JUV015\220407\suite2p\plane0"

        fluo_analysis_dir = r'Z:\HongliWang\Madeline\LateLearning\Analysis\JUV015\220407\fluo'

        fluo_file = os.path.join(fluo_analysis_dir, animal + session + '_dff_df_file.csv')

        gn_series = Suite2pSeries(input_folder)
        gn_series.get_dFF(x, fluo_file)
        new_beh_file = r'Z:\HongliWang\Madeline\LateLearning\Analysis\JUV015\220407\behavior\behAnalysis.pickle'
        analysis = fluoAnalysis(new_beh_file, fluo_file)
        analysis.align_fluo_beh(fluo_analysis_dir)

        fluofigpath = r'Z:\HongliWang\Madeline\LateLearning\Analysis\JUV015\220407\fluo\noise'
        #analysis.plot_dFF(os.path.join(fluofigpath,'cells-combined-cue'))

    # build multiple linear regression
    # arrange the independent variables
        saveFigPath = r'Z:\HongliWang\Madeline\LateLearning\Analysis\JUV015\220407\fluo\pseudo'
    #
        saveDataPath = os.path.join(fluo_analysis_dir,
                                    'MLR')

        saveTbTFile = os.path.join(saveDataPath, 'trialbytrialVar.pickle')

         # load the saved results
        with open(saveTbTFile, 'rb') as pf:
            tbtVar = pickle.load(pf)
            X = tbtVar['X']
            y = tbtVar['y']
            regr_time = tbtVar['regr_time']
        n_predictors = 14
        #labels = ['s(n+1)','s(n)', 's(n-1)','c(n+1)', 'c(n)', 'c(n-1)',
        #       'r(n+1)', 'r(n)', 'r(n-1)', 'x(n+1)', 'x(n)', 'x(n-1)', 'speed', 'lick']
        # X, y, regr_time = analysis.linear_model(n_predictors)
        # tbtVar = {}
        # tbtVar['X'] = X
        # tbtVar['y'] = y
        # tbtVar['regr_time'] = regr_time

        # save X and y
        with open(saveTbTFile, 'wb') as pf:
            pickle.dump(tbtVar, pf, protocol=pickle.HIGHEST_PROTOCOL)
            pf.close()
        #saveDataPath = r'C:\\Users\\linda\\Documents\\GitHub\\madeline_go_nogo\\data\\MLR.pickle'
    # # run MLR; omit trials without fluorescent data
    #     #MLRResult = analysis.linear_regr(X[:,1:-2,:], y[:,1:-2,:], regr_time, saveDataPath)
    #     #analysis.plotMLRResult(MLRResult, labels, saveFigPath)
    #
    # for decoding
    # decode for action/outcome/stimulus
        decodeVar = {}

        decodeVar['action'] = X[4,:,0]
        decodeVar['outcome'] = X[7,:,0]
        decodeVar['stimulus'] = np.array([np.int(analysis.beh['sound_num'][x]) for x in range(len(analysis.beh['sound_num']))])
        decodeSig = y

        trialMask = decodeVar['stimulus'] <= 8
        # check false alarm trials, and probe trials
        decodeVar['trialType'] = np.array(
            [np.nan if np.isnan(x)
             else np.int(x)
             for x in analysis.beh['trialType']])

        subTrialMask = {}
        subTrialMask['FA'] = analysis.beh['trialType'] == -1
        subTrialMask['probe'] = decodeVar['stimulus'] > 8
        subTrialMask['Hit'] = analysis.beh['trialType'] == 2
        subTrialMask['CorRej'] = analysis.beh['trialType'] == 0
    # stimulus 1-4: 1
    # stimulus 5-8: 0
    # stimulus 9-12；2
    # stimulus 13-16: 3
        tempSti = np.zeros(len(decodeVar['stimulus']))
        for ss in range(len(decodeVar['stimulus'])):
            if decodeVar['stimulus'][ss] <= 4:
                tempSti[ss] = 1
            elif decodeVar['stimulus'][ss] > 4 and decodeVar['stimulus'][ss] <= 8:
                tempSti[ss] = 0
            elif decodeVar['stimulus'][ss] > 8 and decodeVar['stimulus'][ss] <= 12:
                tempSti[ss] = 1
            elif decodeVar['stimulus'][ss] > 12:
                tempSti[ss] = 0
            # trialType
        decodeVar['stimulus'] = tempSti
        decodeVar['trialType'][decodeVar['trialType']==2] = 1

        #classifier = "RandomForest"
        classifier = 'SVC'
        varList = ['trialType']
        saveDataPath = r'C:\Users\linda\Documents\GitHub\madeline_go_nogo\data\decode.pickle'
        saveFigPath = r'C:\Users\linda\Documents\GitHub\madeline_go_nogo\data'
        analysis.decoding(decodeSig, decodeVar, varList, trialMask,subTrialMask, classifier, regr_time, saveDataPath)
        analysis.decode_analysis([],saveDataPath, saveFigPath)
    #     neuronRaw = gn_series


        # noise analysis
        save_data_path = saveFigPath
        analysis.pseudoensemble_analysis(decodeSig, decodeVar,
                                         trialMask,
                                         subTrialMask, classifier,
                                         regr_time, save_data_path
                                         )
        #analysis.noise_analysis(subTrialMask, save_data_path)

        '''demixed PCA'''
        stim = np.array([np.int(analysis.beh['sound_num'][x]) for x in range(len(analysis.beh['sound_num']))])
        tempStim = np.zeros(len(stim))
        for ss in range(len(stim)):
            if stim[ss] <= 4:
                tempStim[ss] = 1
            elif stim[ss] > 4 and stim[ss] <= 8:
                tempStim[ss] = 0
            elif stim[ss] > 8 and stim[ss] <= 12:
                tempStim[ss] = 2
            elif stim[ss] > 12:
                tempStim[ss] = 3
        pcaVar = {'stim':tempStim, 'action':X[4,:,0]}

    test_summary = True
    if test_summary:
        root_dir = r'Z:\HongliWang\Madeline\LateLearning'
        fluo_summary = fluoSum(root_dir)
        fluo_summary.process_single_session()

        # plot single cell PSTH
        #fluo_summary.cell_plots()

        # try MLR with 8 cues?
        #fluo_summary.MLR_session()

        # separate groups into end stage (8 cues) and early stage (2-4 cues)
        #c1 = fluo_summary.data_df['numStim'] >= 8

        # go over the age group, set TRA to ADT
        for ii in range(len(fluo_summary.data_df['age'])):
            if fluo_summary.data_df['age'][ii] == 'TRA':
                fluo_summary.data_df['age'][ii] = 'ADT'

        c2 = np.logical_and(fluo_summary.data_df['age'] == 'ADT', fluo_summary.data_df['with_imaging'])
        c3 = np.logical_and(fluo_summary.data_df['age'] == 'JUV',fluo_summary.data_df['with_imaging'])

        ADT_late = fluo_summary.data_df[c2]
        JUV_late = fluo_summary.data_df[c3]
        group_label = 'late'
        #fluo_summary.MLR_summary(ADT_late, JUV_late, group_label)
        # multiple linear regression
        n_predictors = 14
        fluo_summary.decoding_session(n_predictors)

        # similarly, separate groups into end stage and early stage
        fluo_summary.decoding_summary_running(ADT_late, JUV_late, group_label)
        fluo_summary.decoding_summary(ADT_late, JUV_late, group_label)

        #fluo_summary.decoding_hardeasy_session(n_predictors)
        #fluo_summary.decoding_hardeasy_summary(ADT_late, JUV_late, group_label)
        #fluo_summary.dpca_session()
        #fluo_summary.noise_session()
        #fluo_summary.noise_summary(ADT_late, JUV_late, group_label)

        #fluo_summary.pseudo_session()
        fluo_summary.pseudo_summray(ADT_late, JUV_late, group_label)
        x=1
