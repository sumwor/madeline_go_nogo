# pipeline for fluorescent analysis
# %matplotlib inline
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import statsmodels.api as sm
from pyPlotHW import *
from utility_HW import bootstrap
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import binomtest
# read df/f and behavior data, create a class with behavior and df/f data

class fluoAnalysis:

    def __init__(self, beh_file, fluo_file):
        self.beh = pd.read_csv(beh_file)
        self.fluo = pd.read_csv(fluo_file)

    def align_fluo_beh(self):
        # align the fluoresent data to behavior based on trials
        # interpolate fluorescent data and running speed
        nTrials = self.beh.shape[0]
        nCells = self.fluo.shape[1] - 2 # exclude the first column (index) and last column (time)
        startTime = -3
        endTime = 8
        step = 0.01
        interpT = np.arange(startTime, endTime, step)
        self.dFF_aligned = np.zeros((len(interpT), nTrials, nCells))
        self.run_aligned = np.zeros((len(interpT), nTrials))
        self.interpT = interpT

        for tt in range(nTrials):

            # align speed
            speed = self.beh['running_speed'][tt]
            speed = np.array(eval(speed))
            speed_time = self.beh['running_time'][tt]
            speed_time = np.array(eval(speed_time))
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
                # interpolate
                self.dFF_aligned[:,tt,cc] = np.interp(interpT, tempT, tempdFF)

    def plot_dFF(self, savefigpath):
        # PSTH plot for different trial types
        # cue
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

            Hit_dFF = self.dFF_aligned[:,self.beh['choice']==2,cc]
            FA_dFF = self.dFF_aligned[:, self.beh['choice'] == -1, cc]
            Miss_dFF = self.dFF_aligned[:, self.beh['choice'] == -2, cc]
            CorRej_dFF = self.dFF_aligned[:, self.beh['choice'] == 0, cc]
            ProbeLick_dFF = self.dFF_aligned[:, self.beh['choice'] == -3, cc]
            ProbeNoLick_dFF = self.dFF_aligned[:, self.beh['choice'] == -4, cc]

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
        # plot the dFF curve of a given cell for several trials

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
        X = np.zeros((11,len(regr_time)))
        #
        Y = np.zeros((nCells, len(regr_time)))

        # need to determine whether to use exact frequency or go/no go/probe
        X[0, :] = np.ones(len(regr_time)) * self.beh['sound_num'][trial]

        # choice: lick = 1; no lick = -1
        X[1, :] = np.ones(len(regr_time)) * (
            choiceList[trial + 1] if trial < nTrials - 1 else np.nan)
        X[2, :] = np.ones(len(regr_time)) * (choiceList[trial])
        X[3, :] = np.ones(len(regr_time)) * (choiceList[trial - 1] if trial > 0 else np.nan)

        # reward
        X[4, :] = np.ones(len(regr_time)) * (
            rewardList[trial + 1] if trial < nTrials - 1 else np.nan)
        X[5, :] = np.ones(len(regr_time)) * rewardList[trial]
        X[6, :] = np.ones(len(regr_time)) * (rewardList[trial - 1] if trial > 0 else np.nan)

        # interaction
        X[7, :] = X[1, :] * X[4, :]
        X[8, :] = X[2, :] * X[5, :]
        X[9, :] = X[3, :] * X[6, :]

        # running speed
        tStep = np.nanmean(np.diff(regr_time))

        for tt in range(len(regr_time)):
            t_start = regr_time[tt] - tStep / 2
            t_end = regr_time[tt] + tStep / 2
            temp_run = self.run_aligned[:, trial]
            X[10, tt] = np.nanmean(
                temp_run[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])

            # dependent variable: dFF
            for cc in range(nCells):
                temp_dFF = self.dFF_aligned[:, trial, cc]
                Y[cc, tt] = np.nanmean(
                    temp_dFF[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])

        return X, Y

    def linear_model(self):
        # arrange the independent variables and dependent variables for later linear regression
        # model:
        # y = b0 + b1*cue + b2* cn+1 + b3* cn + b4* cn-1 + b5* rn+1 + b6*rn + b7*rn-1 + b8* cn+1*rn+1 + b9* cn*rn + b10* cn-1*rn-1 + b11* running_speed

        tStart = -2.95
        tEnd = 7.95
        tStep = 0.1
        regr_time = np.arange(tStart, tEnd, tStep)
        nTrials = self.beh.shape[0]
        nCells = self.fluo.shape[1]-2
        independent_X = np.zeros((11, nTrials, len(regr_time)))
        dFF_Y = np.zeros((nCells, nTrials, len(regr_time)))

        choiceList = [1 if lick > 0 else -1 for lick in self.beh['licks_out']]
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

    def linear_regr(self, X, y, regr_time):
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
        MLRResult = {'coeff': np.zeros((11, len(regr_time), nCells)), 'pval': np.zeros((11, len(regr_time), nCells)), 'r2': np.zeros((len(regr_time), nCells))}

        n_jobs = -1

        # Parallelize the loop over `trial`
        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(self.run_MLR)(X[:,:,tt],y[:,:,tt]) for tt in
            tqdm(range(len(regr_time))))

        for tt in range(len(regr_time)):
            MLRResult['coeff'][:,tt,:], MLRResult['pval'][:,tt,:], MLRResult['r2'][tt,:] = results[tt]

        MLRResult['regr_time'] = regr_time
        return MLRResult

    def plotMLRResult(self, MLRResult):
        # get the average coefficient plot and fraction of significant neurons
        varList = ['Cue', 'Cn+1', 'Cn', 'Cn-1','Rn+1','Rn', 'Rn-1','Xn+1','Xn','Xn-1','Run']
        # average coefficient
        nPredictors = MLRResult['coeff'].shape[0]

        coeffPlot = StartSubplots(3,4, ifSharey=True)

        for n in range(nPredictors):
            tempBoot = bootstrap(MLRResult['coeff'][n,:,:],1, 1000)
            coeffPlot.ax[n//4, n%4].plot(MLRResult['regr_time'], tempBoot['bootAve'], c =(0,0,0))
            coeffPlot.ax[n // 4, n % 4].fill_between(MLRResult['regr_time'], tempBoot['bootLow'], tempBoot['bootHigh'],
                                          alpha=0.2,  color = (0.7,0.7,0.7))
            coeffPlot.ax[n//4, n%4].set_title(varList[n])
        plt.show()

        # fraction of significant neurons
        sigPlot = StartSubplots(3, 4, ifSharey=True)
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

        plt.show()

        # plot r-square
        r2Boot = bootstrap(MLRResult['r2'], 1, 1000)
        r2Plot = StartPlots()
        r2Plot.ax.plot(MLRResult['regr_time'], r2Boot['bootAve'],c=(0, 0, 0))
        r2Plot.ax.fill_between(MLRResult['regr_time'], r2Boot['bootLow'], r2Boot['bootHigh'],
                                                 color=(0.7, 0.7, 0.7))
        r2Plot.ax.set_title('R-square')
        plt.show()
    def saveMLRResult(self, MLRResult):
        pass

    def run_MLR(self, x, y):
        # running individual MLR for parallel computing
        nCells = y.shape[0]
        coeff = np.zeros((11, nCells))
        pval = np.zeros((11, nCells))
        rSquare = np.zeros((nCells))

        x = sm.add_constant(np.transpose(x))
        for cc in range(nCells):
            model = sm.OLS(y[cc,1:-1], x[1:-1,:]).fit()
            coeff[:,cc] = model.params[1:]
            pval[:,cc] = model.pvalues[1:]
            rSquare[cc] = model.rsquared

        return coeff, pval, rSquare

# define the function for parallel computing

if __name__ == "__main__":
    #beh_file = r"C:\Users\xiachong\Documents\GitHub\madeline_go_nogo\data\JUV015_220409_behavior_output.csv"
    #fluo_file = r"C:\Users\xiachong\Documents\GitHub\JUV015_220409_dff_df_file.csv"
    #fluofigpath = r"C:\Users\xiachong\Documents\GitHub\madeline_go_nogo\data\fluo_plot"
    beh_file = r"C:\Users\hongl\Documents\GitHub\madeline_go_nogo\data\JUV015_220409_behavior_output.csv"
    fluo_file = r"C:\Users\hongl\Documents\GitHub\madeline_imagingData\JUV015_220409_dff_df_file.csv"
    fluofigpath = r"C:\Users\hongl\Documents\GitHub\madeline_go_nogo\data\fluo_plot"

    animal, session = 'JUV011', '211215'
    # dff_df = gn_series.calculate_dff(melt=False)

    beh_data = pd.read_csv(beh_file)
    fluo_data = pd.read_csv(fluo_file)

    # build the linear regression model
    analysis = fluoAnalysis(beh_file,fluo_file)
    analysis.align_fluo_beh()

    # individual cell plots
    # trials = np.arange(20,50)
    # analysis.plot_dFF_singleCell(157, trials)
    # cell plots
    # analysis.plot_dFF(os.path.join(fluofigpath,'cells-combined-cue'))

    # build multiple linear regression
    # arrange the independent variables

    X, y, regr_time = analysis.linear_model()

    MLRResult = analysis.linear_regr(X[:,1:-2,:], y[:,1:-2,:], regr_time)

    x = 1
