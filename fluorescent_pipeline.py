# pipeline for fluorescent analysis
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

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

    def process_X(self, regr_time, choiceList, rewardList, nTrials, nCells, trial):
        X = np.zeros((11,len(regr_time)))
        #
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
            # for cc in range(nCells):
            #     temp_dFF = self.dFF_aligned[:, trial, cc]
            #     Y[cc, tt] = np.nanmean(
            #         temp_dFF[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])

        return X

    def process_Y(self, regr_time, nCells, trial):

        Y = np.zeros((nCells, len(regr_time)))
        tStep = np.nanmean(np.diff(regr_time))

        for tt in range(len(regr_time)):
            t_start = regr_time[tt] - tStep / 2
            t_end = regr_time[tt] + tStep / 2

            # dependent variable: dFF
            for cc in range(nCells):
                temp_dFF = self.dFF_aligned[:, trial, cc]
                Y[cc, tt] = np.nanmean(temp_dFF[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])

        return Y

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

        # define the function for parallel computing


        n_jobs = -1

        # Parallelize the loop over `trial`
        independent_X = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(self.process_X)(regr_time, choiceList, rewardList, nTrials, nCells, trial) for trial in tqdm(range(nTrials)))
        dFF_Y = Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(self.process_Y)(regr_time, nCells, trial) for trial in tqdm(range(nTrials)))
        # convert
        # for trial in tqdm(range(nTrials)):
        #         # need to determine whether to use exact frequency or go/no go/probe
        #         independent_X[0, trial, :] = np.ones(len(regr_time)) * self.beh['sound_num'][trial]
        #
        #         # choice: lick = 1; no lick = -1
        #         independent_X[1, trial, :] = np.ones(len(regr_time)) * (choiceList[trial+1] if trial<nTrials-1 else np.nan)
        #         independent_X[2, trial, :] = np.ones(len(regr_time)) * (choiceList[trial])
        #         independent_X[3, trial, :] = np.ones(len(regr_time)) * (choiceList[trial-1] if trial>0 else np.nan)
        #
        #         # reward
        #         independent_X[4, trial, :] = np.ones(len(regr_time)) * (rewardList[trial+1] if trial<nTrials-1 else np.nan)
        #         independent_X[5, trial, :] = np.ones(len(regr_time)) * rewardList[trial]
        #         independent_X[6, trial, :] = np.ones(len(regr_time)) * (rewardList[trial-1] if trial>0 else np.nan)
        #
        #         # interaction
        #         independent_X[7, trial, :] = independent_X[1, trial, :] * independent_X[4, trial, :]
        #         independent_X[8, trial, :] = independent_X[2, trial, :] * independent_X[5, trial, :]
        #         independent_X[9, trial, :] = independent_X[3, trial, :] * independent_X[6, trial, :]
        #         # running speed
        #         for tt in range(len(regr_time)):
        #             t_start = regr_time[tt] - tStep / 2
        #             t_end = regr_time[tt] + tStep / 2
        #             temp_run = self.run_aligned[:,trial]
        #             independent_X[10, trial, tt] = np.nanmean(temp_run[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])
        #
        #             # dependent variable: dFF
        #             for cc in range(nCells):
        #                 temp_dFF = self.dFF_aligned[:, trial, cc]
        #                 dFF_Y[cc, trial, tt] = np.nanmean(temp_dFF[np.logical_and(self.interpT > t_start, self.interpT <= t_end)])

        return np.array(independent_X), np.array(dFF_Y), regr_time

    def linear_regr(self, X, y, regr_t):
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
        # fit the re
        model = LinearRegression().fit(x, Y)


# define the function for parallel computing

if __name__ == "__main__":
    beh_file = r"C:\Users\xiachong\Documents\GitHub\madeline_go_nogo\data\JUV015_220409_behavior_output.csv"
    fluo_file = r"C:\Users\xiachong\Documents\GitHub\JUV015_220409_dff_df_file.csv"
    animal, session = 'JUV011', '211215'
    # dff_df = gn_series.calculate_dff(melt=False)

    beh_data = pd.read_csv(beh_file)
    fluo_data = pd.read_csv(fluo_file)

    # build the linear regression model
    analysis = fluoAnalysis(beh_file,fluo_file)
    analysis.align_fluo_beh()

    # build multiple linear regression
    # arrange the independent variables

    X, y, regr_time = analysis.linear_model()


    x = 1