from abc import abstractmethod

import numpy as np
import pandas as pd
import h5py
import os
from pyPlotHW import StartPlots, StartSubplots
from utility_HW import bootstrap

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import statsmodels.api as sm

import pandas as pd
import pickle

from behavior_base import PSENode, EventNode

# From matlab file get_Headfix_GoNo_EventTimes.m:
# % eventID=1:    2P imaging frame TTL high
# % eventID=2:    2P imaging frame TTL low
# % eventID=3:    left lick in
# % eventID=4:    left lick out
# % eventID=44:   Last left lick out
# % eventID=5:    right lick 1n
# % eventID=6:    right lick out
# % eventID=66:   Last right lick out
# % eventID=7.01:  new trial, Sound 1 ON
# % eventID=7.02:  new trial, Sound 2 ON
# % eventID=7.0n:  new trial, Sound n ON
# % eventID=7.16:  new trial, Sound 16 ON
# % eventID=81.01: Correct No-Go (no-lick), unrewarded outcome
# % eventID=81.02: Correct Go (lick), unrewarded outcome
# % eventID=81.12: Correct Go (lick), 1 drop rewarded after direct delivery
# % eventID=81.22: Correct Go (lick), 2 drops rewarded (valve on)
# % eventID=82:02  False Go (lick), white noise on
# % eventID=83:    Missed to respond
# % eventID=84:    Aborted outcome
# % eventID=9.01:  Water Valve on 1 time (1 reward)
# % eventID=9.02:  Water Valve on 2 times (2 rewards)
# % eventID=9.03:  Water Valve on 3 times (3 rewards)

class BehaviorMat:
    code_map = {}
    fields = []  # maybe divide to event, ev_features, trial_features
    time_unit = None
    eventlist = None

    def __init__(self, animal, session):
        self.animal = animal
        self.session = session
        self.time_aligner = lambda s: s  # provides method to align timestamps

    @abstractmethod
    def todf(self):
        return NotImplemented

    def align_ts2behavior(self, timestamps):
        return self.time_aligner(timestamps)


class GoNogoBehaviorMat(BehaviorMat):
    code_map = {
        3: ('in', 'in'),
        4: ('out', 'out'),
        44: ('out', 'out'),
        81.01: ('outcome', 'no-go_correct_unrewarded'),
        81.02: ('outcome', 'go_correct_unrewarded'),
        81.12: ('outcome', 'go_correct_reward1'),
        81.22: ('outcome', 'go_correct_reward2'),
        82.02: ('outcome', 'no-go_incorrect'),
        83: ('outcome', 'missed'),
        84: ('outcome', 'abort'),
        9.01: ('water_valve', '1'),
        9.02: ('water_valve', '2'),
        9.03: ('water_valve', '3')
    }
    # 7.0n -> sound n on (n = 1-16)
    for i in range(1, 17):
        code_map[(700 + i) / 100] = ('sound_on', str(i))

    fields = ['onset', 'first_lick_in', 'last_lick_out', 'water_valve_on', 'outcome', 'licks','running_speed','time_0']

    time_unit = 's'

    def __init__(self, animal, session, hfile):
        super().__init__(animal, session)
        self.hfile = h5py.File(hfile, 'r')
        self.animal = animal
        self.session = session
        self.trialN = len(self.hfile['out/result'])
        self.eventlist, self.runningSpeed = self.initialize_node()

        # get time_0 (normalize the behavior start time to t=0)
        for node in self.eventlist:
            # New tone signifies a new trial
            if node.event == 'sound_on':
                # get the time of cue onset in trial 1, normalize all following trials
                if node.trial_index() == 1:
                    self.time_0 = node.etime
                    break

        # get the timestamp for aligning with fluorescent data
        if isinstance(hfile, str):
            with h5py.File(hfile, 'r') as hf:
                frame_time = np.array(hf['out/frame_time']).ravel()
        else:
            frame_time = np.array(hfile['out/frame_time']).ravel()
        self.time_aligner = lambda t: frame_time

        # a dictionary to save all needed behavor metrics
        self.saveData = dict()

    def initialize_node(self):
        code_map = self.code_map
        eventlist = EventNode(None, None, None, None)
        trial_events = np.array(self.hfile['out/GoNG_EventTimes'])
        running_speed = np.array(self.hfile['out/run_speed'])

        exp_complexity = None
        struct_complexity = None
        prev_node = None
        duplicate_events = [81.02, 81.12, 81.22, 82.02]
        for i in range(len(trial_events)):
            eventID, eventTime, trial = trial_events[i]
            # check duplicate timestamps
            if prev_node is not None:
                if prev_node.etime == eventTime:
                    if eventID == prev_node.ecode:
                        continue
                    elif eventID not in duplicate_events:
                        print(f"Warning! Duplicate timestamps({prev_node.ecode}, {eventID})" +
                              f"at time {eventTime} in {str(self)}")
                elif eventID in duplicate_events:
                    print(f"Unexpected non-duplicate for {trial}, {code_map[eventID]}, {self.animal}, "
                          f"{self.session}")
            cnode = EventNode(code_map[eventID][0], eventTime, trial, eventID)
            eventlist.append(cnode)
            prev_node = cnode

        return eventlist, running_speed

    def to_df(self):
        columns = ['trial'] + self.fields
        result_df = pd.DataFrame(np.full((self.trialN, len(columns)), np.nan), columns=columns)
        result_df['animal'] = self.animal
        result_df['session'] = self.session
        result_df['time_0'] = 0 # update later
        result_df = result_df[['animal', 'session', 'trial'] + self.fields]

        result_df['trial'] = np.arange(1, self.trialN + 1)
        result_df['sound_num'] = pd.Categorical([""] * self.trialN, np.arange(1, 16 + 1), ordered=False)
        result_df['reward'] = pd.Categorical([""] * self.trialN, [-1, 0, 1, 2], ordered=False)
        result_df['go_nogo'] = pd.Categorical([""] * self.trialN, ['go', 'nogo'], ordered=False)
        result_df['licks_out'] = np.full((self.trialN, 1), 0)
        result_df['quality'] = pd.Categorical(["normal"] * self.trialN, ['missed', 'abort', 'normal'], ordered=False)
        result_df['water_valve_amt'] = pd.Categorical([""] * self.trialN, [1, 2, 3], ordered=False)


        # add another entry to record all the licks
        result_df['licks'] = [[] for _ in range(self.trialN)] # convert to np.array later
        result_df['choice'] = pd.Categorical([""] * self.trialN, [-4, -3, -2, -1, 0, 1, 2], ordered=False)

        for node in self.eventlist:
            # New tone signifies a new trial
            if node.event == 'sound_on':
                # get the time of cue onset in trial 1, normalize all following trials
                if node.trial_index() == 1:
                    time_0 = node.etime
                    result_df['time_0'] = time_0

                result_df.at[node.trial_index()-1, 'onset'] = node.etime-time_0
                result_df.at[node.trial_index()-1, 'sound_num'] = int(self.code_map[node.ecode][1])


            elif node.event == 'in':
                # add a list contain all licks in the trial

                if not result_df.at[node.trial_index()-1, 'licks']:
                    result_df.at[node.trial_index()-1, 'licks'] = [node.etime-time_0]
                else:
                    result_df.at[node.trial_index()-1, 'licks'].append(node.etime-time_0)

                if np.isnan(result_df.loc[node.trial_index()-1, 'first_lick_in']):
                    result_df.loc[node.trial_index()-1, 'first_lick_in'] = node.etime-time_0
            elif node.event == 'out':
                result_df.loc[node.trial_index()-1, 'last_lick_out'] = node.etime-time_0
                result_df.loc[node.trial_index()-1, 'licks_out'] += 1
            elif node.event == 'outcome':
                result_df.loc[node.trial_index()-1, 'outcome'] = node.etime-time_0
                outcome = self.code_map[node.ecode][1]
                # quality
                if outcome in ['missed', 'abort']:
                    result_df.loc[node.trial_index()-1, 'quality'] = outcome
                # reward
                if '_correct_' in outcome:
                    reward = int(outcome[-1]) if outcome[-1].isnumeric() else 0
                    result_df.loc[node.trial_index()-1, 'reward'] = reward
                else:
                    result_df.loc[node.trial_index()-1, 'reward'] = -1
                # go nogo
                if outcome.startswith('go') or outcome == 'missed':
                    result_df.loc[node.trial_index()-1, 'go_nogo'] = 'go'
                elif outcome.startswith('no-go'):
                    result_df.loc[node.trial_index()-1, 'go_nogo'] = 'nogo'
            elif node.event == 'water_valve':
                num_reward = self.code_map[node.ecode][1]
                result_df.loc[node.trial_index()-1, 'water_valve_amt'] = int(num_reward)
                result_df.loc[node.trial_index()-1, 'water_valve_on'] = node.etime-time_0

        # align running speed to trials
        self.runningSpeed[:, 0] = self.runningSpeed[:, 0] - time_0

        result_df['running_speed'] = [[] for _ in range(self.trialN)]
        result_df['running_time'] = [[] for _ in range(self.trialN)]

        # remap reward to self.outcome
        # -4: probeSti, no lick;
        # -3: probSti, lick;
        # -2: miss;
        # -1: false alarm;
        # 0: correct reject;
        # 1/2: hit for reward amount

        for tt in range(self.trialN):
            t_start = result_df.onset[tt]
            if tt<self.trialN-1:
                t_end = result_df.onset[tt+1]
                result_df.at[tt, 'running_speed'] = self.runningSpeed[np.logical_and(self.runningSpeed[:,0]>=t_start-3, self.runningSpeed[:,0]<t_end),1].tolist()
                result_df.at[tt, 'running_time'] = self.runningSpeed[
                    np.logical_and(self.runningSpeed[:, 0] >= t_start - 3, self.runningSpeed[:, 0] < t_end), 0].tolist()
            elif tt == self.trialN-1:
                result_df.at[tt, 'running_speed'] = self.runningSpeed[self.runningSpeed[:, 0] >= t_start-3, 1].tolist()
                result_df.at[tt, 'running_time'] = self.runningSpeed[self.runningSpeed[:, 0] >= t_start - 3, 0].tolist()
            # remap reward to outcome
            if result_df.sound_num[tt] in [9, 10, 11, 12, 13, 14, 15, 16]:
                result_df.at[tt, 'choice'] = -4 if result_df.at[tt, 'licks_out']==0 else -3
            elif result_df.sound_num[tt] in [1, 2, 3, 4]:
                result_df.at[tt, 'choice'] = -2 if result_df.at[tt, 'licks_out']==0 else result_df.at[tt, 'reward']
            elif result_df.sound_num[tt] in [5, 6, 7, 8]:
                result_df.at[tt, 'choice'] = 0 if result_df.at[tt, 'licks_out']==0 else -1

        # save the data into self
        self.DF = result_df

        self.saveData['behDF'] = result_df

        return result_df

    def output_df(self, outfile, file_type='csv'):
        """
        saves the output of to_df() as a file of the specified type
        """
        if file_type == 'csv':
            self.DF.to_csv(outfile + '.csv')

    ### plot methods for behavior data
    # should add outputs later, for summary plots

    def beh_session(self, save_path):
        # plot the outcome according to trials
        trialNum = np.arange(self.trialN)

        beh_plots = StartPlots()
        # hit trials
        beh_plots.ax.scatter(trialNum[self.DF.choice == 2], np.array(self.DF.choice[self.DF.choice == 2]),
                   s=100, marker='o')

        # miss trials
        beh_plots.ax.scatter(trialNum[self.DF.choice == -2], self.DF.choice[self.DF.choice == -2], s=100,
                   marker='x')

        # false alarm
        beh_plots.ax.scatter(trialNum[self.DF.choice == -1], self.DF.choice[self.DF.choice == -1], s=100,
                   marker='*')

        # correct rejection
        beh_plots.ax.scatter(trialNum[self.DF.choice == 0], self.DF.choice[self.DF.choice == 0], s=100,
                   marker='.')

        # probe lick
        beh_plots.ax.scatter(trialNum[self.DF.choice == -3], self.DF.choice[self.DF.choice == -3], s=100,
                   marker='v')

        # proble no lick
        beh_plots.ax.scatter(trialNum[self.DF.choice == -4], self.DF.choice[self.DF.choice == -4], s=100,
                   marker='^')

        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        beh_plots.ax.set_title('Session summary')
        beh_plots.ax.set_xlabel('Trials')
        beh_plots.ax.set_ylabel('Outcome')
        leg = beh_plots.legend(['Hit', 'Miss', 'False alarm', 'Correct rejection', 'Probe lick', 'Probe miss'])

        #legend.get_frame().set_linewidth(0.0)
        #legend.get_frame().set_facecolor('none')
        beh_plots.fig.set_figwidth(40)
        plt.show()

        # save the plot
        beh_plots.save_plot('Behavior summary.svg', 'svg', save_path)
        beh_plots.save_plot('Behavior summary.tif', 'tif', save_path)
        # trialbytrial.beh_session()

    def d_prime(self):
        # calculate d prime
        nTrialGo = np.sum(np.logical_or(self.DF['choice']==2, self.DF['choice']==-2))
        nTrialNoGo = np.sum(np.logical_or(self.DF['choice'] == -1, self.DF['choice'] == 0))
        Hit_rate = np.sum(self.DF['choice'] == 2) / nTrialGo
        FA_rate = np.sum(self.DF['choice'] == -1) / nTrialNoGo

        d_prime = norm.ppf(Hit_rate) - norm.ppf(FA_rate)

        self.saveData['d-prime'] = d_prime

    def psycho_curve(self, save_path):
        # get variables
        # hfile['out']['sound_freq'][0:-1]
        # use logistic regression
        # L(P(go)/(1-P(go)) = beta0 + beta_Go*S_Go + beta_NoGo * S_NoGo
        # reference: Breton-Provencher, 2022

        numSound = 16

        goCueInd = np.arange(1, 5)
        nogoCueInd = np.arange(5, 9)
        probeCueInd = np.arange(9, 17)

        goFreq = np.array([6.49, 7.07, 8.46, 9.17])
        nogoFreq = np.array([10.9, 11.9, 14.14, 15.41])
        probeFreq = np.array([6.77, 7.73, 8.81, 9.71, 10.29, 11.38, 12.97, 14.76])
        midFreq = (9.17+10.9)/2

        # %%
        # psychometric curve
        sound = np.arange(1, numSound + 1)
        numGo = np.zeros(numSound)

        # sort sound, base on the frequency
        soundIndTotal = np.concatenate((goCueInd, nogoCueInd, probeCueInd))
        soundFreqTotal = np.concatenate((goFreq, nogoFreq, probeFreq))

        sortedInd = np.argsort(soundFreqTotal)
        sortedIndTotal = soundIndTotal[sortedInd]
        sortedFreqTotal = soundFreqTotal[sortedInd]
        stiSortedInd = np.where(np.in1d(sortedIndTotal, np.concatenate((goCueInd, nogoCueInd))))[0]
        probeSortedInd = np.where(np.in1d(sortedIndTotal, probeCueInd))[0]

        for ss in range(len(numGo)):
            numGo[ss] = np.sum(np.logical_and(self.DF.sound_num == ss+1, np.logical_or(self.DF.reward == 2, self.DF.reward==-1)))
            sound[ss] = np.sum(self.DF.sound_num == ss+1)

        sortednumGo = numGo[sortedInd]
        sortednumSound = sound[sortedInd]


        # fit logistic regression
        y = sortednumGo[stiSortedInd] / sortednumSound[stiSortedInd]
        x = np.array((sortedFreqTotal[stiSortedInd],sortedFreqTotal[stiSortedInd]))
        x[0, 4:] = 0 # S_Go
        x[1, 0:4] = 0 # S_NoGo
        x = x.transpose()
        x = sm.add_constant(x)

        if np.count_nonzero(~np.isnan(y)) > 1:
            model = sm.Logit(y, x).fit()

            # save the data
            self.saveData['L-fit'] = model.params

            # generating x for model prediction
            x_pred = np.array((np.linspace(6,16, 50), np.linspace(6,16, 50)))
            x_pred[0,x_pred[0,:]>midFreq] = 0
            x_pred[1,x_pred[1,:]<midFreq] = 0
            x_pred = x_pred.transpose()
            x_pred = sm.add_constant(x_pred)
            y_pred = model.predict(x_pred)

            #xNoGo_fit = np.linspace(6,16, 50)
            #yNoGo_fit = self.softmax(result_NoGo.x, xNoGo_fit-midFreq)

            psyCurve = StartPlots()
            psyCurve.ax.scatter(sortedFreqTotal[stiSortedInd], sortednumGo[stiSortedInd] / sortednumSound[stiSortedInd])
            psyCurve.ax.scatter(sortedFreqTotal[probeSortedInd], sortednumGo[probeSortedInd] / sortednumSound[probeSortedInd])
            psyCurve.ax.plot(np.linspace(6,16, 50), y_pred)
            #psyCurve.ax.plot(xNoGo_fit, yNoGo_fit)

            psyCurve.ax.plot([midFreq, midFreq], [0, 1], linestyle='--')

            # ax.legend()
            psyCurve.ax.set_xlabel('Sound (kHz)')
            psyCurve.ax.set_ylabel('Go rate')
            plt.show()

            psyCurve.save_plot('psychometric.svg', 'svg', save_path)
            psyCurve.save_plot('psychometric summary.tif', 'tif', save_path)
        else:
            self.saveData['L-fit'] = np.nan


    def lick_rate(self, save_path):
        lickTimesH = np.array([])  # lick rate for Hit trials
        lickTimesFA =np.array([])   # lick rage for False alarm trials
        lickTimesProbe = np.array([])
        #lickSoundH = np.array(self.DF.sound_num[self.DF.reward==2])
        #lickSoundFA = np.array(self.DF.sound_num[self.DF.reward==-1])
        lickSoundH = np.array([])
        lickSoundFA = np.array([])
        lickSoundProbe = np.array([])

        binSize = 0.05  # use a 0.05s window for lick rate
        edges = np.arange(0 + binSize / 2, 5 - binSize / 2, binSize)

        for tt in range(self.trialN):
            if self.DF.choice[tt] == 2:
                lickTimesH = np.concatenate((lickTimesH, (np.array(self.DF.licks[tt]) - self.DF.onset[tt])))
                lickSoundH = np.concatenate((lickSoundH, np.ones(len(np.array(self.DF.licks[tt])))*self.DF.sound_num[tt]))
            elif self.DF.choice[tt] == -1:
                lickTimesFA = np.concatenate((lickTimesFA, (np.array(self.DF.licks[tt]) - self.DF.onset[tt])))
                lickSoundFA = np.concatenate(
                    (lickSoundFA, np.ones(len(np.array(self.DF.licks[tt]))) * self.DF.sound_num[tt]))
            elif self.DF.choice[tt] == -3:
                lickTimesProbe = np.concatenate((lickTimesProbe, (np.array(self.DF.licks[tt]) - self.DF.onset[tt])))
                lickSoundProbe = np.concatenate(
                    (lickSoundProbe, np.ones(len(np.array(self.DF.licks[tt]))) * self.DF.sound_num[tt]))

        lickRateH = np.zeros((len(edges), 4))
        lickRateFA = np.zeros((len(edges), 4))
        lickRateProbe = np.zeros((len(edges), 8))

        for ee in range(len(edges)):
            for ssH in range(4):
                lickRateH[ee,ssH] = sum(
                    np.logical_and(lickTimesH[lickSoundH==(ssH+1)] <= edges[ee] + binSize / 2, lickTimesH[lickSoundH==(ssH+1)] > edges[ee] - binSize / 2)) / (
                                    binSize * sum(np.logical_and(np.array(self.DF.choice == 2), np.array(self.DF.sound_num)==(ssH+1))))
            for ssFA in range(4):
                lickRateFA[ee, ssFA] = sum(
                    np.logical_and(lickTimesFA[lickSoundFA == (ssFA + 5)] <= edges[ee] + binSize / 2,
                                   lickTimesFA[lickSoundFA == (ssFA + 5)] > edges[ee] - binSize / 2)) / (
                                             binSize * sum(np.logical_and(np.array(self.DF.choice == -1),
                                                                          np.array(self.DF.sound_num) == (ssFA + 5))))
            for ssProbe in range(8):
                lickRateProbe[ee, ssProbe] = sum(np.logical_and(lickTimesProbe[lickSoundProbe == (ssProbe + 9)] <= edges[ee] + binSize / 2,
                                   lickTimesProbe[lickSoundProbe == (ssProbe + 9)] > edges[ee] - binSize / 2)) / (
                                               binSize * sum(np.logical_and(np.array(self.DF.choice == -3),
                                                                            np.array(self.DF.sound_num) == (ssProbe + 9))))

        # save data
        self.saveData['lickRate'] = pd.DataFrame({'edges': edges})
        for ss in np.unique(self.DF['sound_num']):
            if ss < 5: # hit
                self.saveData['lickRate'][str(ss)] = lickRateH[:,ss-1]
            elif ss >= 5 and ss < 9:
                self.saveData['lickRate'][str(ss)] = lickRateFA[:,ss-5]
            else:
                self.saveData['lickRate'][str(ss)] = lickRateProbe[:,ss-9]

        # plot the response time distribution in hit/false alarm trials
        lickRate = StartPlots()

        lickRate.ax.plot(edges, np.nansum(lickRateH, axis=1))
        lickRate.ax.plot(edges, np.nansum(lickRateFA, axis=1))
        lickRate.ax.plot(edges, np.nansum(lickRateProbe, axis=1))

        lickRate.ax.set_xlabel('Time from cue (s)')
        lickRate.ax.set_ylabel('Frequency (Hz)')
        lickRate.ax.set_title('Lick rate (Hz)')

        lickRate.ax.legend(['Hit', 'False alarm','Probe lick'])

        plt.show()

        lickRate.save_plot('lick rate.svg', 'svg', save_path)
        lickRate.save_plot('lick rate.tif', 'tif', save_path)
        # separate the lick rate into different frequencies
        # fig, axs = plt.subplots(2, 4, figsize=(8, 8), sharey=True)
        #
        # # plot hit
        # for ii in range(4):
        #     axs[0, ii].plot(edges, lickRateH[:,ii])
        #     axs[0, ii].set_title(['Sound # ', str(ii + 1)])
        #
        # # plot false alarm
        # for jj in range(4):
        #     axs[1, jj].plot(edges, lickRateFA[:, jj])
        #     axs[1, jj].set_title(['Sound # ', str(jj + 5)])
        #
        # plt.subplots_adjust(top=0.85)
        # plt.show()

    def response_time(self, save_path):
        """
        aligned_to: time point to be aligned. cue/outcome/licks
        """
        rt = np.zeros(self.trialN)

        for tt in range(len(rt)):
            rt[tt] = self.DF.first_lick_in[tt] - self.DF.onset[tt]

        # plot the response time distribution in hit/false alarm trials
        rtPlot = StartPlots()
        rtHit, bins, _ = rtPlot.ax.hist(rt[np.array(self.DF.choice) == 2], bins=100, range=[0, 0.5], density=True)
        rtFA, _, _ = rtPlot.ax.hist(rt[np.array(self.DF.choice) == -1], bins=bins, density=True)
        #_ = rtPlot.ax.hist(rt[np.array(self.DF.choice) == -3], bins=bins, density=True)

        # save the data
        self.saveData['rt'] = pd.DataFrame({'rtHit':rtHit, 'rtFA': rtFA, 'bin': bins[1:]})
        rtPlot.ax.set_xlabel('Response time (s)')
        rtPlot.ax.set_ylabel('Frequency (%)')
        rtPlot.ax.set_title('Response time (s)')

        rtPlot.ax.legend(['Hit', 'False alarm'])

        plt.show()

        rtPlot.save_plot('Response time.svg', 'svg', save_path)
        rtPlot.save_plot('Response time.tif', 'tif', save_path)
        # separate the response time into different frequencies
        # fig, axs = plt.subplots(2,4,figsize=(8, 8), sharey=True)
        #
        # # plot hit
        # for ii in range(4):
        #     if ii == 0:
        #         _, bins, _ = axs[0,ii].hist(rt[np.logical_and(np.array(self.DF.reward) == 2, self.DF.sound_num.array==(ii+1))], bins=50, range=[0, .5])
        #     else:
        #         _ = axs[0,ii].hist(rt[np.logical_and(np.array(self.DF.reward) == 2, self.DF.sound_num.array==(ii+1))], bins=bins)
        #     axs[0, ii].set_title(['Sound # ', str(ii+1)])
        #
        # # plot false alarm
        # for jj in range(4):
        #     _ = axs[1, jj].hist(rt[np.logical_and(np.array(self.DF.reward) == -1, self.DF.sound_num.array == (jj + 5))],bins=bins)
        #     axs[1, jj].set_title(['Sound # ', str(jj + 5)])
        #
        # plt.subplots_adjust(top=0.85)
        # plt.show()

    def ITI_distribution(self, save_path):
        ITIH = []  # lick rate for Hit trials
        ITIFA = []  # lick rage for False alarm trials
        ITIProbe = []

        ITISoundH = np.array(self.DF.sound_num[np.logical_and(self.DF.reward==2, self.DF.trial!=self.trialN)])
        ITISoundFA = np.array(self.DF.sound_num[np.logical_and(self.DF.reward==-1,self.DF.trial!=self.trialN)])

        binSize = 0.05  # use a 0.05s window for lick rate
        edges = np.arange(0 + binSize / 2, 20- binSize / 2, binSize)

        for tt in range(self.trialN-1):
            if self.DF.reward[tt] == 2:

                ITIH.append(self.DF.onset[tt+1] - self.DF.outcome[tt])


            elif self.DF.reward[tt] == -1:
                ITIFA.append(self.DF.onset[tt+1] - self.DF.outcome[tt])


        # convert to np.arrays
        ITIH = np.array(ITIH)
        ITIFA = np.array(ITIFA)

        ITIRateH = np.zeros((len(edges), 4))
        ITIRateFA = np.zeros((len(edges), 4))

        for ee in range(len(edges)):
            for ssH in range(4):
                ITIRateH[ee, ssH] = sum(
                    np.logical_and(ITIH[ITISoundH == (ssH + 1)] <= edges[ee] + binSize / 2,
                                   ITIH[ITISoundH == (ssH + 1)] > edges[ee] - binSize / 2))
            for ssFA in range(4):
                ITIRateFA[ee, ssFA] = sum(
                    np.logical_and(ITIFA[ITISoundFA == (ssFA + 5)] <= edges[ee] + binSize / 2,
                                   ITIFA[ITISoundFA == (ssFA + 5)] > edges[ee] - binSize / 2))

        # plot
        ITIPlot = StartPlots()

        ITIPlot.ax.plot(edges, np.sum(ITIRateH, axis=1))
        ITIPlot.ax.plot(edges, np.sum(ITIRateFA, axis=1))

        ITIPlot.ax.set_xlabel('ITI duration (s)')
        ITIPlot.ax.set_ylabel('Trials')
        ITIPlot.ax.set_title('ITI distribution')

        ITIPlot.ax.legend(['Hit', 'False alarm'])

        plt.show()

        ITIPlot.save_plot('ITI.svg', 'svg', save_path)
        ITIPlot.save_plot('ITI.tif', 'tif', save_path)
        # # separate the lick rate into different frequencies
        # fig, axs = plt.subplots(2, 4, figsize=(8, 8), sharey=True)
        #
        # # plot hit
        # for ii in range(4):
        #     axs[0, ii].plot(edges, ITIRateH[:, ii])
        #     axs[0, ii].set_title(['Sound # ', str(ii + 1)])
        #
        # # plot false alarm
        # for jj in range(4):
        #     axs[1, jj].plot(edges, ITIRateFA[:, jj])
        #     axs[1, jj].set_title(['Sound # ', str(jj + 5)])
        #
        # plt.subplots_adjust(top=0.85)
        # plt.show()

    def running_aligned(self, aligned_to, save_path):
        """
        aligned_to: reference time point. onset/outcome/lick
        """
        # aligned to cue onset and interpolate the results
        interpT = np.arange(-3,5,0.1)
        numBoot = 1000

        if aligned_to == "onset": # aligned to cue time
            run_aligned = np.full((len(interpT), self.trialN), np.nan)
            for tt in range(self.trialN-1):
                speed = np.array(self.DF.running_speed[tt])
                speedT = np.array(self.DF.running_time[tt])
                if speed.size != 0:
                    t = speedT - self.DF.onset[tt]
                    y = speed
                    y_interp = np.interp(interpT, t, y)
                    run_aligned[:,tt] = y_interp

            # bootstrap
            BootH = bootstrap(run_aligned[:, self.DF.choice == 2], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootFA = bootstrap(run_aligned[:, self.DF.choice == -1], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootMiss = bootstrap(run_aligned[:, self.DF.choice == -2], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootCorRej = bootstrap(run_aligned[:, self.DF.choice == 0], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootProbeLick = bootstrap(run_aligned[:, self.DF.choice == -3], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootProbeNoLick = bootstrap(run_aligned[:, self.DF.choice == -4], dim=1, dim0 = len(interpT),n_sample=numBoot)


        elif aligned_to == 'outcome':
            run_aligned = np.full((len(interpT), self.trialN), np.nan)
            for tt in range(self.trialN-1):
                speed = np.array(self.DF.running_speed[tt])
                speedT = np.array(self.DF.running_time[tt])
                if speed.size != 0:
                    t = speedT - self.DF.outcome[tt]
                    y = speed
                    y_interp = np.interp(interpT, t, y)
                    run_aligned[:,tt] = y_interp
            # bootstrap
            BootH = bootstrap(run_aligned[:, self.DF.choice == 2], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootFA = bootstrap(run_aligned[:, self.DF.choice == -1], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootMiss = bootstrap(run_aligned[:, self.DF.choice == -2], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootCorRej = bootstrap(run_aligned[:, self.DF.choice == 0], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootProbeLick = bootstrap(run_aligned[:, self.DF.choice == -3], dim=1, dim0 = len(interpT),n_sample=numBoot)
            BootProbeNoLick = bootstrap(run_aligned[:, self.DF.choice == -4], dim=1, dim0 = len(interpT),n_sample=numBoot)


        elif aligned_to == 'licks':
            run_aligned = []
            for tt in range(self.trialN-1):
                # loop through licks
                numLicks = len(self.DF.licks[tt])
                temp_aligned = np.full((len(interpT), numLicks), np.nan)
                for ll in range(numLicks):
                    speed = np.array(self.DF.running_speed[tt])
                    speedT = np.array(self.DF.running_time[tt])
                    if speed.size != 0:
                        t = speedT - self.DF.licks[tt][ll]
                        y = speed
                        y_interp = np.interp(interpT, t, y)
                        temp_aligned[:,ll] = y_interp

                run_aligned.append(temp_aligned)

            # bootstrap
            BootH = bootstrap(self.concat_data(run_aligned, 2), dim=1, dim0 = len(interpT), n_sample=numBoot)
            BootFA = bootstrap(self.concat_data(run_aligned, -1), dim=1, dim0 = len(interpT), n_sample=numBoot)
            BootMiss = bootstrap(self.concat_data(run_aligned, -2), dim=1, dim0 = len(interpT), n_sample=numBoot)
            BootCorRej = bootstrap(self.concat_data(run_aligned, 0), dim=1, dim0 = len(interpT), n_sample=numBoot)
            BootProbeLick = bootstrap(self.concat_data(run_aligned, -3), dim=1, dim0 = len(interpT), n_sample=numBoot)
            BootProbeNoLick = bootstrap(self.concat_data(run_aligned, -4), dim=1, dim0 = len(interpT), n_sample=numBoot)

        # save the data
        self.saveData['run_' + aligned_to] = {'interpT': interpT, 'run_aligned': run_aligned}


        runPlot = StartSubplots(2,3,ifSharex=True, ifSharey=True)

        runPlot.fig.suptitle('Aligned to '+aligned_to)

        runPlot.ax[0,0].plot(interpT, BootH['bootAve'])
        runPlot.ax[0,0].fill_between(interpT, BootH['bootLow'], BootH['bootHigh'],alpha=0.2)
        runPlot.ax[0,0].set_title('Hit')
        runPlot.ax[0,0].set_ylabel('Running speed')

        runPlot.ax[0,1].plot(interpT, BootFA['bootAve'])
        runPlot.ax[0,1].fill_between(interpT, BootFA['bootLow'], BootFA['bootHigh'],alpha=0.2)
        runPlot.ax[0,1].set_title('False alarm')

        runPlot.ax[0,2].plot(interpT, BootMiss['bootAve'])
        runPlot.ax[0,2].fill_between(interpT, BootMiss['bootLow'], BootMiss['bootHigh'],alpha=0.2)
        runPlot.ax[0,2].set_title('Miss')

        runPlot.ax[1,0].plot(interpT, BootCorRej['bootAve'])
        runPlot.ax[1,0].fill_between(interpT, BootCorRej['bootLow'], BootCorRej['bootHigh'], alpha=0.2)
        runPlot.ax[1,0].set_title('Correct rejection')
        runPlot.ax[1,0].set_xlabel('Time (s)')
        runPlot.ax[1,0].set_ylabel('Running speed')

        runPlot.ax[1,1].plot(interpT, BootProbeLick['bootAve'])
        runPlot.ax[1,1].fill_between(interpT, BootProbeLick['bootLow'], BootProbeLick['bootHigh'], alpha=0.2)
        runPlot.ax[1,1].set_title('Probe lick')
        runPlot.ax[1,1].set_xlabel('Time (s)')

        runPlot.ax[1,2].plot(interpT, BootProbeNoLick['bootAve'])
        runPlot.ax[1,2].fill_between(interpT, BootProbeNoLick['bootLow'], BootProbeNoLick['bootHigh'], alpha=0.2)
        runPlot.ax[1,2].set_title('Probe nolick')
        runPlot.ax[1,2].set_xlabel('Time (s)')

        runPlot.save_plot('Running' + aligned_to + '.svg', 'svg', save_path)
        runPlot.save_plot('Running' + aligned_to + '.svg', 'tif', save_path)
    ### analysis methods for behavior

    def save_anlaysis(self, save_path):
        # save the analysis result
        # save the analysis result
        # Open a file for writing
        save_file = os.path.join(save_path, 'behAnalysis.pickle')
        with open(save_path, 'wb') as f:
            # Use pickle to dump the dictionary into the file
            pickle.dump(self.saveData, f)

    def fit_softmax(self, x, y):
        # Fit the softmax function to the data using scipy.optimize.minimize
        result = minimize(self.neg_log_likelihood, [0.5], args=(x, y))


    #define the softmax function
    def softmax(self, beta, x):

        return 1 / (1 + (np.exp(beta*x)))

    # Define the negative log-likelihood function
    def neg_log_likelihood(self, beta, x, y):
        p = self.softmax(beta, x)
        return -np.sum(y * np.log(p))

    def concat_data(self, data, outcome):
        # concatenate a list whose elements have different size
        # extract the trials with certain outcome and concatenate the running speed aligned to licks
        trialInd = [i for i, e in enumerate(self.DF.choice) if e == outcome]
        output = np.array([])
        for tt in trialInd:
            if tt < self.trialN-1:
                output = data[tt] if not output.size else np.concatenate((output, data[tt]),axis=1)

        return output


if __name__ == "__main__":
    animal = 'JUV011'
    session = '211215'
    #input_folder = "C:\\Users\\hongl\\Documents\\GitHub\\madeline_go_nogo\\data"
    input_folder = "C:\\Users\\xiachong\\Documents\\GitHub\\madeline_go_nogo\\data"
    input_file = "JUV015_220409_behaviorLOG.mat"
    x = GoNogoBehaviorMat(animal, session, os.path.join(input_folder, input_file))
    x.to_df()
    output_file = r"C:\Users\xiachong\Documents\GitHub\madeline_go_nogo\data\JUV015_220409_behavior_output"
    #output_file = r"C:\\Users\\hongl\\Documents\\GitHub\\madeline_go_nogo\\data\JUV015_220409_behavior_output"
    x.output_df(output_file)
    #x.d_prime()

    # test code for plot

    # x.psycho_curve()
    #x.response_time()
    # x.lick_rate()
    #x.ITI_distribution()
    x.running_aligned('onset')