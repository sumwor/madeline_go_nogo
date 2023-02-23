from abc import abstractmethod

import numpy as np
import pandas as pd
import h5py
import os

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

    fields = ['onset', 'first_lick_in', 'last_lick_out', 'water_valve_on', 'outcome']

    time_unit = 's'

    def __init__(self, animal, session, hfile):
        super().__init__(animal, session)
        self.hfile = h5py.File(hfile, 'r')
        self.animal = animal
        self.session = session
        self.trialN = len(self.hfile['out/result'])
        self.eventlist = self.initialize_node()

    def initialize_node(self):
        code_map = self.code_map
        eventlist = EventNode(None, None, None, None)
        trial_events = np.array(self.hfile['out/GoNG_EventTimes'])
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

        return eventlist

    def to_df(self):
        columns = ['trial'] + self.fields
        result_df = pd.DataFrame(np.full((self.trialN, len(columns)), np.nan), columns=columns)
        result_df['animal'] = self.animal
        result_df['session'] = self.session
        result_df = result_df[['animal', 'session', 'trial'] + self.fields]

        result_df['trial'] = np.arange(1, self.trialN + 1)
        result_df['sound_num'] = pd.Categorical([""] * self.trialN, np.arange(1, 16 + 1), ordered=False)
        result_df['reward'] = pd.Categorical([""] * self.trialN, [-1, 0, 1, 2], ordered=False)
        result_df['go_nogo'] = pd.Categorical([""] * self.trialN, ['go', 'nogo'], ordered=False)
        result_df['licks_out'] = np.full((self.trialN, 1), 0)
        result_df['quality'] = pd.Categorical(["normal"] * self.trialN, ['missed', 'abort', 'normal'], ordered=False)
        result_df['water_valve_amt'] = pd.Categorical([""] * self.trialN, [1, 2, 3], ordered=False)

        for node in self.eventlist:
            # New tone signifies a new trial
            if node.event == 'sound_on':
                result_df.loc[node.trial_index(), 'onset'] = node.etime
                result_df.loc[node.trial_index(), 'sound_num'] = int(self.code_map[node.ecode][1])

            elif node.event == 'in':
                if np.isnan(result_df.loc[node.trial_index(), 'first_lick_in']):
                    result_df.loc[node.trial_index(), 'first_lick_in'] = node.etime
            elif node.event == 'out':
                result_df.loc[node.trial_index(), 'last_lick_out'] = node.etime
                result_df.loc[node.trial_index(), 'licks_out'] += 1
            elif node.event == 'outcome':
                result_df.loc[node.trial_index(), 'outcome'] = node.etime
                outcome = self.code_map[node.ecode][1]
                # quality
                if outcome in ['missed', 'abort']:
                    result_df.loc[node.trial_index(), 'quality'] = outcome
                # reward
                if '_correct_' in outcome:
                    reward = int(outcome[-1]) if outcome[-1].isnumeric() else 0
                    result_df.loc[node.trial_index(), 'reward'] = reward
                else:
                    result_df.loc[node.trial_index(), 'reward'] = -1
                # go nogo
                if outcome.startswith('go') or outcome == 'missed':
                    result_df.loc[node.trial_index(), 'go_nogo'] = 'go'
                elif outcome.startswith('no-go'):
                    result_df.loc[node.trial_index(), 'go_nogo'] = 'nogo'
            elif node.event == 'water_valve':
                num_reward = self.code_map[node.ecode][1]
                result_df.loc[node.trial_index(), 'water_valve_amt'] = int(num_reward)
                result_df.loc[node.trial_index(), 'water_valve_on'] = node.etime

        return result_df

    def output_df(self, outfile, file_type='csv'):
        """
        saves the output of to_df() as a file of the specified type
        """
        if file_type == 'csv':
            self.to_df().to_csv(outfile + '.csv')

    def time_aligner(self):
        # TODO: Implement
        pass


if __name__ == "__main__":
    animal = 'JUV011'
    session = '211215'
    input_folder = fr"\\filenest.diskstation.me\Wilbrecht_file_server\Madeline\processed_data\{animal}\{session}"
    input_file = fr"{animal}_{session}_behaviorLOG.mat"
    x = GoNogoBehaviorMat(animal, session, os.path.join(input_folder, input_file))
    output_file = f"{animal}_{session}_behavior_output"
    x.output_df(os.path.join(input_folder, output_file))

