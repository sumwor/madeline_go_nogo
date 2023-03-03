import numpy as np
from os.path import join as oj

# class to preprocess calcium 2-photon data

class Suite2pSeries:

    def __init__(self, suite2p):
        suite2p = oj(suite2p, 'plane0')
        Fraw = np.load(oj(suite2p, 'F.npy'))
        ops = np.load(oj(suite2p, 'ops.npy'), allow_pickle=True)
        neuropil = np.load(oj(suite2p, 'Fneu.npy'))
        cells = np.load(oj(suite2p, 'iscell.npy'))
        stat = np.load(oj(suite2p, 'stat.npy'), allow_pickle=True)
        self.Fraw = Fraw
        self.ops = ops
        self.neuropil = neuropil
        self.cells = cells
        self.stat = stat

        F = Fraw - neuropil * 0.7  # subtract neuropil
        # find number of cells
        numcells = np.sum(cells[:, 0] == 1.0)
        # create a new array (Fcells) with F data for each cell
        Fcells = np.zeros((numcells, F.shape[1]))
        counter = 0
        for cell in range(0, len(cells)):
            if cells[cell, 0] == 1.0:  # if ROI is a cell
                Fcells[counter] = F[cell]
                counter += 1
        #        F0 = []
        #        for cell in range(0, Fcells.shape[0]):
        #            include_frames = []
        #            std = np.std(Fcells[cell])
        #            avg = np.mean(Fcells[cell])
        #            for frame in range(0, Fcells.shape[1]):
        #                if Fcells[cell, frame] < std + avg:
        #                    include_frames.append(Fcells[cell, frame])
        #            F0.append(np.mean(include_frames))
        #        dFF = np.zeros(Fcells.shape)
        #        for cell in range(0, Fcells.shape[0]):
        #            for frame in range(0, Fcells.shape[1]):
        #                dFF[cell, frame] = (Fcells[cell, frame] - F0[cell]) / F0[cell]

        F0_AQ = np.zeros(Fcells.shape)
        for cell in range(Fcells.shape[0]):
            F0_AQ[cell] = robust_filter(Fcells[cell], method=12, window=200, optimize_window=2, buffer=False)[:, 0]

        dFF = np.zeros(Fcells.shape)
        for cell in range(0, Fcells.shape[0]):
            for frame in range(0, Fcells.shape[1]):
                dFF[cell, frame] = (Fcells[cell, frame] - F0_AQ[cell, frame]) / F0_AQ[cell, frame]

        self.neural_df = pd.DataFrame(data=dFF.T, columns=[f'neuron{i}' for i in range(numcells)])
        self.neural_df['time'] = np.arange(self.neural_df.shape[0])

    def realign_time(self, reference=None):
        if isinstance(reference, BehaviorMat):
            transform_func = lambda ts: reference.align_ts2behavior(ts)
        if self.neural_df is not None:
            self.neural_df['time'] = transform_func(self.neural_df['time'])

    def calculate_dff(self):
        rois = list(self.neural_df.columns[1:])
        melted = pd.melt(self.neural_df, id_vars='time', value_vars=rois, var_name='roi', value_name='ZdFF')
        return melted


if __name__ == "__main__":
    input_folder = r"Z:\Madeline\processed_data\JUV015\220409\suite2p"
    gn_series = Suite2pSeries(input_folder)
    animal, session = 'JUV011', '211215'
    dff_df = gn_series.calculate_dff()