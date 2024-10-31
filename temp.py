for f in tqdm(range(nFiles)):
    saveDataPath = os.path.join(group_adt.iloc[f]['fluo_analysis_dir'], 'decoding')
    saveDataFile = os.path.join(saveDataPath, 'decodingResult_noRun_' + classifier + '.pickle')
    # load the pickle file
    with open(saveDataFile, 'rb') as file:
        decodingResult = pickle.load(file)

    # initialize variables
    if f == 0:
        decodingVars = decodingResult['var']
        regr_time = decodingResult['time']
        decodingSummary_ADT = {}
        for var in decodingVars:
            decodingSummary_ADT[var] = {}
            decodingSummary_ADT[var]['accuracy'] = np.empty((len(regr_time),
                                                             0))
            decodingSummary_ADT[var]['ctrl_accuracy'] = np.empty((len(regr_time),
                                                                  0))
            decodingSummary_ADT[var]['accuracy_nNeurons'] = np.empty((len(regr_time), 0))
            decodingSummary_ADT[var]['ctrl_nNeurons'] = np.empty((len(regr_time), 0))

            # load predictions based on different trial types
            decodingSummary_ADT[var]['prediction_accuracy'] = {}
            decodingSummary_ADT[var]['prediction_accuracy_ctrl'] = {}
            for key in decodingResult[var]['prediction_accuracy'].keys():
                decodingSummary_ADT[var]['prediction_accuracy'][key] = np.empty((len(regr_time), 0))
                decodingSummary_ADT[var]['prediction_accuracy_ctrl'][key] = np.empty((len(regr_time), 0))

            # load ensemble size result

            maxNeurons = 300
            nNeurons = np.arange(0, 300, 10)
            decodingSummary_ADT[var]['ensemble_accuracy0-1'] = np.full((len(nNeurons), nFiles), np.nan)
            decodingSummary_ADT[var]['ensemble_accuracy1-2'] = np.full((len(nNeurons), nFiles), np.nan)
            decodingSummary_ADT[var]['ensemble_size'] = decodingResult[var]['decode_realSize']['nNeurons']
            decodingSummary_ADT[var]['ensemble_ctrl0-1'] = np.full((len(nNeurons), nFiles), np.nan)
            decodingSummary_ADT[var]['ensemble_ctrl1-2'] = np.full((len(nNeurons), nFiles), np.nan)
            decodingSummary_ADT[var]['ensemble_accuracy_trialType'] = {}
            for key in decodingResult[var]['prediction_accuracy'].keys():
                decodingSummary_ADT[var]['ensemble_accuracy_trialType'][key] = {}
                decodingSummary_ADT[var]['ensemble_accuracy_trialType'][key]['accuracy0-1'] = np.full(
                    (len(nNeurons), nFiles), np.nan)
                decodingSummary_ADT[var]['ensemble_accuracy_trialType'][key]['accuracy1-2'] = np.full(
                    (len(nNeurons), nFiles), np.nan)
                decodingSummary_ADT[var]['ensemble_accuracy_trialType'][key]['ctrl0-1'] = np.full(
                    (len(nNeurons), nFiles), np.nan)
                decodingSummary_ADT[var]['ensemble_accuracy_trialType'][key]['ctrl1-2'] = np.full(
                    (len(nNeurons), nFiles), np.nan)

    for var in decodingVars:
        # calculate fraction of neurons that are significant
        decodingSummary_ADT[var]['accuracy'] = np.concatenate((decodingSummary_ADT[var]['accuracy'],
                                                               decodingResult[var]['accuracy'][:, np.newaxis]), 1)
        decodingSummary_ADT[var]['ctrl_accuracy'] = np.concatenate((decodingSummary_ADT[var]['ctrl_accuracy'],
                                                                    decodingResult[var]['ctrl_accuracy'][:,
                                                                    np.newaxis]), 1)
        if nIdx < decodingResult[var]['decode_realSize']['accuracy'].shape[1]:
            subject_ADT_withn.append(group_adt.iloc[f]['subject'])
            decodingSummary_ADT[var]['accuracy_nNeurons'] = np.concatenate(
                (decodingSummary_ADT[var]['accuracy_nNeurons'],
                 decodingResult[var]['decode_realSize']['accuracy'][:, nIdx][:, np.newaxis]), 1)
            decodingSummary_ADT[var]['ctrl_nNeurons'] = np.concatenate((decodingSummary_ADT[var]['ctrl_nNeurons'],
                                                                        decodingResult[var]['decode_realSize'][
                                                                            'ctrl_accuracy'][:, nIdx][:, np.newaxis]),
                                                                       1)
            # get the average decoding accuracy between 1-2 second after cue

            timeMask1 = np.logical_and(regr_time >= 0, regr_time < 1)
            timeMask2 = np.logical_and(regr_time >= 1, regr_time < 2)
            ave_accuracy1 = np.nanmean(decodingResult[var]['decode_realSize']['accuracy'][timeMask1, :], 0)
            ave_accuracy2 = np.nanmean(decodingResult[var]['decode_realSize']['accuracy'][timeMask2, :],
                                       0)
            ave_ctrl1 = np.nanmean(decodingResult[var]['decode_realSize']['ctrl_accuracy'][timeMask1, :],
                                   0)
            ave_ctrl2 = np.nanmean(decodingResult[var]['decode_realSize']['ctrl_accuracy'][timeMask2, :],
                                   0)
            subject_ADT.append(group_adt.iloc[f]['subject'])
            decodingSummary_ADT[var]['ensemble_accuracy0-1'][:len(ave_accuracy1), f] = ave_accuracy1
            decodingSummary_ADT[var]['ensemble_accuracy1-2'][:len(ave_accuracy2), f] = ave_accuracy2
            decodingSummary_ADT[var]['ensemble_ctrl0-1'][:len(ave_accuracy1), f] = ave_ctrl1
            decodingSummary_ADT[var]['ensemble_ctrl1-2'][:len(ave_accuracy2), f] = ave_ctrl2

            for key in decodingResult[var]['prediction_accuracy'].keys():
                ave_accuracy1 = np.nanmean(
                    decodingResult[var]['decode_realSize']['prediction_accuracy'][key][timeMask1, :], 0)
                ave_accuracy2 = np.nanmean(
                    decodingResult[var]['decode_realSize']['prediction_accuracy'][key][timeMask2, :],
                    0)
                ave_ctrl1 = np.nanmean(
                    decodingResult[var]['decode_realSize']['prediction_accuracy_ctrl'][key][timeMask1, :],
                    0)
                ave_ctrl2 = np.nanmean(
                    decodingResult[var]['decode_realSize']['prediction_accuracy_ctrl'][key][timeMask2, :],
                    0)
                decodingSummary_ADT[var]['ensemble_accuracy_trialType'][key]['accuracy0-1'][:len(ave_accuracy1),
                f] = ave_accuracy1
                decodingSummary_ADT[var]['ensemble_accuracy_trialType'][key]['accuracy1-2'][:len(ave_accuracy1),
                f] = ave_accuracy2
                decodingSummary_ADT[var]['ensemble_accuracy_trialType'][key]['ctrl0-1'][:len(ave_accuracy1),
                f] = ave_ctrl1
                decodingSummary_ADT[var]['ensemble_accuracy_trialType'][key]['ctrl1-2'][:len(ave_accuracy1),
                f] = ave_ctrl2

        for key in decodingResult[var]['prediction_accuracy'].keys():
            decodingSummary_ADT[var]['prediction_accuracy'][key] = np.concatenate(
                (decodingSummary_ADT[var]['prediction_accuracy'][key],
                 decodingResult[var]['prediction_accuracy'][key][:, np.newaxis]), 1)
            decodingSummary_ADT[var]['prediction_accuracy_ctrl'][key] = np.concatenate(
                (decodingSummary_ADT[var]['prediction_accuracy_ctrl'][key],
                 decodingResult[var]['prediction_accuracy_ctrl'][key][:, np.newaxis]), 1)
