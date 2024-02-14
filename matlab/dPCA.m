% try elife dPCA here
% combine all sessions together

filepath = 'Z:\HongliWang\Madeline\LastSession2Cues\Analysis\ADT008\220415\fluo\MLR\trialbytrialVar.mat';

data = load(filepath);

stimulus = data.X(2,:,1);
decision = data.X(5,:,1);

numCells = size(data.y,1);

numGo = sum(stimulus); numNoGo = length(stimulus)-sum(stimulus);
numLick = sum(decision); numNoLick = length(decision) - sum(numLick);

maxTrialNum = max([sum((stimulus==0)&(decision==0)), ...
                   sum((stimulus==0)&(decision==1)), ...
                   sum((stimulus==1)&(decision==0)), ...
                   sum((stimulus==1)&(decision==1))]);

trialNum = zeros(numCells, 2,2);
for s=1:2
    for d= 1:2
        trialNum(:,s,d) = ones(numCells,1)*sum((stimulus==s-1)&(decision==d-1));
    end
end
% initialize the df/f matrix
activity = NaN(numCells, 2, 2, length(data.regr_time), maxTrialNum);

trialIdx = 1:length(stimulus);
%
% s: 1/nogo; 2/go
% d: 1/nogo; 2/go

for s=1:2
    for d = 1:2
        trialMask = trialIdx((stimulus==s-1)&(decision==d-1));
        nTrials = length(trialMask);
        activity(:,s,d,:,1:nTrials) = permute(data.y(:,trialMask,:),[1,3,2]);
    end
end

activityAverage = nanmean(activity, 5);

%% Define parameter grouping

% *** Don't change this if you don't know what you are doing! ***
% activity array has [N S D T E] size; herewe ignore the 1st dimension 
% (neurons), i.e. we have the following parameters:
%    1 - stimulus 
%    2 - decision
%    3 - time
% There are three pairwise interactions:
%    [1 3] - stimulus/time interaction
%    [2 3] - decision/time interaction
%    [1 2] - stimulus/decision interaction
% And one three-way interaction:
%    [1 2 3] - rest
% As explained in the eLife paper, we group stimulus with stimulus/time interaction etc.:

combinedParams = {{1, [1 3]}, {2, [2 3]}, {3}, {[1 2], [1 2 3]}};
margNames = {'Stimulus', 'Decision', 'Condition-independent', 'S/D Interaction'};
margColours = [23 100 171; 187 20 25; 150 150 150; 114 97 171]/256;

% For two parameters (e.g. stimulus and time, but no decision), we would have
% activity array of [N S T E] size (one dimension less, and only the following
% possible marginalizations:
%    1 - stimulus
%    2 - time
%    [1 2] - stimulus/time interaction
% They could be grouped as follows: 
%    combinedParams = {{1, [1 2]}, {2}};

% Time events of interest (e.g. stimulus onset/offset, cues etc.)
time = data.regr_time;
% They are marked on the plots with vertical lines
timeEvents = time(round(length(time)/2));

% % check consistency between trialNum and activity
% for n = 1:size(activity,1)
%     for s = 1:size(activity,2)
%         for d = 1:size(activity,3)
%             assert(isempty(find(isnan(activity(n,s,d,:,1:trialNum(n,s,d))), 1)), 'Something is wrong!')
%         end
%     end
% end

%% Step 1: PCA of the dataset

X = activityAverage(:,:);
X = bsxfun(@minus, X, mean(X,2));

[W,~,~] = svd(X, 'econ');
W = W(:,1:20);

% minimal plotting
dpca_plot(activityAverage, W, W, @dpca_plot_default);

% computing explained variance
explVar = dpca_explainedVariance(activityAverage, W, W, ...
    'combinedParams', combinedParams);

% a bit more informative plotting
dpca_plot(activityAverage, W, W, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours);

%% Step 2: PCA in each marginalization separately

dpca_perMarginalization(activityAverage, @dpca_plot_default, ...
   'combinedParams', combinedParams);
%% Step 3: dPCA without regularization and ignoring noise covariance

% This is the core function.
% W is the decoder, V is the encoder (ordered by explained variance),
% whichMarg is an array that tells you which component comes from which
% marginalization

tic
[W,V,whichMarg] = dpca(activityAverage, 20, ...
    'combinedParams', combinedParams);
toc

explVar = dpca_explainedVariance(activityAverage, W, V, ...
    'combinedParams', combinedParams);

dpca_plot(activityAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3, ...
    'legendSubplot', 16);

%% Step 4: dPCA with regularization

% This function takes some minutes to run. It will save the computations 
% in a .mat file with a given name. Once computed, you can simply load 
% lambdas out of this file:
%   load('tmp_optimalLambdas.mat', 'optimalLambda')

% Please note that this now includes noise covariance matrix Cnoise which
% tends to provide substantial regularization by itself (even with lambda set
% to zero).

optimalLambda = dpca_optimizeLambda(activityAverage, activity, trialNum, ...
    'combinedParams', combinedParams, ...
    'simultaneous', ifSimultaneousRecording, ...
    'numRep', 2, ...  % increase this number to ~10 for better accuracy
    'filename', 'tmp_optimalLambdas.mat');

Cnoise = dpca_getNoiseCovariance(activityAverage, ...
    activity, trialNum, 'simultaneous', ifSimultaneousRecording);

[W,V,whichMarg] = dpca(activityAverage, 20, ...
    'combinedParams', combinedParams, ...
    'lambda', optimalLambda, ...
    'Cnoise', Cnoise);

explVar = dpca_explainedVariance(activityAverage, W, V, ...
    'combinedParams', combinedParams);

dpca_plot(activityAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3,           ...
    'legendSubplot', 16);

%% Optional: estimating "signal variance"

explVar = dpca_explainedVariance(activityAverage, W, V, ...
    'combinedParams', combinedParams, ...
    'Cnoise', Cnoise, 'numOfTrials', trialNum);

% Note how the pie chart changes relative to the previous figure.
% That is because it is displaying percentages of (estimated) signal PSTH
% variances, not total PSTH variances. See paper for more details.

dpca_plot(activityAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3,           ...
    'legendSubplot', 16);

%% Optional: decoding
S=2;
decodingClasses = {[(1:S)' (1:S)'], repmat([1:2], [S 1]), [], [(1:S)' (S+(1:S))']};

accuracy = dpca_classificationAccuracy(activityAverage, activity, trialNum, ...
    'lambda', optimalLambda, ...
    'combinedParams', combinedParams, ...
    'decodingClasses', decodingClasses, ...
    'simultaneous', ifSimultaneousRecording, ...
    'numRep', 5, ...        % increase to 100
    'filename', 'tmp_classification_accuracy.mat');

dpca_classificationPlot(accuracy, [], [], [], decodingClasses)

accuracyShuffle = dpca_classificationShuffled(activity, trialNum, ...
    'lambda', optimalLambda, ...
    'combinedParams', combinedParams, ...
    'decodingClasses', decodingClasses, ...
    'simultaneous', ifSimultaneousRecording, ...
    'numRep', 100, ...        % increase to 100
    'numShuffles',100 , ...  % increase to 100 (takes a lot of time)
    'filename', 'tmp_classification_accuracy.mat');

dpca_classificationPlot(accuracy, [], accuracyShuffle, [], decodingClasses)

componentsSignif = dpca_signifComponents(accuracy, accuracyShuffle, whichMarg);

dpca_plot(activityAverage, W, V, @dpca_plot_default, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3,           ...
    'legendSubplot', 16,                ...
    'componentsSignif', componentsSignif);
