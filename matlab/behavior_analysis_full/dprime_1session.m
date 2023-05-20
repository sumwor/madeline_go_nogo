function [dprime, bycue, lick_fraction] = dprime_1session(numtrials, behavior, cues)

% initialize NaN arrays to count number of each cue and number of licks
% indices correspond to cue value (ie, lick_count(1) = licks to cue 1)
stimulus_count = NaN(1,16);
lick_count = NaN(1,16);
cues_in_session = unique(cues);

% if cues exist in session, set start value = 0
for cue = 1:length(cues_in_session)
    stimulus_count(cues_in_session(cue)) = 0;
    lick_count(cues_in_session(cue)) = 0;
end

% for each trial, identify the cue and increase the value in the
% corresponding index of stimulus_count. Then look at animal's behavior and
% assign licks to each cue index when licking occurs
% start at trial 2 because sometimes trial 1 doesn't play the tone
for trial = 2:numtrials
    stimulus_count(cues(trial)) = stimulus_count(cues(trial)) + 1;
    if cues(trial) == 8 || cues(trial) == 7 || cues(trial) == 6 || cues(trial) == 5 % nogo cues
        if behavior(trial) == 2.02 % nogo cue lick
            lick_count(cues(trial)) = lick_count(cues(trial)) + 1;
        end
    elseif cues(trial) < 5 % go cues
        if behavior(trial) < 2 % go cue lick
            lick_count(cues(trial)) = lick_count(cues(trial)) + 1;
        end
    end
end

lick_fraction = lick_count ./ stimulus_count; % [1,16] array with fraction of trials licked to, by cue
hits = nansum(lick_count(1:4))/nansum(stimulus_count(1:4)); % calculates fraction of correct licks
FA = nansum(lick_count(5:8))/nansum(stimulus_count(5:8)); % calculates fraction of incorrect licks
dprime = dprime_simple(hits, FA); % calculates dprime

% calculates dprime individually for different cue sets
bycue = [dprime_simple(lick_fraction(2), lick_fraction(7)) % training set
         dprime_simple(lick_fraction(1), lick_fraction(8)) % "easy" cue set 2
         dprime_simple(lick_fraction(3), lick_fraction(6)) % "hard" cue set 3
         dprime_simple(lick_fraction(4), lick_fraction(5))]; % "hardest" cue set 4

end
