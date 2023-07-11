function [binned] = dprime_binned(numtrials, behavior, cues, num_bins)
% this function is identical to dprime_1session except we calculate dprime
% as fractions of a session. The number of fractions is specified by
% num_bins

stimulus_count = NaN(1,16);
lick_count = NaN(1,16);
cues_in_session = unique(cues);
bin_size = floor(numtrials/num_bins);
binned = [];

for cue = 1:length(cues_in_session)
    stimulus_count(cues_in_session(cue)) = 0;
    lick_count(cues_in_session(cue)) = 0;
end

for chunk = 2:bin_size:numtrials-(bin_size-2)
    for trial = chunk:chunk+bin_size-2
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
    lick_fraction = lick_count ./ stimulus_count; 
    hits = nansum(lick_count(1:4))/nansum(stimulus_count(1:4)); 
    FA = nansum(lick_count(5:8))/nansum(stimulus_count(5:8));
    binned = [binned dprime_simple(hits, FA)];
end

end

