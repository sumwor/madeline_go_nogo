function [dprime, lick_fraction] = dprime_1session(numtrials, behavior, cues)

stimulus_count = zeros(1,16);
lick_count = zeros(1,16);

for trial = 2:numtrials
    stimulus_count(cues(trial)) = stimulus_count(cues(trial)) + 1;
    if cues(trial) == 8 || cues(trial) == 7 || cues(trial) == 6 || cues(trial) == 5
        if behavior(trial) == 2.02 % nogo cue lick
            lick_count(cues(trial)) = lick_count(cues(trial)) + 1;
        end
    elseif cues(trial) < 5
        if behavior(trial) < 2
            lick_count(cues(trial)) = lick_count(cues(trial)) + 1;
        end
    end
end

lick_fraction = lick_count ./ stimulus_count;
hits = sum(lick_count(1:4))/sum(stimulus_count(1:4));
FA = sum(lick_count(5:8))/sum(stimulus_count(5:8));
dprime = dprime_simple(hits, FA);

end
