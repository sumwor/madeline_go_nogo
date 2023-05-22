function [out] = gonogo_extract_behavior(input, output, cutoff)
    % (folder, animal, session) -> filename of the behavior_raw  fullfile
    %filename = fullfile(folder, 'raw_behavior/to_process', animal, session);
    if ~exist(output, 'dir')
        mkdir(output);
    end
    behav_log = fullfile(output, [input(44:57), 'behaviorLOG.mat']);
    %if ~exist(behav_log, 'file')
        out = get_Headfix_GoNo_EventTimes(input, output);
        out.cutoff = cutoff
        save(behav_log, '-v7.3', 'out');
    %end
end