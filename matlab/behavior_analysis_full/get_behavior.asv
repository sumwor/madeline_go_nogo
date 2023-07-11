function [dprimes, lickrate, session_info, out] = get_behavior(animal_folder, output)


session_files = dir(fullfile(animal_folder,'*.mat'));

for day = 1:size(session_files,1) 

    day_file = session_files(day).name; 
    session = ['session' num2str(day)];
    exper = load(fullfile(animal_folder,day_file));
    num_trials_old = exper.headfix_sound_gong.param.countedtrial.value;

    [adj_behav, adj_cues, adj_trials] = remove_disengaged_trials(day_file, ...
        exper.headfix_sound_gong.param.result.value(1:num_trials_old), ...
        exper.headfix_sound_gong.param.schedule.value(1:num_trials_old));
    session_info.num_adj_trials(day) = adj_trials;
    session_info.behavior_level(day) = cumulative_cues(adj_cues);
    
    %% old pipeline to generate behavior LOG.mat for python process

    input = fullfile(animal_folder,day_file);
    cutoff =adj_trials;
    [out] = gonogo_extract_behavior(input, output, cutoff);


    %% behavior analysis
    [dprime, bycue, licking] = dprime_1session(adj_trials, adj_behav, adj_cues);
    %binned_dprimes = dprime_binned(adj_trials, adj_behav, adj_cues, num_bins);
    dprimes.overall_by_session(day) = dprime;
    dprimes.cueset1_by_session(day) = bycue(1);
    dprimes.cueset2_by_session(day) = bycue(2);
    dprimes.cueset3_by_session(day) = bycue(3);
    dprimes.cueset4_by_session(day) = bycue(4);
    %dprimes.binned.(session) = binned_dprimes;
    lickrate.overall_by_session.(session) = licking;
    lickrate.go_easy(day) = nanmean(licking(1:2)); 
    lickrate.go_hard(day) = nanmean(licking(3:4));
    lickrate.nogo_hard(day) = nanmean(licking(5:6));
    lickrate.nogo_easy(day) = nanmean(licking(7:8));

    beh.dprimes = dprimes(day);
    beh.licks = lickrate(day);
    beh.info = session_info;
    beh.out = out;

        % save the struct
    behav_log = fullfile(animal_output, [input(44:57), 'behaviorLOG.mat']);

    save(behav_log, '-v7.3', 'out');

end 

end

