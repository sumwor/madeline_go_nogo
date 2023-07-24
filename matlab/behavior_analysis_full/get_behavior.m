function [dprimes, lickrate, session_info, out] = get_behavior(animal_folder, output)


sessions= dir(animal_folder);

for day = 1:size(sessions,1)
    if ~strcmp(sessions(day).name, '.') & ~strcmp(sessions(day).name, '..')
        session_files = dir(fullfile(sessions(day).folder, sessions(day).name,'*.mat'));
        day_file = session_files.name;
        session = ['session' num2str(day-2)];
        exper = load(fullfile(session_files.folder,day_file));
        num_trials_old = exper.exper.headfix_sound_gong.param.countedtrial.value;

        [adj_behav, adj_cues, adj_trials] = remove_disengaged_trials(day_file, ...
            exper.exper.headfix_sound_gong.param.result.value(1:num_trials_old), ...
            exper.exper.headfix_sound_gong.param.schedule.value(1:num_trials_old));
        session_info.num_adj_trials(day-2) = adj_trials;
        session_info.behavior_level(day-2) = cumulative_cues(adj_cues);

        %% old pipeline to generate behavior LOG.mat for python process

        input_file = fullfile(session_files.folder,day_file);
        output_path = fullfile(output,sessions(day).name);
        if ~exist(output_path)
            mkdir(output_path)
        end
        cutoff =adj_trials;
        [out] = gonogo_extract_behavior(input_file, output_path, cutoff);


        %% behavior analysis
        [dprime, bycue, licking] = dprime_1session(adj_trials, adj_behav, adj_cues);
        %binned_dprimes = dprime_binned(adj_trials, adj_behav, adj_cues, num_bins);
        dprimes.overall_by_session(day-2) = dprime;
        dprimes.cueset1_by_session(day-2) = bycue(1);
        dprimes.cueset2_by_session(day-2) = bycue(2);
        dprimes.cueset3_by_session(day-2) = bycue(3);
        dprimes.cueset4_by_session(day-2) = bycue(4);
        %dprimes.binned.(session) = binned_dprimes;
        lickrate.overall_by_session.(session) = licking;
        lickrate.go_easy(day-2) = nanmean(licking(1:2));
        lickrate.go_hard(day-2) = nanmean(licking(3:4));
        lickrate.nogo_hard(day-2) = nanmean(licking(5:6));
        lickrate.nogo_easy(day-2) = nanmean(licking(7:8));
        
        beh.dprimes = struct;
        fields = fieldnames(dprimes);
        for i = 1:numel(fields)
            field = fields{i};
            originalArray = dprimes.(field);
            beh.dprimes.(field) = originalArray(day-2);
        end
        
        beh.licks = struct;
        fields = fieldnames(lickrate);
        for i = 2:numel(fields)
            field = fields{i};
            originalArray = lickrate.(field);
            beh.licks.(field) = originalArray(day-2);
        end

        beh.info = struct;
        fields = fieldnames(session_info);
        for i = 1:numel(fields)
            field = fields{i};
            originalArray = session_info.(field);
            beh.licks.(field) = originalArray(day-2);
        end

        beh.out = out;

        % save the struct
        behav_log = fullfile(output_path, [input_file(end-23:end-11), '-behaviorLOG.mat']);

        save(behav_log, '-v7.3', 'beh');

    end

end

