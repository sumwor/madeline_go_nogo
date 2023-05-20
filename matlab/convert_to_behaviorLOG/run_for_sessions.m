%% run all files
%function [out] = run_for_sessions()
    main_folder = '/Volumes/Wilbrecht_file_server/Madeline';
    behav_folder = dir(strcat(main_folder,'/raw_behavior/to_process'));
    for animal=4:length(behav_folder)
        animal_folder = behav_folder(animal).name;
        behav_files = dir(strcat(main_folder,'/raw_behavior/to_process/',animal_folder,'/*.mat'));
        for session=1:size(behav_files,1)
            animal = animal_folder;
            session = behav_files(session).name(8:13);
            gonogo_extract_behavior_data(main_folder, animal, session);
        end
    end     
%end

% %% run only certain animals/days
% function [out] = run_for_sessions()
%     folder = '/Volumes/Wilbrecht_file_server/Madeline';
%     animals = {'JUV010', 'JUV011'};
%     sessions = {{'211210', '211211'};
%     for i=1:length(animals)
%         for j=1:length(sessions{i})
%             animal = char(animals{i});
%             session = char(sessions{i}{j});
%             try
%                 exper_extract_behavior_data(folder, animal, session, 'bonsai');
%             catch
%                 disp([animal '_' session '_error']);
%             end
%         end
%     end     
% end
% 
% %% run from CSV file 
% % NOT FINISHED - THIS IS CURRENTLY COPIED FROM ALBERT'S CODE!
% function [out] = run_for_sessions_csv()
% %     folder = 'Z:\2ABT\ProbSwitch';
%     folder = '/Volumes/Wilbrecht_file_server/2ABT/ProbSwitch';
%     csvfile = fullfile(folder, 'probswitch_neural_subset_RRM.csv');
%     expr_tb = readtable(csvfile);
%     targ_etb = expr_tb(expr_tb.recorded==1, :);
%     for i=1:height(targ_etb)
%         ani_name = char(targ_etb.animal{i});
%         if startsWith(ani_name, 'RRM')
%             animal = char(targ_etb.animal{i});
%         else
%             animal = char(targ_etb.animal_ID{i});
%         end
%         age = char(targ_etb.age(i));
%         if mod(age, 1) == 0
%             session = sprintf('p%d', age);
%         else
%             digit = mod(age, 1);
%             if abs(digit-0.05) <= 1e-10
%                 session = sprintf('p%d_session0', floor(age));
%             else
%                 splt = split(string(digit), '.');
%                 session = sprintf('p%d_session%s', floor(age), char(splt(end)));
%             end
%         end
%         try
%             exper_extract_behavior_data(folder, animal, session, 'bonsai');
%         catch
%             disp([animal '_' session '_error']);
%         end
%     end     
% end