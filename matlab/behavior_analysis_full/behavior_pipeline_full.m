%% Analyze Behavior
root_path = 'Z:\HongliWang\Madeline\LateLearning';

%% making folders
data_path = fullfile(root_path, 'Data'); % use fullfile to accomodate for different file separators in windows/osx
analysis_path = fullfile(root_path,'Analysis')
summary_path = fullfile(root_path,'Summary')

if ~exist(analysis_path)
    mkdir(analysis_path)
end
if ~exist(summary_path)
    mkdir(summary_path)
end

animals_to_process = dir(data_path); 
% Once you select your folder it appears as a
% variable in the workspace. You will see that there are 3 extra hidden 
% files in each folder that have to do with computer stuff. As a result,
% any time you reference the folder number, you need to subtract 3. That's
% why we start the for loop below at 4

%% maybe add a user input for output folders as well?
output_folder = analysis_path; 
% save the result of preprocessed data (LOG.mat file) into the same folder
if ~exist(output_folder)
    mkdir(output_folder)
end

%% iterate through every animal 
for animal = 1:length(animals_to_process) % for each animal
    if ~strcmp(animals_to_process(animal).name, '.') & ~strcmp(animals_to_process(animal).name, '..') & ~strcmp(animals_to_process(animal).name, '.DS_Store')
        animal_folder = animals_to_process(animal).name; 
        animal_output = fullfile(output_folder, animal_folder);

        if ~exist(animal_output)
            mkdir(animal_output)
        end
        current_animal = fullfile(data_path,animal_folder);
        [performance, lick_behavior, info, out] = get_behavior(current_animal,animal_output);
        behavior_struct.(animal_folder).dprimes = performance;
        behavior_struct.(animal_folder).licks = lick_behavior;
        behavior_struct.(animal_folder).info = info;

    end
end
