%% Analyze Behavior
selpath = uigetdir;
animals_to_process = (dir(selpath)); % Once you select your folder it appears as a
% variable in the workspace. You will see that there are 3 extra hidden 
% files in each folder that have to do with computer stuff. As a result,
% any time you reference the folder number, you need to subtract 3. That's
% why we start the for loop below at 4

for animal = 4:length(animals_to_process) % for each animal
    animal_folder = animals_to_process(animal).name; 
    current_animal = fullfile(selpath,animal_folder);
    [performance, lick_behavior, info] = get_behavior(current_animal);
    behavior_struct.(animal_folder).dprimes = performance;
    behavior_struct.(animal_folder).licks = lick_behavior;
    behavior_struct.(animal_folder).info = info;
end
