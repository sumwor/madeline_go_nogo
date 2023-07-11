function [level] = cumulative_cues(cues)
% this function labels behavior based on the presence of different cues

if ismember(5,cues)==1
    level = 4;
elseif ismember(6,cues)==1
    level = 3;
elseif ismember(8,cues)==1
    level = 2;
elseif ismember(7,cues)==1
    level = 1;
else
    level = 0;
end


end