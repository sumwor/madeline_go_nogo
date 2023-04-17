% initialize variables for multiple animals' data
num_ADT = 20;
num_JUV = 14;
longest_ADT = 15;
longest_JUV = 17;
ADT_dprimes = [];
ADT_performance = [];
ADT_dp_cues = [];
JUV_dprimes = [];
JUV_performance = [];
JUV_dp_cues = [];

%% Run for each animal separately
selpath = uigetdir;
files = dir(strcat(selpath,'/*.mat'));

num_days = size(files,1);
dprime_by_session = [];
lickrate_by_session = [];
adj_trials_by_session = [];

for day = 1:size(files,1) 
    day_file = files(day).name; 
    load(strcat(selpath,'/',day_file));
    num_trials = exper.headfix_sound_gong.param.countedtrial.value;
    [adj_behav, adj_cues, adj_trials] = remove_disengaged_trials(day_file, exper.headfix_sound_gong.param.result.value(1:num_trials), exper.headfix_sound_gong.param.schedule.value(1:num_trials));
    adj_trials_by_session = [adj_trials_by_session adj_trials];
    [dprime, lickrate] = dprime_1session(adj_trials, adj_behav, adj_cues);
    dprime_by_session = [dprime_by_session; dprime];
    if lickrate(4) > 0
        lickrate_by_session = [lickrate_by_session; lickrate];
    end
end

if size(lickrate_by_session,1) > 2
    first3days_8cues = mean(lickrate_by_session(1:3,:),1);
else
    first3days_8cues = mean(lickrate_by_session,1);
end

if size(dprime_by_session,1) < longest_JUV % pad array with NaN to concatenate
    dprime_by_session = [dprime_by_session; NaN(longest_JUV-size(dprime_by_session,1),1)];
end
JUV_dprimes = [JUV_dprimes dprime_by_session];
JUV_performance = [JUV_performance; first3days_8cues];
%ADT_dprimes = [ADT_dprimes dprime_by_session];
%ADT_performance = [ADT_performance; first3days_8cues];

%% 
% add licking data to appropriate matrix for adults or adolescents
for animal = 1:size(ADT_performance,1)
    dp1 = dprime_simple(ADT_performance(animal,2), ADT_performance(animal,7));
    dp2 = dprime_simple(ADT_performance(animal,1), ADT_performance(animal,8));
    dp3 = dprime_simple(ADT_performance(animal,3), ADT_performance(animal,6));
    dp4 = dprime_simple(ADT_performance(animal,4), ADT_performance(animal,5));
    ADT_dp_cues = [ADT_dp_cues; dp1 dp2 dp3 dp4];
end
%%
for animal = 1:size(JUV_performance,1)
    dp1 = dprime_simple(JUV_performance(animal,2), JUV_performance(animal,7));
    dp2 = dprime_simple(JUV_performance(animal,1), JUV_performance(animal,8));
    dp3 = dprime_simple(JUV_performance(animal,3), JUV_performance(animal,6));
    dp4 = dprime_simple(JUV_performance(animal,4), JUV_performance(animal,5));
    JUV_dp_cues = [JUV_dp_cues; dp1 dp2 dp3 dp4];
end
%% Plot d'
figure;

for animal = 1:size(adults,2)
    plot(adults(:,animal),'-','color','#ff8f8f','linewidth',1);
    hold on
end

for animal = 1:size(JUV_dprimes,2)
    plot(adolescents(:,animal),'-','color','#8a95ff','linewidth',1);
    hold on
end

yline(1.8,'--','color','#666666','linewidth',1);
yline(0,'--','color','black','linewidth',2);
avg_adult = nanmean(adults,2);
sem_adult = nansem(adults,2);
%adt = plot(avg_adult(2:end),'-o','color','red','linewidth',3);
adt = errorbar(avg_adult(1:14),sem_adult(1:14),'-o','color','red','linewidth',3);
hold on
avg_adol = nanmean(adolescents,2);
sem_adol = nansem(adolescents,2);
%ado = plot(avg_adol(2:end),'-o','color','blue','linewidth',3);
ado = errorbar(avg_adol(1:14),sem_adol(1:14),'-o','color','blue','linewidth',3);
hold on

labeledlines = [adt ado];
legend(labeledlines,'Adults (n=20)','Adolescents (n=14)','fontsize',12,'location','northwest');
ylabel('d prime','Fontsize',14)
xlabel('Training Session','Fontsize',14)
xticks([1 4 7 10 14])
xticklabels({'1','4','7','10','14'})
title('D prime Over Sessions','Fontsize',16)

%% Plot easy vs. hard cues
figure;

subplot(1,2,1);
yline(1.8,'--','color','#666666','linewidth',1); hold on;
for animal = 1:size(JUV_dp_cues,1)
    plot(JUV_dp_cues(animal,:),'-o','color','#8a95ff','linewidth',1);
    hold on
end
plot(mean(JUV_dp_cues),'-o','color','blue','linewidth',2);
ylabel('d prime','Fontsize',12)
ylim([-0.5 3.5])
xticks([1 2 3 4])
xticklabels({'cues 2,7','cues 1,8','cues 3,6','cues 4,5'})
xlabel('Cue Set, from Easy to Hard','Fontsize',12)
title('Adolescent Dprime by Cue','Fontsize',14)

subplot(1,2,2);
yline(1.8,'--','color','#666666','linewidth',1); hold on;
for animal = 1:size(ADT_dp_cues,1)
    plot(ADT_dp_cues(animal,:),'-o','color','#ff8f8f','linewidth',1);
    hold on
end
plot(mean(ADT_dp_cues),'-o','color','red','linewidth',2);
ylabel('d prime','Fontsize',12)
ylim([-0.5 3.5])
xticks([1 2 3 4])
xticklabels({'cues 2,7','cues 1,8','cues 3,6','cues 4,5'})
xlabel('Cue Set, from Easy to Hard','Fontsize',12)
title('Adult Dprime by Cue','Fontsize',14)
