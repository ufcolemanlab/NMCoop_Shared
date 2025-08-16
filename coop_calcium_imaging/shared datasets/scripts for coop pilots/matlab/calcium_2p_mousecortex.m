% MATLAB R2023b
% 
% Script for importing raw calcium signal intensity data from a CSV table
% (imported from FIJI ROI-analysis step) and generating several figures for
% unit and network visualization and analysis of neuronal/cellular 'ensembles'.
% 
% Example inputs:
% pathname = '/Users/full/pathname/here/'
% filename = 'mThy6s2_alldrift_D4_001Z1_ROIdata.csv';

% Outputs: PNG files of plots and summaries, CSV and MAT files of results,
%          written to current directory
% 
% Author: Marcelo Febo <febo@ufl.edu>
% 2024-August Created function
% 2024-9-28: Annotate for general UF-MBI NeuroMicroscope Co-op use, mods for MATLAB R2021b (jcoleman@ufl.edu)
% 
% Copyright (c) 2024 Marcelo Febo <febo@ufl.edu>
% 
% MIT License
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

clear; 

%Enter directory path and filename to process
% pathname = '/Users/jcoleman/Documents/GitHub/NMCoop_Shared/coop_calcium_imaging/shared datasets/thy1gcamp6s_dataset1/';
pathname = '/Users/jcoleman/Documents/GitHub/NMCoop_Shared/coop_calcium_imaging/shared datasets/thy1gcamp6s_dataset2/';
% filename = 'mThy6s2_alldrift_D4_001Z1_ROIdata.csv';
filename = 'DATA_mThy6s2_alldrift_D5_001Z1hz1.csv';
csv_data = [pathname filename]

%Import signal using 'csv_data' 
calciumSig = readtable(csv_data);

% Identify data and coordinate columns to be used in analyses
mask = startsWith( calciumSig.Properties.VariableNames, 'RawIntDen');
Ca2Sigs = table2array(calciumSig(:,mask)); clear mask;

mask = startsWith( calciumSig.Properties.VariableNames, 'X');
Ca2Coord(1,:) = table2array(calciumSig(1,mask)); clear mask;

mask = startsWith( calciumSig.Properties.VariableNames, 'Y');
Ca2Coord(2,:) = table2array(calciumSig(1,mask)); clear mask;
Ca2Coord=Ca2Coord';

%Check neuron coordinates
% NOTE (jc 9/28/24) there is an issue with 'fontsize' function in Matlab R2021b
% nPos = scatter(Ca2Coord(:,1),Ca2Coord(:,2),"filled", 'FontSize', 10);
nPos = scatter(Ca2Coord(:,1),Ca2Coord(:,2),"filled");
set(gca,'FontSize',10)
    title('Neuron Position','FontSize',10);ylim([0 512]);xlim([0 512]); % image pixel dimensions
nPos.LineWidth = 0.6;
nPos.MarkerEdgeColor = 'b';
nPos.MarkerFaceColor = [0 0.5 0.5];
print(gcf,'NeuronPosition.png','-dpng','-r1200');

% Enter parameters here
params2p.freq = 1000;
params2p.fps = 30;
params2p.sec = 1/params2p.fps;
params2p.delay = 60;
params2p.frames = size(Ca2Sigs,1);
params2p.Orients = 8;
params2p.driftings = {'0','45','90','135','180','225','270','315'};
params2p.drftrate = 3;
params2p.dur = 5;
params2p.intsess= 7;
params2p.onsetdelay = 53;
params2p.frames = size(Ca2Sigs,1);
params2p.expdur = (params2p.dur+params2p.intsess)*8;
params2p.interval1 = (times(params2p.delay,params2p.fps)+1);
params2p.interval2 = (times(params2p.delay,params2p.fps)+1)+(params2p.fps*params2p.expdur);
params2p.interval3 = (times(params2p.delay,params2p.fps)+1)+(params2p.fps*params2p.expdur)*2;
params2p.interval4 = (times(params2p.delay,params2p.fps)+1)+(params2p.fps*params2p.expdur)*3;
params2p.interval5 = (times(params2p.delay,params2p.fps)+1)+((params2p.fps*params2p.expdur)*4);
params2p.baseT = params2p.sec:params2p.sec:params2p.delay;
params2p.stimT = params2p.sec:params2p.sec:(params2p.interval2-params2p.interval1)*params2p.sec;
params2p.totalT = params2p.sec:params2p.sec:params2p.frames*params2p.sec;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Processing of signals
Ca2SigsDespike = medfilt1(Ca2Sigs,30);
%Ca2SigsDt=detrend(Ca2SigsOutliers,2);
%Ca2SigsLP = lowpass(Ca2SigsDespike,15,params2p.fps);
%Ca2SigsOutliers = filloutliers(Ca2SigsDespike,"center","quartiles",2);
Ca2SigsScaled = normalize(Ca2SigsDespike,"range",[1 2]);

figure
Preplot = tiledlayout(2,2);
title(Preplot,'Preprocessed Signals')
xlabel(Preplot,'Image Frame (30 fps)')
ylabel(Preplot,'Normalized dF/F')
Preplot.Padding = 'compact';
Preplot.TileSpacing = 'compact';
nexttile
plot(Ca2Sigs(:,1)); title('Neuron 1');xlim([0 16200]);
nexttile
plot(Ca2Sigs(:,2)); title('Neuron 2');xlim([0 16200]);
nexttile
plot(Ca2Sigs(:,29)); title('Neuron 29');xlim([0 16200]);
nexttile
plot(Ca2Sigs(:,30)); title('Neuron 32');xlim([0 16200]);
print(gcf,'PreProcessing_Signals.png','-dpng','-r1200');

figure
Postplot = tiledlayout(2,2);
title(Postplot,'Postprocessed Signals')
xlabel(Postplot,'Image Frame (30 fps)')
ylabel(Postplot,'Normalized dF/F')
Postplot.Padding = 'compact';
Postplot.TileSpacing = 'compact';
nexttile
plot(Ca2SigsScaled(:,1)); title('Neuron 1');xlim([0 16200]);
nexttile
plot(Ca2SigsScaled(:,2)); title('Neuron 2');xlim([0 16200]);
nexttile
plot(Ca2SigsScaled(:,29)); title('Neuron 29');xlim([0 16200]);
nexttile
plot(Ca2SigsScaled(:,30)); title('Neuron 30');xlim([0 16200]);
print(gcf,'PostProcessing_Signals.png','-dpng','-r1200');



% Average trials to increase SNR and reduce data for each neuron
Ca2SigsBase = Ca2SigsScaled(1:(times(params2p.delay,params2p.fps)),:);
St1 = Ca2SigsScaled(params2p.interval1:(params2p.interval2-1),:);
St2 = Ca2SigsScaled(params2p.interval2:(params2p.interval3-1),:);
St3 = Ca2SigsScaled(params2p.interval3:(params2p.interval4-1),:);
St4 = Ca2SigsScaled(params2p.interval4:(params2p.interval5-1),:);
St5 = Ca2SigsScaled(params2p.interval5:params2p.frames,:);
StimProfiles = cat(3,St1,St2,St3,St4,St5);
StimProfile=mean(StimProfiles,3); clear St1 St2 St3 St4 St5; 
writematrix(StimProfile,'StimulusProfile.csv');
writematrix(Ca2SigsBase,'BaselineProfile.csv');
% Check results in plots - these are printed out

% figure
% TiledNeurons = tiledlayout(4,8);
% title(TiledNeurons,'Individual Neurons')
% xlabel(TiledNeurons,'Image Frame (30 fps)')
% ylabel(TiledNeurons,'Normalized dF/F')
% TiledNeurons.Padding = 'compact';
% TiledNeurons.TileSpacing = 'compact';
% nexttile
% plot(StimProfile(:,1));fontsize(6,"points");title('Neuron 1','FontSize',6);
% nexttile
% plot(StimProfile(:,2));fontsize(6,"points");title('Neuron 2','FontSize',6);
% nexttile
% plot(StimProfile(:,3));fontsize(6,"points");title('Neuron 3','FontSize',6);
% nexttile
% plot(StimProfile(:,4));fontsize(6,"points");title('Neuron 4','FontSize',6);
% nexttile
% plot(StimProfile(:,5));fontsize(6,"points");title('Neuron 5','FontSize',6);
% nexttile
% plot(StimProfile(:,6));fontsize(6,"points");title('Neuron 6','FontSize',6);
% nexttile
% plot(StimProfile(:,7));fontsize(6,"points");title('Neuron 7','FontSize',6);
% nexttile
% plot(StimProfile(:,8));fontsize(6,"points");title('Neuron 8','FontSize',6);
% nexttile
% plot(StimProfile(:,9));fontsize(6,"points");title('Neuron 9','FontSize',6);
% nexttile
% plot(StimProfile(:,10));fontsize(6,"points");title('Neuron 10','FontSize',6);
% nexttile
% plot(StimProfile(:,11));fontsize(6,"points");title('Neuron 11','FontSize',6);
% nexttile
% plot(StimProfile(:,12));fontsize(6,"points");title('Neuron 12','FontSize',6);
% nexttile
% plot(StimProfile(:,13));fontsize(6,"points");title('Neuron 13','FontSize',6);
% nexttile
% plot(StimProfile(:,14));fontsize(6,"points");title('Neuron 14','FontSize',6);
% nexttile
% plot(StimProfile(:,15));fontsize(6,"points");title('Neuron 15','FontSize',6);
% nexttile
% plot(StimProfile(:,16));fontsize(6,"points");title('Neuron 16','FontSize',6);
% nexttile
% plot(StimProfile(:,17));fontsize(6,"points");title('Neuron 17','FontSize',6);
% nexttile
% plot(StimProfile(:,18));fontsize(6,"points");title('Neuron 18','FontSize',6);
% nexttile
% plot(StimProfile(:,19));fontsize(6,"points");title('Neuron 19','FontSize',6);
% nexttile
% plot(StimProfile(:,20));fontsize(6,"points");title('Neuron 20','FontSize',6);
% nexttile
% plot(StimProfile(:,21));fontsize(6,"points");title('Neuron 21','FontSize',6);
% nexttile
% plot(StimProfile(:,22));fontsize(6,"points");title('Neuron 22','FontSize',6);
% nexttile
% plot(StimProfile(:,23));fontsize(6,"points");title('Neuron 23','FontSize',6);
% nexttile
% plot(StimProfile(:,24));fontsize(6,"points");title('Neuron 24','FontSize',6);
% nexttile
% plot(StimProfile(:,25));fontsize(6,"points");title('Neuron 25','FontSize',6);
% nexttile
% plot(StimProfile(:,26));fontsize(6,"points");title('Neuron 26','FontSize',6);
% nexttile
% plot(StimProfile(:,27));fontsize(6,"points");title('Neuron 27','FontSize',6);
% nexttile
% plot(StimProfile(:,28));fontsize(6,"points");title('Neuron 28','FontSize',6);
% nexttile
% plot(StimProfile(:,29));fontsize(6,"points");title('Neuron 29','FontSize',6);
% nexttile
% plot(StimProfile(:,30));fontsize(6,"points");title('Neuron 30','FontSize',6);
% nexttile
% plot(StimProfile(:,31));fontsize(6,"points");title('Neuron 31','FontSize',6);
% nexttile
% plot(StimProfile(:,32));fontsize(6,"points");title('Neuron 32','FontSize',6);
% print(gcf,'TiledNeurons.png','-dpng','-r1200');

% NOTE (jc 9/28/24) there is an issue with 'fontsize' function in Matlab R2021b -
% workaround (need to validate, but seems fine) from here \/\/\/\/:
figure
TiledNeurons = tiledlayout(4, 8);
title(TiledNeurons, 'Individual Neurons')
xlabel(TiledNeurons, 'Image Frame (30 fps)')
ylabel(TiledNeurons, 'Normalized dF/F')
TiledNeurons.Padding = 'compact';
TiledNeurons.TileSpacing = 'compact';

for i = 1:32
    nexttile
    plot(StimProfile(:, i));
    title(['Neuron ', num2str(i)], 'FontSize', 6);
    set(gca, 'FontSize', 6);
end

print(gcf, 'TiledNeurons.png', '-dpng', '-r1200');
% to here ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


figure
StimResponseProfile=mean(StimProfile(:,[3 5 13 14 19 22 23 25]),2);
ResponsiveNeurons = tiledlayout(1,1);
title(ResponsiveNeurons,'Mean Responsive Neurons')
xlabel(ResponsiveNeurons,'Image Frame (30 fps)')
ylabel(ResponsiveNeurons,'Normalized dF/F')
ResponsiveNeurons.Padding = 'compact';
ResponsiveNeurons.TileSpacing = 'compact';
nexttile
plot(StimResponseProfile);
print(gcf,'ResponsiveNeurons.png','-dpng','-r1200');

figure
StimResponseProfile2=mean(StimProfile(:,[20 27 28 29 30 31 32]),2);
ResponsiveNeurons3 = tiledlayout(1,1);
title(ResponsiveNeurons3,'Mean Responsive Neurons')
xlabel(ResponsiveNeurons3,'Image Frame (30 fps)')
ylabel(ResponsiveNeurons3,'Normalized dF/F')
ResponsiveNeurons3.Padding = 'compact';
ResponsiveNeurons3.TileSpacing = 'compact';
nexttile
plot(StimResponseProfile2);
print(gcf,'ResponsiveNeurons2.png','-dpng','-r1200');

%% Sponteanous transients
%Ca2SigsBase

% figure
% BaseNeurons = tiledlayout(4,8);
% title(BaseNeurons,'Individual Neurons')
% xlabel(BaseNeurons,'Image Frame (30 fps)')
% ylabel(BaseNeurons,'Normalized dF/F')
% BaseNeurons.Padding = 'compact';
% BaseNeurons.TileSpacing = 'compact';
% nexttile
% plot(Ca2SigsBase(:,1));fontsize(6,"points");title('Neuron 1','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,2));fontsize(6,"points");title('Neuron 2','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,3));fontsize(6,"points");title('Neuron 3','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,4));fontsize(6,"points");title('Neuron 4','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,5));fontsize(6,"points");title('Neuron 5','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,6));fontsize(6,"points");title('Neuron 6','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,7));fontsize(6,"points");title('Neuron 7','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,8));fontsize(6,"points");title('Neuron 8','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,9));fontsize(6,"points");title('Neuron 9','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,10));fontsize(6,"points");title('Neuron 10','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,11));fontsize(6,"points");title('Neuron 11','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,12));fontsize(6,"points");title('Neuron 12','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,13));fontsize(6,"points");title('Neuron 13','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,14));fontsize(6,"points");title('Neuron 14','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,15));fontsize(6,"points");title('Neuron 15','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,16));fontsize(6,"points");title('Neuron 16','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,17));fontsize(6,"points");title('Neuron 17','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,18));fontsize(6,"points");title('Neuron 18','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,19));fontsize(6,"points");title('Neuron 19','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,20));fontsize(6,"points");title('Neuron 20','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,21));fontsize(6,"points");title('Neuron 21','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,22));fontsize(6,"points");title('Neuron 22','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,23));fontsize(6,"points");title('Neuron 23','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,24));fontsize(6,"points");title('Neuron 24','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,25));fontsize(6,"points");title('Neuron 25','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,26));fontsize(6,"points");title('Neuron 26','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,27));fontsize(6,"points");title('Neuron 27','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,28));fontsize(6,"points");title('Neuron 28','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,29));fontsize(6,"points");title('Neuron 29','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,30));fontsize(6,"points");title('Neuron 30','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,31));fontsize(6,"points");title('Neuron 31','FontSize',6);xlim([0 1800]);
% nexttile
% plot(Ca2SigsBase(:,32));fontsize(6,"points");title('Neuron 32','FontSize',6);xlim([0 1800]);
% print(gcf,'BaselineNeuronActivity.png','-dpng','-r1200');
% Workaround 'fontsize' error in R2021b:
figure
BaseNeurons = tiledlayout(4, 8);
title(BaseNeurons, 'Individual Neurons')
xlabel(BaseNeurons, 'Image Frame (30 fps)')
ylabel(BaseNeurons, 'Normalized dF/F')
BaseNeurons.Padding = 'compact';
BaseNeurons.TileSpacing = 'compact';

for i = 1:32
    nexttile
    plot(Ca2SigsBase(:, i));
    title(['Neuron ', num2str(i)], 'FontSize', 6);
    set(gca, 'FontSize', 6);
    xlim([0 1800]);
end

print(gcf, 'BaselineNeuronActivity.png', '-dpng', '-r1200');

%Statistical assessments
BonfThr = 0.005/((size(StimProfile,2))^2);
[Rdata,Pdata, ~,~] = corrcoef(Ca2SigsBase,'Alpha',BonfThr);
Zdata = atanh(Rdata);
Zdata(isinf(Zdata))=0;
Pdata(Pdata>=BonfThr)=0; Pdata(Pdata>0)=1;
Zdata_thr_Pos=times(Zdata,Pdata);Zdata_thr_Pos(Zdata_thr_Pos<0)=0;
Zdata_thr_Neg=times(Zdata,Pdata);Zdata_thr_Neg(Zdata_thr_Neg>0)=0;

figure 
h1=heatmap(Zdata,Colormap=autumn); title('Pearson Correlations');xlim([1 32]);ylim([1 32]);
% fontsize(8,"points");
% Workaround 'fontsize' error in R2021b:
set(gca,'FontSize',8);
print(gcf,'Heatmaps_Correlations.png','-dpng','-r1200');
figure
h2=heatmap(Zdata_thr_Pos,Colormap=gray); title('Positive Correlations');xlim([1 32]);ylim([1 32]);
% fontsize(8,"points");
set(gca,'FontSize',8);
print(gcf,'Heatmaps_PosCorr.png','-dpng','-r1200');
figure
h3=heatmap(Zdata_thr_Neg,Colormap=gray); title('Negative Correlations');xlim([1 32]);ylim([1 32]);
% fontsize(8,"points");
set(gca,'FontSize',8);
print(gcf,'Heatmaps_NegCorr.png','-dpng','-r1200');


%Network plot
CIJ1=Zdata_thr_Pos ;
figure;
myColorMap = lines(length(CIJ1));
circularGraph(CIJ1,'Colormap',myColorMap);
print(gcf,'CircleNetworkPlotPos.png','-dpng','-r1200');

%Network plot
CIJ2=Zdata_thr_Neg ;
figure;
myColorMap = lines(length(CIJ2));
circularGraph(CIJ2,'Colormap',myColorMap);
print(gcf,'CircleNetworkPlotNeg.png','-dpng','-r1200');

BasePks(1:size(Ca2SigsBase,2),1)=0;
for i =1:size(Ca2SigsBase,2)
pd = fitdist(Ca2SigsBase(:,i),'Normal');
ci = paramci(pd,'Alpha',.05);  
%Ca2Median = median(Ca2SigsBase(:,1));
%Ca2Std = std(Ca2SigsBase(:,1));
%Ca2Thr = Ca2Median + (Ca2Std*10);
Temp=findpeaks(Ca2SigsBase(:,i),1./params2p.fps,'MinPeakHeight',ci(2,1)+(ci(2,2)*3)); 
BasePks(i,1)=numel(Temp); 
end
clear Temp;

numCells(1,1) = sum(BasePks == 0);
numCells(2,1) = sum(BasePks > 0 & BasePks <6);
numCells(3,1) = sum(BasePks >=6 );

BasePks = day1.BasePks;
Ca2Coord = day1.Ca2Coord;
Ca2SigsBase = day1.Ca2SigsBase

n=1:32;
figure
C23d = bubblechart3(Ca2Coord(:,1),Ca2Coord(:,2),BasePks(1:32,1),BasePks(1:32,1),n);
title([''],['Size of Spontaneous Baseline Events'])
xlabel('X Position')
ylabel('Y Position')
zlabel('Event #')
xlim([0 500])
ylim([0 500])
% zlim([0 max(BasePks(1:32,1))])
zlim([0 8]) % Opitmized for Sample dataset for *D4* and *D5* comparison
set(gca, 'FontSize', 16);
print(gcf,'BubbleChartEvents.png','-dpng','-r1200');

figure
BaseResponseProfile=mean(Ca2SigsBase(:,[3 5 13 14 19 22 23 25]),2);
ResponsiveNeurons2 = tiledlayout(1,1);
title(ResponsiveNeurons2,'Mean Responsive Neurons')
xlabel(ResponsiveNeurons2,'Image Frame (30 fps)')
ylabel(ResponsiveNeurons2,'Normalized dF/F')
ResponsiveNeurons2.Padding = 'compact';
ResponsiveNeurons2.TileSpacing = 'compact';
nexttile
plot(BaseResponseProfile);
print(gcf,'BaseResponsiveNeurons.png','-dpng','-r1200');

save('twoPcalcium_mouse.mat')

% End Original Script - Below this section is experimental code

%% Handling "D4" and "D5" data sets (acquired on day0 and day13)
% Load both datasets
path1 = '/Users/jcoleman/UFL Dropbox/Jason Coleman/--GRANTS--/2025/Febo - R01 tau-DMN-2p/calcium soma Thy1-GCAMP6f day1Vday2 Figures from Jason/D4(day1)/section2/twoPcalcium_mouse_day1.mat'
path2 = '/Users/jcoleman/UFL Dropbox/Jason Coleman/--GRANTS--/2025/Febo - R01 tau-DMN-2p/calcium soma Thy1-GCAMP6f day1Vday2 Figures from Jason/D5(day2)/section2/twoPcalcium_mouse_day2.mat'
% Load both datasets
day1 = load(path1);
day2 = load(path2);

% Extract calcium signals and fps
Ca2Sigs_day1 = day1.Ca2SigsBase;
Ca2Sigs_day2 = day2.Ca2SigsBase;
fps = day1.params2p.fps;

nROIs = size(Ca2Sigs_day1, 2);
BasePks_day1 = zeros(nROIs, 1);
BasePks_day2 = zeros(nROIs, 1);

% Count spontaneous peaks using thresholding
for i = 1:nROIs
%     pd = fitdist(Ca2Sigs_day1(:, i), 'Normal');
%     ci = paramci(pd, 'Alpha', 0.05);
    baseline = Ca2Sigs_day1(:, i);
    medF = median(baseline);
    madF = mad(baseline, 1);  % robust estimate of noise
    threshold = medF + 3 * madF;
%     % OR Fit a normal distribution to the calcium signal vector
%     pd = fitdist(Ca2Sigs_day1(:, i), 'Normal');
%     % Get 95% confidence interval on the mean and std
%     ci = paramci(pd);  % ci(1,:) = mu, ci(2,:) = sigma
%     % Define the threshold (mean + 3 * std upper bound)
%     threshold = ci(2,1) + ci(2,2) * 3; 
%     pks = findpeaks(Ca2Sigs_day1(:, i), 1/fps, ...
%         'MinPeakHeight', ci(2,1) + ci(2,2) * 3);
%     pks = findpeaks(baseline, 1/fps, 'MinPeakHeight', threshold);

    [pks,locs] = findpeaks(baseline, 'MinPeakHeight', threshold);

    BasePks_day1(i) = numel(pks);
    BasePksIndices_day1{i} = pks;
    BaseLocs_day1{i} = locs;

end
plot(baseline); hold on;
plot(BaseLocs_day1{i}, BasePksIndices_day1{i}, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6)

for i = 1:nROIs
%     pd = fitdist(Ca2Sigs_day2(:, i), 'Normal');
%     ci = paramci(pd, 'Alpha', 0.05);
    baseline = Ca2Sigs_day2(:, i);
    medF = median(baseline);
    madF = mad(baseline, 1);  % robust estimate of noise
    threshold = medF + 3 * madF;
%     pks = findpeaks(Ca2Sigs_day2(:, i), 1/fps, ...
%         'MinPeakHeight', ci(2,1) + ci(2,2) * 3);
%     pks = findpeaks(baseline, 1/fps, 'MinPeakHeight', threshold);

    [pks,locs] = findpeaks(baseline, 1/fps, 'MinPeakHeight', threshold);

    BasePks_day2(i) = numel(pks);
    BasePksIndices_day2{i} = pks;
    BaseLocs_day2{i} = locs;
end

% Classify ROIs into bins
categorize = @(x) [sum(x == 0); sum(x > 0 & x <= 6); sum(x > 6)];
num_day1 = categorize(BasePks_day1);
num_day2 = categorize(BasePks_day2);
barData = [num_day1, num_day2];

% Bar Plot with updated colors
figure('Position', [400 300 320 450]);
bh = bar(barData, 'FaceColor', 'flat');

% Custom colors
bh(1).CData = repmat([0.3 0.8 1], 3, 1);    % Day 1 = turquoise blue
bh(2).CData = repmat([1 0.2 0.8], 3, 1);    % Day 2 = magenta

xticklabels({'Silent (0)', 'Normal (0–6)', 'Hyperactive (>6)'});
xtickangle(45);
ylabel('# of Neurons');
title('Spontaneous Transients');

% Update legend to match new colors
legend({'Day 1', 'Day 2'}, 'Location', 'northoutside', 'Orientation', 'horizontal');

ylim([0 max(barData(:)) + 2]);
set(gca, 'Box', 'off', 'FontSize', 10);
set(gcf, 'Color', 'w');

% Optional: save output
% print(gcf, 'SpontaneousTransients_TurquoiseMagenta.png', '-dpng', '-r300');

%%
% --- Load data ---
day1 = load(path1);
day2 = load(path2);

Ca2_day1 = day1.Ca2SigsBase;
Ca2_day2 = day2.Ca2SigsBase;
fps = day1.params2p.fps;
nROIs = size(Ca2_day1, 2);

BasePks_day1 = zeros(nROIs, 1);
BasePks_day2 = zeros(nROIs, 1);

% --- Peak count using median + MAD threshold ---
for i = 1:nROIs
    baseline = Ca2_day1(:, i);
    medF = median(baseline);
    madF = mad(baseline, 1);
    threshold = medF + 3 * madF;
    pks = findpeaks(baseline, 1/fps, 'MinPeakHeight', threshold);
    BasePks_day1(i) = numel(pks);
end

for i = 1:nROIs
    baseline = Ca2_day2(:, i);
    medF = median(baseline);
    madF = mad(baseline, 1);
    threshold = medF + 3 * madF;
    pks = findpeaks(baseline, 1/fps, 'MinPeakHeight', threshold);
    BasePks_day2(i) = numel(pks);
end

% --- Classify ROIs ---
classify = @(x) (x == 0) + 2*(x > 0 & x <= 6) + 3*(x > 6);
% classify = @(x) [sum(x == 0); sum(x > 0 & x <= 6); sum(x > 6)];
classes_day1 = classify(BasePks_day1);  % returns 1, 2, or 3
classes_day2 = classify(BasePks_day2);

% --- Create binary classification matrix (3 x N) ---
% B1 = [classes_day1 == 1; classes_day1 == 2; classes_day1 == 3];  % 3xN
% B2 = [classes_day2 == 1; classes_day2 == 2; classes_day2 == 3];
B1 = [ ...
    (classes_day1 == 1)';
    (classes_day1 == 2)';
    (classes_day1 == 3)'];

B2 = [ ...
    (classes_day2 == 1)';
    (classes_day2 == 2)';
    (classes_day2 == 3)'];


% --- Compute mean ± SEM ---
mean_vals = [mean(B1, 2), mean(B2, 2)];
sem_vals = [std(B1, 0, 2)/sqrt(nROIs), std(B2, 0, 2)/sqrt(nROIs)];

disp('BasePks_day1 summary:')
disp(tabulate(BasePks_day1))

disp('Class breakdown - day 1:')
disp([ ...
    sum(classes_day1 == 1), ...
    sum(classes_day1 == 2), ...
    sum(classes_day1 == 3)])

disp('Class breakdown - day 2:')
disp([ ...
    sum(classes_day2 == 1), ...
    sum(classes_day2 == 2), ...
    sum(classes_day2 == 3)])

disp('Mean fraction per category:')
disp(mean_vals)


% --- Plot longitudinal line plot ---
figure; hold on
xvals = [1 2];
labels = {'Silent (0)', 'Normal (0–6)', 'Hyperactive (>6)'};
colors = [0.3 0.8 1; 1 0.2 0.8; 1 0.6 0.3];  % consistent color scheme

for c = 1:3
    y = mean_vals(c, :);
    yerr = sem_vals(c, :);
    
    % Plot line
    plot(xvals, y, '-', 'Color', colors(c, :), 'LineWidth', 2, ...
        'DisplayName', labels{c});
    
    % SEM shading
    fill([xvals, fliplr(xvals)], ...
         [y + yerr, fliplr(y - yerr)], ...
         colors(c, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end

xticks(xvals)
xticklabels({'Day 1', 'Day 2'})
ylabel('Fraction of Neurons')
title('Longitudinal Spontaneous Transient Categories')
legend('Location', 'northoutside', 'Orientation', 'horizontal')
ylim([0, 1])
set(gca, 'FontSize', 10)
set(gcf, 'Color', 'w')

% --- Optional: save figure ---
print(gcf, 'LongitudinalSpontaneousTransients_MADfix.png', '-dpng', '-r300')

%%
disp("Fraction of neurons per class (Day 1, Day 2):")
disp("Silent:"), disp(mean_vals(1, :))
disp("Normal:"), disp(mean_vals(2, :))
disp("Hyperactive:"), disp(mean_vals(3, :))

%%
% Replace 0s with small values for visualization
BasePks_day1_plot = BasePks_day1;
BasePks_day2_plot = BasePks_day2;
BasePks_day1_plot(BasePks_day1_plot == 0) = 0.001;
BasePks_day2_plot(BasePks_day2_plot == 0) = 0.001;

% Scatter plot: Day 1 vs Day 2
figure; hold on
scatter(BasePks_day1_plot, BasePks_day2_plot, ...
    60, 'o', 'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', [0.4 0.7 1], ...
    'LineWidth', 1.2);

plot([0.001, max([BasePks_day1; BasePks_day2])], ...
     [0.001, max([BasePks_day1; BasePks_day2])], ...
     'k--', 'LineWidth', 1);  % unity line

xlabel('Spontaneous Peaks (Day 1)')
ylabel('Spontaneous Peaks (Day 2)')
title('Neuron-by-Neuron Spontaneous Transients')
set(gca, 'FontSize', 10)
set(gcf, 'Color', 'w')
set(gca, 'XScale', 'log', 'YScale', 'log')  % optional: log scale
xlim([0.001, max(BasePks_day1_plot) + 1])
ylim([0.001, max(BasePks_day2_plot) + 1])
grid on
axis square

% Optional save
print(gcf, 'Scatter_SpontaneousTransients_Day1vsDay2.png', '-dpng', '-r300')
%%
% Replace zeros for plotting
BasePks_day1_plot = BasePks_day1;
BasePks_day2_plot = BasePks_day2;
BasePks_day1_plot(BasePks_day1_plot == 0) = 0.001;
BasePks_day2_plot(BasePks_day2_plot == 0) = 0.001;

% --- Define colors for categories ---
colors = [0.3 0.8 1; 1 0.2 0.8; 1 0.6 0.3];  % Silent, Normal, Hyperactive
class_labels = {'Silent', 'Normal', 'Hyperactive'};

% --- Create main scatter plot with marginal histograms ---
figure('Position', [300, 300, 600, 600]);
t = tiledlayout(4, 4, 'Padding', 'compact', 'TileSpacing', 'compact');

% Top histogram
nexttile(2, [1, 2]);
histogram(BasePks_day1, 'BinMethod', 'integers', 'FaceColor', [0.5 0.5 0.5]);
xlim([0, max(BasePks_day1)+1])
title('Day 1')
set(gca, 'XTick', [], 'YTick', [])

% Right histogram
nexttile(8, [2, 1]);
histogram(BasePks_day2, 'BinMethod', 'integers', 'Orientation', 'horizontal', ...
    'FaceColor', [0.5 0.5 0.5]);
ylim([0, max(BasePks_day2)+1])
set(gca, 'XTick', [], 'YTick', [])

% Main scatter plot
nexttile(6, [2, 2]); hold on

for c = 1:3
    idx = (classes_day1 == c);  % or classes_day2 depending on what you want
    scatter(BasePks_day1_plot(idx), BasePks_day2_plot(idx), ...
        60, 'filled', 'MarkerFaceColor', colors(c,:), ...
        'MarkerEdgeColor', 'k', 'DisplayName', class_labels{c});
end

% ROI index labels (optional)
show_labels = false;
if show_labels
    for i = 1:nROIs
        text(BasePks_day1_plot(i)+0.1, BasePks_day2_plot(i), ...
             num2str(i), 'FontSize', 7, 'Color', [0.2 0.2 0.2]);
    end
end

plot([0.001, max([BasePks_day1_plot; BasePks_day2_plot])], ...
     [0.001, max([BasePks_day1_plot; BasePks_day2_plot])], ...
     'k--', 'LineWidth', 1);  % unity line

xlabel('Spontaneous Peaks (Day 1)')
ylabel('Spontaneous Peaks (Day 2)')
title('Neuron-by-Neuron Spontaneous Transients')
set(gca, 'XScale', 'log', 'YScale', 'log')
xlim([0.001, max(BasePks_day1_plot) + 1])
ylim([0.001, max(BasePks_day2_plot) + 1])
axis square
grid on
legend('Location', 'southoutside', 'Orientation', 'horizontal')
set(gca, 'FontSize', 10)

% Save output
print(gcf, 'Scatter_Transients_wHistograms_andCategories.png', '-dpng', '-r300')

%%
% Figures for grant 2025 RO1

% Zdata = day1.Zdata;
% Zdata_thr_Pos = day1.Zdata_thr_Pos;
% Zdata_thr_Neg = day1.Zdata_thr_Neg;

Zdata = day2.Zdata;
Zdata_thr_Pos = day2.Zdata_thr_Pos;
Zdata_thr_Neg = day2.Zdata_thr_Neg;
%%
figure 
h1=heatmap(Zdata,Colormap=autumn); title('Pearson Correlations');xlim([1 32]);ylim([1 32]);
% fontsize(8,"points");
% Workaround 'fontsize' error in R2021b:
set(gca,'FontSize',8);
print(gcf,'Heatmaps_Correlations.png','-dpng','-r1200');
figure
h2=heatmap(Zdata_thr_Pos,Colormap=gray); title('Positive Correlations');xlim([1 32]);ylim([1 32]);
% fontsize(8,"points");
set(gca,'FontSize',8);
print(gcf,'Heatmaps_PosCorr.png','-dpng','-r1200');
figure
h3=heatmap(Zdata_thr_Neg,Colormap=gray); title('Negative Correlations');xlim([1 32]);ylim([1 32]);
% fontsize(8,"points");
set(gca,'FontSize',8);
print(gcf,'Heatmaps_NegCorr.png','-dpng','-r1200');

%%
clim = [-0.58 1.25]; % set this to whatever min/max you want

figure
imagesc(Zdata, clim);
colormap(autumn);
colorbar;
title('Pearson Correlations');
set(gca,'FontSize',8);

figure
imagesc(Zdata_thr_Pos, clim);
colormap(gray);
colorbar;
title('Positive Correlations');
set(gca,'FontSize',8);

figure
imagesc(Zdata_thr_Neg, clim);
colormap(gray);
colorbar;
title('Negative Correlations');
set(gca,'FontSize',8);
%%
% Assume: data_day1 and data_day13 are [nROIs x nFrames] matrices
% Use either raw deltaF/F or Z-scored traces (Z-score is good for normalization).

% Zdata = day1.Zdata;
% Zdata_thr_Pos = day1.Zdata_thr_Pos;
% Zdata_thr_Neg = day1.Zdata_thr_Neg;

% % Z-score each trace (optional but often useful)
z_day1 = zscore(data_day1, 0, 2);    % Z-score along time (each ROI)
z_day13 = zscore(data_day13, 0, 2);

% Compute cross-correlation matrix between Day 1 and Day 13
corr_mat = corr(z_day1', z_day13');  % [nROIs x nROIs]

figure
imagesc(corr_mat, [-.6 1.2]);           % Force color scale for comparison
colormap(jet); colorbar
xlabel('ROIs Day 13'); ylabel('ROIs Day 1');
title([''], ['ROI-wise Correlation: Day 1 vs Day 13']);
axis square
set(gca, 'FontSize', 16);

%%
% Z-score traces if you wish
% z_day1 = zscore(data_day1, 0, 2);
% z_day13 = zscore(data_day13, 0, 2);

nROIs = size(z_day1, 1);
corrs = zeros(nROIs, 1);

for i = 1:nROIs
    corrs(i) = corr(z_day1(i,:)', z_day13(i,:)');
end

figure
bar(corrs);
ylim([-1 1]);
xlabel('ROI'); ylabel('Correlation (Day 1 vs 13)');
title([''], ['Per-ROI Activity Correlation Across Days']);
set(gca,'FontSize',16);

%%
% Each frame: nROIs-dimensional vector. Correlate Day 1's vector at time t with Day 13's at same t.
minFrames = min(size(z_day1,2), size(z_day13,2));
corr_whole = diag(corr(z_day1(:,1:minFrames), z_day13(:,1:minFrames)));

figure
plot(corr_whole);
xlabel('Frame'); ylabel('Correlation');
title('Population Pattern Similarity Across Days');
set(gca,'FontSize',8);


