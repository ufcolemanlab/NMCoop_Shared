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
pathname = '/Users/jcoleman/Documents/GitHub/NMCoop_Shared/#Febo/thy1gcamp6s_dataset1/';
filename = 'mThy6s2_alldrift_D4_001Z1_ROIdata.csv';
% filename = 'DATA_mThy6s2_alldrift_D5_001Z1hz1.csv'
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

n=1:32;
figure
C23d = bubblechart3(Ca2Coord(:,1),Ca2Coord(:,2),BasePks(1:32,1),BasePks(1:32,1),n);
title('Size of Spontaneous Baseline Events')
xlabel('X Position')
ylabel('Y Position')
zlabel('Event #')
xlim([0 500])
ylim([0 500])
zlim([0 max(BasePks(1:32,1))])
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