% FUNCTION NAME:
%   importLinescan()
%
% DESCRIPTION:
%   Import linescan data acquired from ScanImage v2021 (line-multiROI data)
%
% INPUT:
%   in1 - (double) Input one
%   in2 - (double) Input two
%   in3 - (double) Input three
%
% OUTPUT:
%   out - (double) Calculated average of the three inputs
%
% ASSUMPTIONS AND LIMITATIONS:
%   None
%
% REVISION HISTORY:
%   05/02/2018 - mmyers
%       * Initial implementation
%

function importLinescan()
% Data file locations for 1-18-23 Pizzi experiments
% This should open most linescan files/sessions (*.meta.txt, *.pmt.dat, etc)

% 02-04-23 - jcoleman@ufl.edu
% 1) Plots sample PMT data (recognizing data with 1- or 2-channel pmt data; 2
% or 3-d datasets, eg 3-d from multi-frame datasets (size,2 = 3))
%   - zero data
%   - Guassian smoothing
%
% 2) Heat plots - imagesc, etc
% 3) Need kymogr aph TIF outputs (<512 x 512 kymographs> x dy frames), scanline data map output overlay over reference
% gg image 


%file1 = '/Users/jcoleman/Documents/--LARGE DATA--/Pizzi data - 1-18-23 /tbi2_011723-2 roi1/T10_tbi2_0117-2_gg_4X_800nm_00002';
%file1_roi = '/Users/jcoleman/Documents/--LARGE DATA--/Pizzi data - 1-18-23 /tbi2_011723-2 roi1/tbi2_011723-2_2500Hz_roi_1000reps_roi1.roi';
%dir1='/Users/jcoleman/Documents/--LARGE DATA--/Pizzi data - 1-18-23 /tbi2_011723-2 roi2 /';
%**file1 = '/Users/jcoleman/Documents/--LARGE DATA--/Pizzi data - 1-18-23 /tbi2_011723-2 roi2flux/LS2500_tbi2_0117-2_gg_4X_800nm_roi2_00002.meta.txt';
%**file1 = '/Volumes/Samsung USB/abisambra lab/2p_2023/01-25-23 pizzi tbi2/LS1_16X6x_gg_800nm_m5_roi3_1_00002';
%file1 = '/Volumes/Samsung USB/abisambra lab/2p_2023/01-25-23 pizzi tbi2/LS1_16X6x_gg_800nm_m5_roi3_1_crossD_00001';
%file1_roi='/Users/jcoleman/Documents/--LARGE DATA--/Pizzi data - 1-18-23 /tbi2_011723-2 area2roi2flux/tbi2_011723-2_2500Hz_roi_1000reps_area2roi2_flux.roi';
%*&^&*^&^

[filename, pathname] = uigetfile('*.dat', 'Select Line Scan Data File');

filename1 = filename(1:end-8)
file1 = [pathname, filename1];
filename_dat1 = [file1 '.pmt.dat'];
filename_meta1 = [file1 '.meta.txt'];
filename_ref1 = [file1 '.ref.dat'];

% variables to help open variable file/session types - NEEDS WORK
extract_all = 0; %0 to only focus on roiNumber below; 1 parses 2 or 3 roi fields - 
                 % need to expand for auot-detect roi field numbers (length of roiGroup)
roiNumber = 1; % Which roi to extract from - number corresponds to roiGroup field number (each roi is a field)
% roi info
numRois_cells = length(roiGroup.activeRois)
roiGroup.activeRois(1, 1).scanfields.shortDescription
roiGroup.activeRois(1, 2).scanfields.shortDescription 
roiGroup.activeRois(1, 3).scanfields.shortDescription
roiGroup.activeRois(1, 4).scanfields.shortDescription 
%This could be a function to get data out of any roiGroup field (see FUNCTION)
gauss_factor = 10; % Gaussian smoothing factor


% Import data from ROI/linescan files (auto detects *.pmt.dat, *.meta.txt files)

% read ROI files
%filename_roi = file1_roi; %'/Users/jcoleman/Dropbox (UFL)/--LAbisambra--/jc-->pizzi share/figures - pizzi/40ms linescan example/sham1_4x_002deeper.roi';
%roigroup = scanimage.mroi.RoiGroup.loadFromFile(filename_roi);

% read line scan data files and zero the raw data (need root of filename only)
%fileName = '/Users/jcoleman/Desktop/test data1/test_linescan/galvo_1x_roi6_00001';
[header, pmtData, scannerPosData, roiGroup] = scanimage.util.readLineScanDataFiles(file1);

% squeeze is needed to remove the first dimension since it becomes singleton
if size(pmtData,2) < 3 %no LOOPS present (ie LOOPS refers to # of frames/acquisitions)

    ChanNum = size(pmtData,2);
    
    % process PMT1 data
    tempdataA = squeeze(pmtData(:,1)); %PMT1

    zeroed_data1 = zeros(length(tempdataA),size(tempdataA,2));

    for i = 1:1
        dataAmin = min(tempdataA(:,i));
        dataAtrans = tempdataA(:,i) + abs(dataAmin);
        zeroed_data1(:,i) = (dataAtrans/mean(dataAtrans)); %zeroed data from PMT1
    end

    % Conditional processing of PMT2 data (ie if present)
    if ChanNum == 2
        tempdataB = squeeze(pmtData(:,2)); %PMT2

        zeroed_data2 = zeros(length(tempdataB),size(tempdataB,2));

        for i = 1:1
            dataBmin = min(tempdataB(:,i));
            dataBtrans = tempdataB(:,i) + abs(dataBmin);
            zeroed_data2(:,i) = (dataBtrans/mean(dataBtrans)); %zeroed data from PMT1
        end
    end

end

% 1-20-2025 : ?SETUP FOR POINTSCAN - generalize etc
if size(pmtData,2) == 3 % ie if LOOPS/mulit-frames present    
    %ChanNum = size(pmtData,2)
    ChanNum = 2;
    frameNum = header.numFrames; 10; %NEED VAR FOR FRAME NUMBER
    
    tempdataA = squeeze(pmtData(:,1,:)); %PMT1 by LOOP/frame

    % zero the data and normalize - Find the min value, add it to all values, then divide all by the mean.
    zeroed_data1 = zeros(length(tempdataA),size(tempdataA,2));

    for i = 1:frameNum %NEED VAR FOR FRAME NUMBER
        dataAmin = min(tempdataA(:,i));
        dataAtrans = tempdataA(:,i) + abs(dataBmin);
        zeroed_data1(:,i) = (dataAtrans/mean(dataAtrans)); %zeroed data from PMT2
    end
       

    if ChanNum == 2
        tempdataB = squeeze(pmtData(:,2,:)); %PMT2 by frame

        zeroed_data2 = zeros(length(tempdataB),size(tempdataB,2));
    
            for i = 1:frameNum
                dataBmin = min(tempdataB(:,i));
                dataBtrans = tempdataB(:,i) + abs(dataBmin);
                zeroed_data2(:,i) = (dataBtrans/mean(dataBtrans)); %zeroed data from PMT2
            end
    end

end


%FUNCTION
% extractState moved to top
if extract_all == 0
    extractState = 0
    sprintf (['Processing ' num2str(roiNumber) ' ROI(s)'])
end
if extract_all == 1
    extractState = 1
    sprintf ('!!!NOTE!!! This case only holds for experiments where roi1=pause; roi2=line')
    sprintf ('Future implementation - variable smart-roi')
end
numberOfrois = size(roiGroup.activeRois,2);


line_duration_sec = roiGroup.rois(1, roiNumber).scanfields.duration;
pmtData_line_chunk = line_duration_sec*header.sampleRate;
acq_duration = header.acqDuration;
samplerate = header.sampleRate; %(pmtData)/(acq_duration);
%sanity check ... should = 0 for linescan chunk
sanity_check1 = line_duration_sec - (1/(samplerate/pmtData_line_chunk)); 
    if sanity_check1 == 0

        disp(" ... chunk size checks out... all done!")

    else

        disp(" ... chunk size does not check out - check sample rate!")
    end

if extract_all==1 & numberOfrois>1 
    % use to find rois with specific parameters (eg duration = 0.4msec, repetitions = 1000, etc)

    if numberOfrois == 2 % holds for roi1 = pause; roi2 = line
        % scan details and calculations (sample rate [Hz] and how to "chunk" the pmtData)
        extractState = sprintf ('!!!NOTE!!! This case only holds for experiments where roi1=pause; roi2=line')
        roi_LINE = 1;
        roi_PAUSE = 2;
        line_duration_sec = roiGroup.rois(1, roi_LINE).scanfields.duration;
        pause_duration_sec = roiGroup.rois(1, roi_PAUSE).scanfields.duration;
        pmtData_line_chunk = line_duration_sec*header.sampleRate;
        pmtData_pause_chunk = pause_duration_sec*header.sampleRate;
        
        acq_duration = header.acqDuration;
        
        samplerate = header.sampleRate; %(pmtData)/(acq_duration);
        %chunk_length = length(pmtData)/numberOfrois;
            %sanity check ... should = 0 for linescan chunk
            sanity_check1 = line_duration_sec - (1/(samplerate/pmtData_line_chunk));
        
            if sanity_check1 == 0
                disp(" ... chunk size checks out... all done!")
            else
                disp(" ... chunk size does not check out - check sample rate!")
            end
    
    end
    
    if numberOfrois == 3 % holds for roi1=line; roi2=pause; roi3=orthogonal line

        sprintf ('!!!NOTE!!! This case only holds for experiments where roi1=line; roi2=pause; roi3=orthogonal line')

        % scan details and calculations (sample rate [Hz] and how to "chunk" the pmtData)
        roi_LINE = 1;
        roi_PAUSE = 2;
        roi_ORTHO = 3;
        line_duration_sec = roiGroup.rois(1, roi_LINE).scanfields.duration;
        pause_duration_sec = roiGroup.rois(1, roi_PAUSE).scanfields.duration;
        ortho_duration_sec = roiGroup.rois(1, roi_ORTHO).scanfields.duration;
        pmtData_line_chunk = line_duration_sec*header.sampleRate;
        pmtData_pause_chunk = pause_duration_sec*header.sampleRate;
        pmtData_ortho_chunk = ortho_duration_sec*header.sampleRate;
        
        acq_duration = header.acqDuration;
        
        samplerate = header.sampleRate; %(pmtData)/(acq_duration);
        %chunk_length = length(pmtData)/numberOfrois;
            %sanity check ... should = 0 for linescan chunk
            sanity_check1 = line_duration_sec - (1/(samplerate/pmtData_line_chunk));
        
            if sanity_check1 == 0
                disp(" ... chunk size checks out... all done!")
            else
                disp(" ... chunk size does not check out - check sample rate!")
            end
    
    end

end
%END FUNCTION BUT NEED SOME PARAMETERS
% PLOTS 1 - requires data from ^^^^^^^^^^^^^^
% raw PMT data (1 or 2 channels using variables from above code)

temp_chunk = pmtData_line_chunk/1; %  = elements/frames per linescan (/100 = 0.001sec)
temp_chunk_sec = temp_chunk/samplerate;
temp_chunk_msec = temp_chunk/samplerate *1000;

y1 = smoothdata(zeroed_data1, 'gaussian', gauss_factor);

if ChanNum == 2
    y2 = smoothdata(zeroed_data2, 'gaussian', gauss_factor);
end

figure('Name',['PMT1/2 zeroed data - ' num2str(numberOfrois) 'X ROIs over ' num2str(acq_duration) 'sec']);
hold;
plot(y1);
if ChanNum == 2
    plot(y2+4);
end

figure('Name',['PMT1 zeroed data - ' num2str(temp_chunk_msec) 'ms sample chunk']);
plot(y1(1:temp_chunk)); 
hold;
plot(zeroed_data1(1:temp_chunk)); %  = elements/frames per line (/100 = 0.01sec)

if ChanNum == 2
    figure('Name',['PMT2 zeroed data - ' num2str(temp_chunk_msec) 'ms sample chunk']);
        plot(y2(1:temp_chunk));
        hold;
    plot(zeroed_data2(1:temp_chunk));
end


%% PLOTS 2 - linescan PMT data - - requires data from ^^^^^^^^^^^^^^
% raw PMT data (1 or 2 channels using variables from above code)

%reshape data into roi chunks
chunk_length = pmtData_line_chunk;
%numberOfrois = 1;

%data1_ind = 500000;
data1_ind = 50000;
    data1 = zeroed_data1(1:data1_ind,:);

if numberOfrois == 3 & extract_all == 1

    data1i_ind = 500000; %for 0.5msec * 1000 reps, 10msec pause - ??? need to generate???
        data1i = zeroed_data1(1:data1_ind,:);
    data1ii_ind = 512501; %for 0.5msec * 1000 reps, 10msec pause - ??? need to generate???
        data1ii = zeroed_data1(data1ii_ind:end,:);

    Ai = data1i; %PMT1
    Aii= data1ii; %PMT1
    Agauss_i = smoothdata(Ai, 'gaussian', gauss_factor); %PMT1
    Agauss_ii = smoothdata(Aii, 'gaussian', gauss_factor); %PMT1

    sample_MHz = samplerate/1000000;
    figure('Name',['PMT1 zeroed data i - ' num2str(line_duration_sec) 'sec per ROI @' num2str(sample_MHz) 'Ms/sec']);
    hold;

    for i = 1:size(Agauss_i,2)
        plot(Agauss_i(:,i)+i);
    end

    figure('Name',['PMT1 zeroed data ii - ' num2str(line_duration_sec) 'sec per ROI @' num2str(sample_MHz) 'Ms/sec']);
    hold;

    for i = 1:size(Agauss_ii,2)
        plot(Agauss_ii(:,i)+i);
    end

else

    %A = reshape(zeroed_data1(:,1),[chunk_length, numberOfrois]); %PMT1
    %Agauss = smoothdata(A, 'gaussian', gauss_factor); %PMT1
    A = data1; %PMT1
    Agauss = smoothdata(A, 'gaussian', gauss_factor); %PMT1
    
    if ChanNum == 2 %(*&(*^)(*&^()*^&*&^*&^*&^*&^*&^*&^%&*(^*&^^*&^R%^*E$^%CRTCYUVT
        data2 = zeroed_data2(12501:62500,:); %(&*)*&*Y
        B = data2; %PMT2
        Bgauss = smoothdata(B, 'gaussian', gauss_factor); %PMT2
    end
    
    % plots2
    sample_MHz = samplerate/1e6;
    figure('Name',['PMT1 zeroed data - ' num2str(line_duration_sec) 'sec per ROI @' num2str(sample_MHz) 'Ms/sec']);
    hold;
    
    for i = 1:size(Agauss,2)
        plot(Agauss(:,i)+i);
    end
    
    if ChanNum == 2
        figure('Name',['PMT2 zeroed data - ' num2str(line_duration_sec) 'sec per ROI @' num2str(sample_MHz) 'Ms/sec']);
        hold;
    
        for i = 1:size(Bgauss,2)
            plot(Bgauss(:,i)+i);
        end
    end

end

if dataAll == 1

        filename_data = [file1 '.mat'];
        filename_plots = [file1 '_plots.eps'];
        
        % save all variables in the workspace to a MAT file
        save(filename_data);
        % save current fig(s) to *.ps file (open as PDF or EPS)
        print(gcf, '-dpsc', ...
                   '-append', ...
                   [filenames.csvroot '_fwhm.ps']);
        % Loop through all open window/figures and save to a single *.PS file
        FolderName = tempdir;   % Your destination folder
        FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
        for iFig = 1:length(FigList)
          FigHandle = FigList(iFig);
          FigName   = num2str(get(FigHandle, 'Number'));
          set(0, 'CurrentFigure', FigHandle);
          savefig(fullfile(FolderName, [FigName '.fig']));
        end
        
        %plot(PeakSig); %hold on; % not a LOOP HERE _ HOW TO SAVE ALL FIGS?
        %findpeaks(PeakSig,x,'MinPeakProminence',peak_threshold,'Annotate','extents')
            %saveas(gcf,[filenames.csvroot '_fwhm.eps'],'eps2c');
            %print(gcf, '-dpsc', '-append', [filenames.csvroot '_fwhm.ps']) %will append PDF file
        
        close all;

end

end