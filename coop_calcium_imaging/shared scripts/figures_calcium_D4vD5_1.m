% | Panel | Purpose                                            | Input                                                 |
% | ----- | -------------------------------------------------- | ----------------------------------------------------- |
% | **A** | Registered FOV + ROI overlays                      | (optional `ROI mask` or centroid coords if available) |
% | **B** | Example ΔF/F traces for same neurons               | `Ca2SigsScaled`                                       |
% | **C** | Responsiveness Day 1 vs Day 2 scatter              | `Ca2SigsScaled`                                       |
% | **D** | Cluster motif assignments (e.g., pie chart or bar) | cluster labels from ΔF/F                              |
% | **E** | Silent/Normal/Hyperactive counts                   | `Ca2SigsBase` via `findpeaks`                         |
% 


% Load both datasets
path1 = '/Users/jcoleman/UFL Dropbox/Jason Coleman/--GRANTS--/2025/Febo - R01 tau-DMN-2p/calcium soma Thy1-GCAMP6f day1Vday2 Figures from Jason/D4(day1)/section2/twoPcalcium_mouse_day1.mat'
path2 = '/Users/jcoleman/UFL Dropbox/Jason Coleman/--GRANTS--/2025/Febo - R01 tau-DMN-2p/calcium soma Thy1-GCAMP6f day1Vday2 Figures from Jason/D5(day2)/section2/twoPcalcium_mouse_day2.mat'
% Load both datasets
day1 = load(path1);
day2 = load(path2);

F_day1 = day1.Ca2SigsScaled;  % ΔF/F traces during base and stim
F_day2 = day2.Ca2SigsScaled;
Fbase_day1 = day1.Ca2SigsBase;  % ΔF/F traces during base
Fbase_day2 = day2.Ca2SigsBase;
fps = day1.params2p.fps;

%%
example_ROIs = [2, 10, 15];
T = size(F_day1, 1);
t = (0:T-1) / fps;

figure('Position', [200 300 900 200])
for i = 1:length(example_ROIs)
    roi = example_ROIs(i);
    subplot(1, 3, i)
    plot(t, F_day1(:, roi), 'b'); hold on;
    plot(t, F_day2(:, roi), 'r--');
    title(sprintf('Neuron %d', roi))
    xlabel('Time (s)')
    ylabel('ΔF/F')
    legend({'Day 1', 'Day 2'})
end
%%
example_ROIs = [2, 10, 15];
T = size(Fbase_day1, 1);
t = (0:T-1) / fps;

figure('Position', [200 300 900 200])
for i = 1:length(example_ROIs)
    roi = example_ROIs(i);
    subplot(1, 3, i)
    plot(t, Fbase_day1(:, roi), 'b'); hold on;
    plot(t, Fbase_day2(:, roi), 'r');
    title(sprintf('Neuron %d', roi))
    xlabel('Time (s)')
    ylabel('ΔF/F')
    legend({'Day 1', 'Day 2'})
end
%%
% Mean ΔF/F during stimulus
resp1 = mean(Fbase_day1, 1);  % [1 x nROIs]
resp2 = mean(Fbase_day2, 1);

figure
scatter(resp1, resp2, 50, 'filled')
xlabel('Day 1 Response')
ylabel('Day 2 Response')
title('Neuron Responsiveness: Day 1 vs Day 2')
lsline  % regression line
axis equal

%%
% Use PCA + KMeans for quick clustering
X = [F_day1'; F_day2'];  % [2n x T]
[coeff, score] = pca(X);
Z = score(:, 1:5);  % reduce to 5 PCs
k = 3;
labels = kmeans(Z, k);

% Pie chart
figure
counts = histcounts(labels(1:size(F_day1,2)), 1:k+1);  % just Day 1 neurons
pie(counts)
title('Motif Clusters (Day 1)')

%%
BasePks_day1 = zeros(size(F_day1,2),1);
BasePks_day2 = zeros(size(F_day2,2),1);

for i = 1:size(F_day1,2)
    pd = fitdist(day1.Ca2SigsBase(:,i), 'Normal');
    ci = paramci(pd, 'Alpha', .05);
    pk = findpeaks(day1.Ca2SigsBase(:,i), 1/fps, 'MinPeakHeight', ci(2,1)+3*ci(2,2));
    BasePks_day1(i) = numel(pk);
end

for i = 1:size(F_day2,2)
    pd = fitdist(day2.Ca2SigsBase(:,i), 'Normal');
    ci = paramci(pd, 'Alpha', .05);
    pk = findpeaks(day2.Ca2SigsBase(:,i), 1/fps, 'MinPeakHeight', ci(2,1)+3*ci(2,2));
    BasePks_day2(i) = numel(pk);
end

countBins = @(x)[sum(x == 0), sum(x > 0 & x <= 6), sum(x > 6)];
group_day1 = countBins(BasePks_day1);
group_day2 = countBins(BasePks_day2);

figure
bar([group_day1; group_day2]')
set(gca, 'XTickLabel', {'Silent','Normal','Hyperactive'})
legend({'Day 1', 'Day 2'})
ylabel('# Neurons')
title('Spontaneous Activity Classification')
%%
% keep only stable ones
classify = @(x) (x == 0) * 1 + (x > 0 & x <= 6) * 2 + (x > 6) * 3;
cat_day1 = arrayfun(classify, BasePks_day1);  % 1 = Silent, 2 = Normal, 3 = Hyperactive
cat_day2 = arrayfun(classify, BasePks_day2);

% % Only keep ROIs that stayed in same category
same_category = (cat_day1 == cat_day2);
stable_category = cat_day1 .* same_category;  % 0 for unstable, else 1/2/3

figure; hold on
% 
% colors = [0 0.9 0.6;      % 0 = Shifted
%           1 1 0.6;          % 1 = Silent
%           0.6 0.6 1;        % 2 = Normal
%           1 0.4 0.1];       % 3 = Hyperactive
colors = [ 
    0.10, 0.49, 0.73;  % 0 = Dark Blue (shifted)
    0.99, 0.76, 0.28;  % 1 = Yellow/Gold (silent) 
    0.51, 0.83, 0.71;  % 2 = Teal (normal)
    0.90, 0.40, 0.13;  % 3 = Orange/Red (hyperactive)
];


category_labels = {'Shifted', 'Silent', 'Normal', 'Hyperactive'};
scatterHandles = [];

% Plot points by category
for cat = 0:3  % 0 = shifted, 1-3 = stable types
    idx = (stable_category == cat);
    if any(idx)
        h = scatter(resp1(idx), resp2(idx), 100, ...
                    'filled', ...
                    'MarkerFaceColor', colors(cat+1,:), ...
                    'MarkerEdgeColor', 'k');
        scatterHandles(end+1) = h;
    end
end

% Identity line
lims = [min([resp1 resp2]) - 0.05, max([resp1 resp2]) + 0.05];
plot(lims, lims, 'k--', 'LineWidth', 1.2);


% xlabel('My label','FontSize',10) 
xlabel('Mean ΔF/F (Day 1)','FontSize',16)
ylabel('Mean ΔF/F (Day 2)', 'FontSize',16)
title({'Baseline Responsiveness Over Time'}, 'FontSize',16)
% title({['Baseline Responsiveness Over Time'], ['w/ Stable Spontaneous Events']}, 'FontSize',16)
ax = gca; 
ax.FontSize = 16;
axis equal
xlim(lims); ylim(lims);
grid on

% Legend for only present categories
% legend(scatterHandles, category_labels([0 1 2 3]+1), ...
%        'Location', 'bestoutside')

%% Average each gray + stim response block across 5 repetitions (per ROI × orientation)
%   Extract amplitudes by computing ΔF/F = (stim − pre_gray) / pre_gray
%   Plot Day 1 and Day 2 as heatmaps, sorted by temporal peak of Day 1

% 1. Set Parameters
% Define frame counts
nGray = 210;     % 7s at 30Hz
nStim = 150;     % 5s at 30Hz
epoch_len = nGray + nStim;  % 360
n_reps = 5;
n_orient = 8;
n_rois = size(day1.StimProfiles, 2);
%%
% 2. Reshape and Average Across Reps
% Day 1
SP1 = reshape(day1.StimProfiles, epoch_len, n_rois, n_reps, n_orient);
avg_day1 = mean(SP1, 3);  % [360 x 32 x 8]

% Day 2
SP2 = reshape(day2.StimProfiles, epoch_len, n_rois, n_reps, n_orient);
avg_day2 = mean(SP2, 3);
%%
% 3. Extract Pre-Stim and Stim Averages & Compute Amplitudes
% Define time windows
pre_idx = 1:nGray;
stim_idx = nGray+1:nGray+nStim;

% Initialize amplitude matrices: [n_rois x n_orient]
amp_day1 = zeros(n_rois, n_orient);
amp_day2 = zeros(n_rois, n_orient);

for o = 1:n_orient
    for r = 1:n_rois
        baseline1 = mean(avg_day1(pre_idx, r, o));
        stim1 = mean(avg_day1(stim_idx, r, o));
        amp_day1(r, o) = (stim1 - baseline1) / max(baseline1, 1e-3);  % ΔF/F

        baseline2 = mean(avg_day2(pre_idx, r, o));
        stim2 = mean(avg_day2(stim_idx, r, o));
        amp_day2(r, o) = (stim2 - baseline2) / max(baseline2, 1e-3);
    end
end
%%
% 4. Sort ROIs by Peak Amplitude (Day 1)
% Sort by max amplitude across orientations
[~, sort_idx] = sort(max(amp_day1, [], 2), 'descend');

amp_day1_sorted = amp_day1(sort_idx, :);
amp_day2_sorted = amp_day2(sort_idx, :);

%%
figure('Position', [200 300 800 400])

subplot(1,2,1)
imagesc(amp_day1)
colormap('jet'); colorbar
title('Day 1 ΔF/F Amplitude')
xlabel('Orientation')
ylabel('ROIs (sorted by Day 1 peak)')

subplot(1,2,2)
imagesc(amp_day2)
colormap('jet'); colorbar
title('Day 2 ΔF/F Amplitude')
xlabel('Orientation')
ylabel('ROIs (same sort as Day 1)')

%%
consistency = zeros(n_rois, 1);
for i = 1:n_rois
    r = corrcoef(amp_day1(i, :), amp_day2(i, :));
    consistency(i) = r(1, 2);  % correlation coefficient
end

% Rank ROIs by descending correlation
[~, top_idxs] = sort(consistency, 'descend');

%%
n_show = 10;
ori = linspace(0, 360, n_orient+1); ori(end) = [];  % orientation labels

figure('Position', [300 100 1000 800])
for i = 1:n_show
    roi_idx = top_idxs(i);
    
    subplot(5, 2, i)
    plot(ori, amp_day1(roi_idx,:), '-ob', 'LineWidth', 1.5); hold on
    plot(ori, amp_day2(roi_idx,:), '--or', 'LineWidth', 1.5)
    title(sprintf('ROI %d (r = %.2f)', roi_idx, consistency(roi_idx)))
    xlim([0 360]); ylim auto
    xticks(0:90:360)
    xlabel('Orientation (°)')
    ylabel('ΔF/F')
    legend({'Day 1', 'Day 2'}, 'Location', 'northeast')
end

sgtitle('Top 10 Most Consistent Tuning Curves (ΔF/F)')

%%
% 1. Average StimProfiles Across Trials and Normalize 
% Use previous reshaped form:
% [360 x 32 x 5 x 8] → time x ROI x rep x orientation
% SP1 = reshape(day1.StimProfiles, 360, 32, 5, 8);  % [360 x 32 x 5 x 8]
SP1 = reshape(day2.StimProfiles, 360, 32, 5, 8);  % [360 x 32 x 5 x 8]

% 2. Baseline Correction (convert to ΔF/F)
baseline = mean(SP1(1:210, :, :, :), 1);  % pre-stim
dff = (SP1 - baseline) ./ max(baseline, 1e-3);  % [360 x 32 x 5 x 8]

% baseline = mean(avg_traces(1:210, :, :), 1);  % mean over gray window
% dff = (avg_traces - baseline) ./ max(baseline, 1e-3);  % [360 x 32 x 8]
dff_avg = mean(dff, 3);  % [360 x 32 x 8]
dff_avg = squeeze(mean(dff, 3));  % [360 x 32 x 8]

% 3. Reshape to [ROI×Orientations, Time] Matrix
% Reshape: [360 x 32 x 8] → [256 x 360]
% dff_reshaped = reshape(permute(dff, [2 3 1]), [], 360);  % 32*8 = 256 rows
% Permute to ROI × orientation × time
dff_avg_perm = permute(dff_avg, [2, 3, 1]);  % [32 x 8 x 360] % ROI x Orientation x Time

% Reshape to [n_samples x time] = [256 x 360]
dff_reshaped = reshape(dff_avg_perm, [], 360);


% 4. Z-score Across Time (Optional)
% X = zscore(dff_reshaped, 0, 2);  % Z-score each trace (along time)
% X_day1 = zscore(dff_reshaped, 0, 2);  % Z-score each trace (along time)
X_day2 = zscore(dff_reshaped, 0, 2);  % Z-score each trace (along time)

%%
% 5. Determine Cluster Count with Elbow Plot
inertia = [];
K_range = 1:10;

for k = K_range
    [~, ~, sumd] = kmeans(X, k, 'Replicates', 10, 'MaxIter', 300);
    inertia(end+1) = sum(sumd);  % total within-cluster distance
end

figure
plot(K_range, inertia, '-o')
xlabel('Number of Clusters (k)')
ylabel('Total Within-Cluster Distance')
title('Elbow Method for Choosing k')
grid on

%%
% 6. Final Clustering with Chosen k (e.g., k = 4)
optimal_k = 3;
[labels, C] = kmeans(X, optimal_k, 'Replicates', 10, 'MaxIter', 300);  % C = cluster centers

% NOTE: Use same dff_reshaped index logic for day2
% Apply same clustering pipeline (e.g., PCA + centroid assignment)
% Use Day 1 centroids C (from kmeans)
% labels_day2 = knnsearch(C, dff_reshaped);  % assign Day 2 traces to closest cluster center

% 7. Plot Cluster-Averaged Traces
figure('Position', [200 300 800 300])

for k = 1:optimal_k
    subplot(1, optimal_k, k)
%     members = X(labels == k, :);
    members = X(labels == k, :); 
    if isempty(members), continue; end
    avg_trace = mean(members, 1);
    std_trace = std(members, 0, 1);
    time = (0:359) / 30;
    
    plot(time, avg_trace, 'LineWidth', 2); hold on
    fill([time fliplr(time)], ...
         [avg_trace - std_trace, fliplr(avg_trace + std_trace)], ...
         'm', 'FaceAlpha', 0.2, 'EdgeColor', 'none')
    title(sprintf('Day 13: Cluster %d (n=%d)', k, size(members,1)))
    xlabel('Time (s)'); ylabel('z-scored ΔF/F')
    ylim([-2 4])
end

%%
% Option 2: Concatenate Days and Cluster Together
% Treat this as a longitudinal dataset:
% 
% matlab
% Copy
% Edit
% X_all = [X_day1; X_day2];
% X_all_z = zscore(X_all, 0, 2);
% [labels_all, C] = kmeans(X_all_z, k, 'Replicates', 10);

% Separate again
% labels1 = labels_all(1:256);
% labels2 = labels_all(257:end);
% ✅ Benefit: clusters are guaranteed to exist in both datasets
% 📉 Tradeoff: hard to interpret day-specific drift, but excellent for tracking stability

% 🧠 Should You Cluster Per Day or Fix Centroids?
%           Goal	                                              Best Approach
% Interpret each day s structure independently	    Run KMeans per day (like you did)
% Track motif stability across time	                Use fixed clusters from Day 1 (Option 1 or 2)
% Compare group statistics or treatment effects	    Use shared clustering space (Option 2)

% Assuming you've already computed dff_reshaped for both days:
% Each: [256 x 360] = 32 ROIs × 8 orientations

X1 = X_day1; %dff_reshaped_day1;  % Day 1
X2 = X_day2; %dff_reshaped_day2;  % Day 13

X_all = [X1; X2];  % [512 x 360]
X_all_z = zscore(X_all, 0, 2);  % z-score each row trace

k = 3;  % or use elbow method
[labels_all, C] = kmeans(X_all_z, k, 'Replicates', 10, 'MaxIter', 300);

% Separate back into Day 1 and Day 2
labels_day1 = labels_all(1:256);
labels_day2 = labels_all(257:end);

figure('Position', [100 300 1200 500])
time = (0:359)/30;

for cluster_id = 1:k
    n1 = sum(labels_day1 == cluster_id);
    n2 = sum(labels_day2 == cluster_id);

    % --- Day 1 (Top Row)
    subplot(2, k, cluster_id)
    members1 = X_all_z(labels_day1 == cluster_id, :);
    if ~isempty(members1)
        avg1 = mean(members1, 1);
        std1 = std(members1, 0, 1);
        fill([time fliplr(time)], ...
             [avg1 - std1, fliplr(avg1 + std1)], ...
              'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); hold on
        plot(time, avg1, 'b', 'LineWidth', 2); hold on
        title(sprintf('Cluster %d – Day 1 (n = %d)', cluster_id, n1))
    end
    ylim([-2.5 3.5])
    xlabel('Time (s)'); ylabel('Z-scored ΔF/F')
    
    % --- Day 13 (Bottom Row)
    subplot(2, k, cluster_id + k)
    members2 = X_all_z(labels_day2 == cluster_id, :);
    if ~isempty(members2)
        avg2 = mean(members2, 1);
        std2 = std(members2, 0, 1);
        fill([time fliplr(time)], ...
             [avg2 - std2, fliplr(avg2 + std2)], ...
              'm', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); hold on
        plot(time, avg2, 'm', 'LineWidth', 2); 
        title(sprintf('Cluster %d – Day 13 (n = %d)', cluster_id, n2))
    end
    ylim([-2.5 3.5])
    xlabel('Time (s)'); ylabel('Z-scored ΔF/F')
end


% legend({'Day 1 ± STD', 'Day 1 mean', 'Day 13 ± STD', 'Day 13 mean'}, ...
%        'Location', 'southoutside', 'Orientation', 'horizontal')
%%
% Create transition matrix (Day 1 cluster → Day 13 cluster)
trans_matrix = zeros(k, k);
for i = 1:k
    for j = 1:k
        trans_matrix(i, j) = sum(labels_day1 == i & labels_day2 == j);
    end
end

conf_matrix_prop = trans_matrix ./ sum(trans_matrix, 2);  % row-normalized

figure('Position', [500 400 350 300])
% imagesc(trans_matrix)
imagesc(conf_matrix_prop)

colormap('jet'); colorbar
title('Cluster Transitions')
xlabel('Day 13 Cluster')
ylabel('Day 1 Cluster')
axis square
set(gca, 'XTick', 1:k, 'YTick', 1:k)
%%
% Cluster labels
labels_day1 = labels_all(1:256);
labels_day2 = labels_all(257:end);

% Δ spontaneous peaks
% delta_spont = BasePks_day2 - BasePks_day1;
% Expand to match 256 ROI-orient entries
BasePks_day1_exp = repelem(BasePks_day1, 8);  % [32 x 1] → [256 x 1]
BasePks_day2_exp = repelem(BasePks_day2, 8);
delta_spont = BasePks_day2_exp - BasePks_day1_exp;  % [256 x 1]

% % Jittered scatter by cluster
% figure('Position', [400 300 400 300]); hold on
% colors = lines(k);  % or define your own
% for clust = 1:k
%     idx = find(labels_day1 == clust);  % day1-based cluster IDs
%     x = clust + 0.2 * (rand(size(idx)) - 0.5);  % jitter
%     y = delta_spont(idx);
%     
%     scatter(x, y, 40, 'filled', ...
%         'MarkerFaceColor', colors(clust,:), ...
%         'MarkerEdgeColor', 'k')
% end
% 
% % Formatting
% xlim([0.5, k+0.5])
% xticks(1:k)
% xlabel('Cluster ID (based on joint KMeans)')
% ylabel('Δ Spontaneous Peaks (Day13 - Day1)')
% title('Change in Baseline Activity by Cluster')
% yline(0, '--k')  % reference line for no change
% box on

% Assumes:
%   delta_spont = BasePks_day2_exp - BasePks_day1_exp;  % [256 x 1]
%   labels_day1 = joint KMeans labels for Day 1 traces (first 256)
% 
% xlim([0.5, k+0.5])
% xticks(1:k)
% xlabel('Cluster ID (joint KMeans)')
% ylabel('Δ Spontaneous Peaks (Day13 − Day1)')
% yline(0, '-', 'LineWidth', 1.2, 'Color', [0.4 0.4 0.4])
% title('Cluster-Wise Change in Baseline Activity')
% box on

% Inputs:
%   delta_spont: [256 x 1] array of (Day13 - Day1) peak counts
%   labels_day1: [256 x 1] cluster labels from joint clustering

figure('Position', [400 300 400 300]); 

% Make the boxplot first
h = boxplot(delta_spont, labels_day1, ...
    'Colors', 'k', ...           % black outlines
    'Symbol', '.', ...           % show outliers
    'Widths', 0.6);

% Then customize:
boxes = findobj(gca, 'Tag', 'Box');
medians = findobj(gca, 'Tag', 'Median');

% Light blue fill for all boxes
for j = 1:length(boxes)
    patch(get(boxes(j), 'XData'), get(boxes(j), 'YData'), [0.6 0.8 1], ...
        'FaceAlpha', 0.6, 'EdgeColor', 'k', 'LineWidth', 1.2);
end

% Make medians solid black
for j = 1:length(medians)
    medians(j).Color = 'k';
    medians(j).LineWidth = 1.5;
end

% Final plot touches
xlim([0.5, max(labels_day1)+0.5])
xticks(1:max(labels_day1))
xlabel('Cluster ID (joint KMeans)')
ylabel('Δ Spontaneous Peaks (Day13 − Day1)')
yline(0, '--', 'Color', [0.8 0.0 0.0], 'LineWidth', 1.2, alpha=0.5) %[0.3 0.3 0.3]
title('Cluster-Wise Change in Baseline Activity')
box on

%%
% Assumptions:
% - X_all_z is [512 x 360]
% - Rows 1:256 = Day 1; 257:512 = Day 13
% - time = 0:359 / 30 = 12s
% - stim onset at ~7s (frame 210)

time = (0:359)/30;
stim_onset_sec = 7;
n_traces = 12;  % Top N traces to show

% Extract day1 and day13
X_day1 = X_all_z(1:256, :);     % [256 x 360]
X_day13 = X_all_z(257:end, :);  % [256 x 360]

% Rank Day 1 traces by peak z-score
[~, peak_idx] = sort(max(X_day1, [], 2), 'descend');
top_idx = peak_idx(1:n_traces);

% Plot
figure('Position', [600 400 900 500]); hold on
offset = 0;

colors = lines(n_traces);

for i = 1:n_traces
    row_idx = top_idx(i);
    
    z_day1 = X_day1(row_idx, :) + offset;
    z_day13 = X_day13(row_idx, :) + offset;
    
    plot(time, z_day1, '-', 'Color', colors(i,:), 'LineWidth', 1.2);  % Day 1
    plot(time, z_day13, '-', 'Color', colors(i,:), 'LineWidth', 1); % Day 13
    
    offset = offset + 6;  % vertical spacing
end

xline(stim_onset_sec, 'r--', 'LineWidth', 1.5)
xlabel('Time (s)')
ylabel('Z-scored ΔF/F (offset)')
title('Top Z-scored Responses — Day 1 (solid) vs Day 13 (dashed)')
ylim([-1, offset])
box on

%% PLOT TRACES _ ALL
roi_id = 2;  % change this from 1 to 32 to loop through all ROIs
orientations = 1:8;

start_frame = 151;
end_frame = 360;
kept_frames = end_frame - start_frame + 1;  % = 210
time = (0:kept_frames - 1) / 30;

stim_onset_frame = 211 - start_frame + 1;  % = 61
stim_onset_time = (stim_onset_frame - 1) / 30;  % = 2.0 s


% Pull traces for one ROI (each has 8 epochs/orientations)
% ROI i occupies rows: [(i-1)*8 + 1 : i*8]
idx_day1 = (roi_id-1)*8 + (1:8);            % day 1 rows
idx_day13 = idx_day1 + 256;                 % day 13 rows

Z_day1 = X_all_z(idx_day1, start_frame:end);    % [8 x 255]
Z_day13 = X_all_z(idx_day13, start_frame:end);  % [8 x 255]

% Plot
figure('Position', [600 300 700 300]); hold on
colors = lines(8);
for o = 1:8
    y1 = Z_day1(o,:) + (o-1)*6;
    y2 = Z_day13(o,:) + (o-1)*6;

    plot(time, y1, '-', 'Color', colors(o,:), 'LineWidth', 1.4)
    
    % Simulate alpha by mixing with white
    faded = colors(o,:) * 0.5 + [1 1 1] * 0.5;
    plot(time, y2, '-', 'Color', faded, 'LineWidth', 1.4)
end

xline(stim_onset_time, 'r-', 'LineWidth', 1.5)  % stimulus starts at frame 1 (after trim)
xlabel('Time (s)')
ylabel('Z-scored ΔF/F (offset)')
title(['ROI ' num2str(roi_id) ' — 8 Orientation Trials'])
ylim([-1, 8*4])
yticks([])
box on


%% AS SUBPLOTS NPW
% Inputs:
%   X_all_z: [512 x 360] — zscored traces (32 ROIs × 8 orientations × 2 days)
%   Assumes Day1 = top half, Day13 = bottom half

% Time parameters
% Parameters
start_frame = 151;
end_frame = 360;
kept_frames = end_frame - start_frame + 1;
time = (0:kept_frames - 1) / 30;
stim_onset_time = (211 - start_frame) / 30;

n_rois = 32;
n_orient = 8;
rows = 4;
cols = 8;
colors = lines(n_orient);

% Set up layout
figure('Position', [100 100 3200 800])
tl = tiledlayout(rows, cols, 'Padding', 'compact', 'TileSpacing', 'compact');

for roi = 1:n_rois
    ax = nexttile;

    idx_day1 = (roi - 1) * n_orient + (1:n_orient);
    idx_day13 = idx_day1 + 256;

    z1 = X_all_z(idx_day1, start_frame:end_frame);
    z2 = X_all_z(idx_day13, start_frame:end_frame);

    offset = 0;
    for o = 1:n_orient
        y1 = z1(o,:) + offset;
        y2 = z2(o,:) + offset;

        plot(time, y1, '-', 'Color', colors(o,:), 'LineWidth', 1); hold on
        faded = colors(o,:) * 0.5 + [1 1 1] * 0.5;
        plot(time, y2, '-', 'Color', faded, 'LineWidth', 1);

        offset = offset + 4;
    end

    xline(stim_onset_time, 'r-', 'LineWidth', 1)

    xlim([0, 7])
    ylim([-1, 8*4])
    xticks([0 2 4 6])
    yticks([])
    title(['ROI ' num2str(roi)], 'FontSize', 10)
end

% Add shared axis labels AFTER plotting
xlabel(tl, 'Time (s)', 'FontSize', 14)
ylabel(tl, 'Z-scored ΔF/F (offset)', 'FontSize', 14)
sgtitle('ROI-wise Orientation Responses — Day 1 (solid) vs Day 13 (dashed)', 'FontSize', 16)

% Save as PDF
print(gcf, 'ROI_Orientation_Traces_Comparison.pdf', '-dpdf', '-bestfit')
print(gcf, 'ROI_Orientation_Traces_Comparison.png', '-dpng', '-r300')
%%

% heat plot versions?
% Define frame trimming and time axis
start_frame = 151;
end_frame = 360;
time = (0:(end_frame - start_frame)) / 30;

% Define stim_window relative to trimmed trace (frames 211–360 of full trace)
stim_start = 211;
stim_end = 360;
% Convert to trimmed-frame indices
stim_window = (stim_start - start_frame + 1):(stim_end - start_frame + 1);  % 61–210


figure('Position', [200 100 800 600])
% tiledlayout(4, 8, 'Padding', 'tight', 'TileSpacing', 'compact');
tl = tiledlayout(4, 8, 'Padding', 'tight', 'TileSpacing', 'compact');
clim = [-3 3];  % adjust to make stim stand out more

for roi = 1:32
    nexttile

    idx_d1 = (roi-1)*8 + (1:8);
    idx_d13 = idx_d1 + 256;

    Z1 = X_all_z(idx_d1, start_frame:end_frame);
    Z2 = X_all_z(idx_d13, start_frame:end_frame);

    % Find stim peak values in Z1 for sorting
    [~, peak_order] = sort(max(Z1(:,stim_window), [], 2), 'descend');

    % Apply sorting
    Z1_sorted = Z1(peak_order, :);
    Z2_sorted = Z2(peak_order, :);

    Zall = [Z1_sorted; Z2_sorted];  % 16 x 210

%     imagesc(time, 1:16, Zall)
    imagesc(time, 1:16, Zall, clim);
    yline(8.5, 'w-', 'LineWidth', 2.0);  % White divider between Day 1 and Day 13
    colormap('turbo')  % or parula, etc.
    caxis([-2 3])

    xline(2, 'r-', 'LineWidth', 2.0)

    set(gca, 'YTick', [4.5, 12.5], 'YTickLabel', {'Day 1', 'Day 13'})
    title(['ROI ' num2str(roi)])
    xticks([0 3 6])
end

% Shared label
% xlabel(tiledlayout, 'Time (s)')
% ylabel(tiledlayout, 'Orientation Rank (Day1-based)')
xlabel(tl, 'Time (s)')
ylabel(tl, 'Orientation Rank (Day1-based)')
sgtitle('Stimulus-Aligned Z-Traces by Orientation (Each ROI)')

% Save
print(gcf, 'Heatmap_Orientation_ByROI_Day1_Day13.pdf', '-dpdf', '-bestfit')
%%
% /ALA FEBO
% ROI ID to plot
roi = 10;

% Constants
orient_order = [0 45 90 135 180 225 270 315];
fps = 30;
epoch_len = 360;
pregray_len = 210;
stim_len = 150;

% Extract Day 1 and Day 13
idx_d1 = (roi-1)*8 + (1:8);
idx_d13 = idx_d1 + 256;

Z1 = X_all_z(idx_d1, :);   % [8 x 360]
Z2 = X_all_z(idx_d13, :);  % [8 x 360]

% Concatenate epochs
trace_d1 = reshape(Z1', 1, []);   % [1 x 2880]
trace_d13 = reshape(Z2', 1, []);

% Time vector
T = length(trace_d1);
t = (0:T-1)/fps;  % in seconds

% Plot
figure('Position', [300 400 1000 300]); hold on

offset=8;

% Plot stimulus zone highlights (light blue)
for i = 0:7
    stim_start = i * epoch_len + pregray_len + 1;
    stim_end = (i+1) * epoch_len;
    fill([t(stim_start) t(stim_end) t(stim_end) t(stim_start)], ...
         [min([trace_d1 trace_d13])-0.2, min([trace_d1 trace_d13])-0.2, ...
          max([trace_d1 trace_d13])+0.2+offset, max([trace_d1 trace_d13])+0.2+offset], ...
         [0.85 0.92 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
end

% Overlay traces
plot(t, trace_d1 + offset , 'k-', 'LineWidth', 1.0)
plot(t, trace_d13, 'r-', 'LineWidth', 1.0)

% Vertical lines at each epoch start (optional)
for i = 1:7
    xline(i * epoch_len / fps, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5)
end

% Labels
xlabel('Time (s)')
ylabel('Z-scored ΔF/F')
title(['ROI ' num2str(roi) ' across all orientations'])
ylim padded
xlim([0 max(t)])
box off

% Optional: orientation labels (at stim center)
for i = 0:7
    stim_center = (i * epoch_len + pregray_len + stim_len/2) / fps;
    text(stim_center, min([trace_d1 trace_d13])-0.1, ...
         [num2str(orient_order(i+1)) '°'], ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
         'FontSize', 10, 'Color', [0.3 0.3 0.3]);
end

legend({'Stimulus (150f)', 'Day 1', 'Day 13'}, 'Location', 'northeast')
