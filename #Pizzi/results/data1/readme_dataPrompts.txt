7-24-20

Can you interpret these plots? And provide some synthesis of the results? For context, the data analyzed are from measurements in mouse brain vasculature imaged using intravital 2-photon imaging and are derived from ROIs in 3D image stacks acquired using galvo-galvo scanner and 6X zoom with a Nikon 16x lens. There are two groups represented (control-red and experimental-black) but the data were analyzed blind to group (only represented in the PCA plot by text color of ROI#). Here are stattisal results when comparing group PC scores : Results for PC1:
  Test: Mann-Whitney U test
  Statistic: 7534.0, p-value: 0.4224659152628756
Results for PC2:
  Test: Mann-Whitney U test
  Statistic: 6909.0, p-value: 0.062434681758270716
Results for PC3:
  Test: t-test
  Statistic: 0.7367184225588419, p-value: 0.4619655774546293


****

Is PC2 likely to be Zmicrons (coritcal depth) or FeFv (measure of vessel leakage)? Also if I can include X,Y coordinates and Additional data related to T-series of each metric (except X. Y. and Z - eg mean, stdev, min, max for a time series at each Zstep), do you think that could up the sensitivity/power of the analysis? Finally can i use the PCs scores to feed into a clustering algorithm like K-means or UMAN?

*****


Would it change the PCA analhsis results if also included the grouping variable for splitting the 2 groups? ie would it enhance the power?