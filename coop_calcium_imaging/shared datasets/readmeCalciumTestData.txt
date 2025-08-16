Files for calcium test analysis

MATLAB Notes:
To process raw calcium signals from image ROIs, start with Matlab code (R2023b-, R2021b-tested) 'calcium_2p_mousecortex.m' (Plus dependencies: 'node.m', 'circularGraph.m').

PYTHON Notes: Coming soon. Code available upon request (jcoleman@ufl.edu)

Under '/shared datasets/'
Two different data sets from Thy1-GCaMP6s Tg mice (GP4.12; https://www.jax.org/strain/025776), head-fixed, awake under visual stimulation. Some movement may be detected in the traces (e.g., coordinated drift across channels/ROIs). All data sets are sample from the same cells/ROIs across sessions.


General scheme of the files:
The first 60sec (i.e. the first 60s*30fps=1800frames) is "spontaneous" activity in awake primary visual cortex (V1) while viewing a gray screen (50% contrast).

At frame# 1801, the subjects begin viewing a series of sinusoidal gratings of 5 or 8 different orientations and interleaved with gray screen. Each signal trace or region of interest (ROI) represents a single neuron.


Data sets 1 and 2:
Day 1 and Day 2 of imaging the same set of cells (ROIs) in ~layers 2/3 under identical conditions (10Hz drifting sinusoidal gratings, 8 orientation-directions, 5 sessions each for averaging)

- PNG/EPS files contain cell maps with ROI labels (TIF file shows average intensity from raw Tseries)
- CSV files contain raw data from FIJI ("raw intensity data", "centroid XY coordinates", etc)

I recommend starting with 'Data sets 1 and 2', which you can to walk through in the notebook file.
Both data sets are are the same rois/cells, but imaged several days apart (set 1 on 9/23/17; set 2 on 10/17/17).

NOTE: If only interested in 'spontaneous' non-visual stimulus driven epochs, the "gray" sections at the beginning/end can be used, ignoring the rest. Or, of course, you can process and ignore the timestamps altogether as the data are continuous. 

Feel free to email jcoleman@ufl.edu or contact via our Slack channel with any questions.
