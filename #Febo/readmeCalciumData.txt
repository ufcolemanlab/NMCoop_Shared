Files for Febo lab - calcium test analysis:

3 different data sets from Thy1-GCaMP6s Tg mice (GP4.12; https://www.jax.org/strain/025776), head-fixed, awake under visual stimulation. Some movement may be detected in the traces (e.g., coordinated drift across channels/ROIs). All data sets are sample from the same cells/ROIs across sessions.

Data sets 1 and 2
Day 1 and Day 2 of imaging the same set of cells (ROIs) in ~layers 2/3 under identical conditions (10Hz drifting sinusoidal gratings, 8 orientations)

Data set 3
One day of data, same cells in Data sets 1 and 2; 2 different visual stimulation regimes 		   (0.5Hz phase-reversing sinusoidal gratings, 5 orientations)

- PNG/EPS files contain cell maps with ROI labels (TIF file shows average intensity from raw Tseries)
- CSV files contain "raw intensity data", "centroid XY coordinates"
- TXT files contain information about each experiment, but all summarized in the Jupiter Notebook as well.

I recommend starting with 'Data sets 1 and 2', which I tried to walk you through in the notebook file. Both data sets are are the same cells, but imaged several days apart (set 1 on 9/23/17; set 2 on 10/17/17).

The other data set is also from 10/17/17 (same day as set 2), but viewing a slightly different stimulus - 0.5Hz phase-reversing sinusoidal gratings vs drifting gratings. Could be of interest - some fodder for analysis by comparing the same cells under different viewing conditions.

I have the timestamp files, but the code for extrapolating needs to be re-worked. Nonetheless, I included basic info about how the data are arranged and how it can be "chunked" by session (for example start-stop indices for each 'epoch' by extrapolating times). Alternatively, you can just focus on "gray" sections at the beginning/end for spontaneous activity or just analyze the whole stretch and look for patterns/correlations/etc! 

Hopefully the Jupyter Notebook file (>Python 3.9) has all the info you need, but feel free to email jcoleman@ufl.edu with any questions.
