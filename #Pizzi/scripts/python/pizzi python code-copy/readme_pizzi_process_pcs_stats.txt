_01

Code to import data from Pickle file, run PCA and stats

GITHUB
Links - ?

8-6-24 Notes
Working on parsing data by 'vessel type' and 'vessel location'

Yes, absolutely. One of the best methods for characterizing vessels in vivo was:

Classification of cerebral vessels
In reporter mice, cerebral blood vessels were manually classified pre M2CAO using the following criteria. Arteries (A) possess elongated, streamlined endothelial cells and are > 45 µm in diameter, while arterioles (Ae) branch off arteries and are 10–45 µm in diameter. Veins (V) have irregular, uneven endothelial cells and are > 50 µm in diameter, while venules (Ve) converge to form veins and are 10–50 µm in diameter. A single endothelial cell covers the whole lumen of a capillary (C) which has a diameter of < 10 µm.
2:18
From the paper: Fluids and Barriers of the CNS (2024) 21:35

**Need to acquire ROIs w/ renaming - use combo code to parse later (add or rename? Can still use index and sort even if names all the same - use 'List' to view details, Z etc)

**
# need to convert 'vessel type' and 'location along AV axis' to numerical categories
# eg arteriole=1, capillary=2, venule=3
# eg Prior to bifurctation=1, post bifurcation=2, mid-capillary=3, mid vessel=4
# Define the mapping dictionaries
vessel_type_map = {
    'arteriole': 1,
    'capillary': 2,
    'venule': 3
}

location_map = {
    'Prior to bifurctation': 1,
    'pre bifurcation': 1, #calling the same as 'Prior...'
    'post bifurcation': 2,
    'mid-capillary': 3,
    'mid vessel': 4,
    ' mid vessel': 4 #calling the same as 'mid vessel' inclded for one 'typo' with space in front of 'mid'
}

https://github.com/ufcolemanlab/NMCoop_Shared/tree/d6cb5eb3115bdeab3e0b16ba44b024ce54cae605/%23Pizzi