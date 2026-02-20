UF MBI Microscopy Facility | NeuroMicroscope Co-op

This notebook runs locally on your Mac using Jupyter (opens in browser, Python-based).

This contains instructions for installing Miniforge on a Mac (Silicon chip).

Miniforge is a lightweight, community-driven installer for the Conda package manager, focused on using the conda-forge channel by default for open-source packages, offering a minimal setup for Python and other languages, and often including the faster Mamba solver and support for various CPU architectures like Apple Silicon (M1/M2/M3). It's an alternative to Miniconda/Anaconda, providing faster, more accessible dependency resolution for data science and scientific computing.

Note on Conda - 'env' stands for environment and allows you create different Python installations with different pace kages/modules so you avoid conflicts, breaking important installs, etc.. So this tutorial takes you through how to create a env for working with ScanImage files, but general image analysis, plotting, stats, working with arrays etc. In the future, I plan to add new envs for different applications (ie will provide new 'YML' files to create new conda environments).

In sum, Miniforge (conda) will allow the running and sharing of Python and related scripts and programs for processing and analyzing image data (and any data for that matter!). 

One-time setup for Mac OS (10–15 minutes):

1) Download/install Miniforge from:
https://github.com/conda-forge/miniforge

See the website for install details - here is minimal excerpt:
- Unix-like platforms (macOS, Linux, & WSL):
From a terminal window, download the installer appropriate for your computer's architecture using curl or wget or your favorite program.

	For example, to download 'Miniforge3-MacOSX-arm64.sh' file for new Mac Silicon chips (arm):
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"

Navigate to the local Download directory (see Appendix B below)
	For example: cd ~/Downloads
	Check for file in the new directory: ls
	Once you are in the correct directory, run the install script:
bash Miniforge3-MacOSX-arm64.sh

	The interactive installation will prompt you to initialize conda with your shell. This is typically with recommended workflow.

2) Once installed, close and re-Open Terminal.

3) Navigate to the folder with the notebook file (*.ipynb file - see example below)

	Run:
	 conda env create -f scanimage_env.yml
	 conda activate scanimage_env
	 jupyter lab

4) A browser window will open — double-click the notebook to run it.

Now you can click the "play" icon at the top to run each "cell" of code individually. If present, an output/result will appear below each cell (or update the existing output from a prior run). There are plenty of tutorials and guides to help navigate (hint - Gemini or chatGPT can be a great help), but feel free to reach out anytime (jcoleman@ufl.edu).

5) After this first setup, you only need to type the following lines to run Jupyter:

	conda activate scanimage_env
	jupyter lab

6) Once established, you can add new packages/modules or create new 'ends' etc using new 'YML' files. To add new packages/modules, update the existing 'scanimage_env' conda environment (or other env you want to update) as follows (NOTE: It's very important that you only execute these commands when the scanimage_env (or other env) is showing activated in your Terminal prompt (ie not 'base').

	For example:

		conda activate scanimage_env
		conda env update -f scanimage_env_update1.yml

*** <><><><><><><><><><><><><> ***
APPENDIX A
> Use scanimage_env for: importing ZT TIF stacks and metadata from ScanImage, QC plots, basic processing.
> Use scanimage_napari_env for: browsing huge volumes / light-sheet stacks interactively.
> [Future: Use scanimage_ml_env for: Cellpose / Torch / heavy segmentation.]

Napari - Create and install scanimage_napari_env for browsing huge volumes / light-sheet stacks interactively.

	YouTube link below for introduction, background, and practical, interactive demos. There may even be an associated Jupyter Notebook linked under the video.
		Intro video link: https://www.youtube.com/watch?v=I473cUAly8A

	Install and use Napari image viewer and processing/annotation tool (the last step allows you to use Napari in your Jupyter Notebook):

	 conda env create -f scanimage_napari_env.yml
	 conda activate scanimage_napari_env
	 python -m ipykernel install --user --name scanimage_napari_env --display-name "ScanImage-Napari"

	Then you should be able to 
	1) Launch Napari from your terminal:
		conda activate scanimage_napari_env
		napari

	2) Launch w/in a JupyterLab notebook:
		Launch JupyterLab as above and select the napari kernel
			Go to MAIN MENU and select Kernel->Change Kernel... then choose scanimage_napari_env)

		Then in a notebook cell, run Napari as follows:
		import napari
		napari.Viewer()


Always:
--> launch JupyterLab from scanimage_env (ie run 'conda activate scanimage_env', then 'jupyter lab')
--> Once JupyterLab is running, Pick the kernel that matches what you’re doing (MAIN MENU: Kernel->Change Kernel...) :

	ScanImage (scanimage_env) → analysis
	ScanImage-Napari (scanimage_napari_env → interactive image browsing
	[Future: ScanImage-ML (scanimage_ml_env) → segmentation / ML]


----> Send any questions to jcoleman@ufl.edu. Optional - run Jupyter via HiperGator (will require HiperGator access and ability to install proper packages/YML file.


APPENDIX B
If you want to try and use the notebooks from the YouTube demo, make a dedicated “workshop” env (recommended) - navigate to the folder with 'napari_latam.yml' file:
# installs mamba as an alternative to conda for installation - makes sure you have a good solver (conda-forge can be slower)
conda install -n base -c conda-forge mamba

# create the workshop env from their yml
mamba env create -f napari_latam.yml
conda activate napari-latam
python -m ipykernel install --user --name napari-latam --display-name "Napari-LatAm-2023"


APPENDIX C
Example: How to navigate to a specific directory in Terminal.

(Base) jcoleman@macbookpro % cd ~/Documents
(Base) jcoleman@macbookpro Documents % ls
--LARGE DATA--			MadMapper			PUBLIC				tztack_8bit.tif
Arduino				MATLAB				PYTHON				Zoom
GitHub				Presentations and Figures	Resolume Arena
JA_LAB_DATA			Processing			stv.terminal


HiperGator organization example:
/blue/MBI_MC/
├── incoming/
│   ├── user_lastname_project/
│   └── README.txt
├── raw/
├── processed/
