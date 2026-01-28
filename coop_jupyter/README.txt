UF MBI NeuroMicroscope Co-op

This notebook runs locally on your Mac using Jupyter (opens in browser, Python-based).

This contains instructions for installing Miniforge on a Mac (Silicon chip).

Miniforge is a lightweight, community-driven installer for the Conda package manager, focused on using the conda-forge channel by default for open-source packages, offering a minimal setup for Python and other languages, and often including the faster Mamba solver and support for various CPU architectures like Apple Silicon (M1/M2/M3). It's an alternative to Miniconda/Anaconda, providing faster, more accessible dependency resolution for data science and scientific computing.

In sum, Miniforge (conda) will allow the running and sharing of Python and related scripts and programs for processing and analyzing image data (and any data for that matter!). 

One-time setup for Mac OS (10–15 minutes):

1) Download/install Miniforge from:
https://github.com/conda-forge/miniforge

See the website for install details - here is minimal excerpt:
- Unix-like platforms (macOS, Linux, & WSL):
From a terminal window, download the installer appropriate for your computer's 		architecture using curl or wget or your favorite program.

	For example, to download 'Miniforge3-MacOSX-arm64.sh' file for new Mac Silicon chips (arm):
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"

Navigate to the local Download directory
	For example: cd ~/Downloads
	Check for file in the new directory: ls
	Once you are in the correct directory, run the install script:
bash Miniforge3-MacOSX-arm64.sh

	The interactive installation will prompt you to initialize conda with your shell. This is typically with recommended workflow.

2) Once installed, close and re-Open Terminal.

3) Navigate to the folder with the notebook file (*.ipynb file - see example below)

	Run:

	conda env create -f environment.yml
	conda activate scanimage_env
	jupyter lab

4) A browser window will open — double-click the notebook to run it.

Now you can click the "play" icon at the top to run each "cell" of code individually. If present, an output/result will appear below each cell (or update the existing output from a prior run). There are plenty of tutorials and guides to help navigate (hint - Gemini or chatGPT can be a great help), but feel free to reach out anytime (jcoleman@ufl.edu).

5) After this first setup, you only need to type the following lines to run Jupyter:

	conda activate scanimage_env
	jupyter lab

6) Send any questions to jcoleman@ufl.edu. Optional - run Jupyter via HiperGator (will require HiperGator access and ability to install proper packages/YML file.


Extra notes:
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
