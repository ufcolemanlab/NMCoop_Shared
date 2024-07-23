"""
This Jython script for FIJI performs the following tasks:

1. Opens an image stack selected by the user and removes scale (set to pixels only).
2. Optionally applies a 3D median filter and adjusts the brightness/contrast.
3. Allows the user to select a Lookup Table (LUT) for visualization.
4. Selects the line drawing tool with a default width of 20 pixels.
5. Opens the ROI manager window and prompts the user to draw ROIs.
6. Saves the drawn ROIs as a ZIP file with a user-inputted tag in the filename.
7. Prints the path where the ZIP file is saved.
8. Closes all open windows after the process is complete.
You can adjust the script as needed for your specific requirements and workflow.
"""

# Import necessary classes from Fiji
from ij import IJ
from ij.gui import GenericDialog, WaitForUserDialog
from ij.plugin.frame import RoiManager
from ij.WindowManager import getWindow

# Import necessary classes from the ImageJ API
from ij import IJ
from ij.measure import Calibration

# Function to open a TIF file and remove the scale
def open_tif_and_remove_scale(filepath):
    # Open the image
    imp = IJ.openImage(filepath)
    
    # Check if the image was successfully opened
    if imp is None:
        print("Error: Could not open image.")
        return
    
    # Get the current calibration of the image
    cal = imp.getCalibration()
    
    # Print the current calibration (for debugging purposes)
    print("Current calibration:", cal)
    
    # Set the unit to pixels
    cal.setUnit("pixel")
    
    # Remove the scale by setting the pixel width and height to 1.0
    cal.pixelWidth = 1.0
    cal.pixelHeight = 1.0
    
    # Apply the modified calibration to the image
    imp.setCalibration(cal)
    
    # Print the modified calibration (for debugging purposes)
    print("Modified calibration:", cal)
    
    # Show the image
    #imp.show()
    
    return imp


def apply_median_filter(imp):
    IJ.run(imp, "Median 3D...", "x=2 y=2 z=1")  # Applying 2x2x1 median filter

def apply_lut(imp, selected_lut):
    if selected_lut != "Grayscale":
        IJ.run(imp, selected_lut, "")
    else:
        IJ.run(imp, "Grays", "")

def select_line_tool_with_width(line_width):
	IJ.setTool("line")
	IJ.run("Line Width...", "width=" + str(line_width))

def open_roi_manager():
    IJ.run("ROI Manager...")

def save_roi_zip(filepath):
    roi_manager = RoiManager.getInstance()
    if roi_manager is not None:
        roi_manager.runCommand("Save", filepath)
        
#def create_stack_ontage(imp):
#%imp should be a stack - options should include #col,rows and #slices dsplayed so user can input col-rows

# Prompt the user to select an image stack
stack_path = IJ.getFilePath("Select Image Stack")
if stack_path is None:
    print("User canceled the selection.")
    quit()

# Open the image stack
#imp = IJ.openImage(stack_path)
#imp.show()
imp = open_tif_and_remove_scale(stack_path)
imp.show()

# Create a dialog to select options
gd = GenericDialog("Image Stack Options")
gd.addCheckbox("Open Brightness/Contrast Window", True)
gd.addCheckbox("Apply 3D Median Filter", True)
gd.showDialog()

# Get the selected options
open_bc_window = gd.getNextBoolean()
apply_median_filter_flag = gd.getNextBoolean()

# Open the brightness/contrast window if selected
if open_bc_window:
    IJ.run("Brightness/Contrast...")

# Apply 3D Median Filter if selected
if apply_median_filter_flag:
    apply_median_filter(imp)

# Create a dialog to select Lookup Table (LUT)
gd_lut = GenericDialog("Lookup Table (LUT)")
gd_lut.addChoice("Lookup Table (LUT):", ["Grayscale", "Fire", "Red", "Green", "mpl-inferno", "mpl-viridis", "Rainbow RGB", "Spectrum", "Thermal"], "Fire")
gd_lut.showDialog()

# Get the selected LUT
selected_lut = gd_lut.getNextChoice()

# Apply the selected LUT
apply_lut(imp, selected_lut)

# Select the line drawing tool and set the width to 20 pixels by default
select_line_tool_with_width(20)

# Open the ROI manager window
open_roi_manager()

# Wait for the user to finish drawing ROIs
myWait = WaitForUserDialog("Draw ROIs", "Draw the ROIs on the image, press 't' to add to ROI mgr, and click OK when finished.")
myWait.show()

# Get the ROI manager window
roi_manager = getWindow("ROI Manager")
if roi_manager is None:
    print("ROI Manager is not open.")
    quit()

# Activate the ROI manager window
roi_manager.toFront()

# Get the filename from the TIF file path
#filename = IJ.getFileInfo().fileName
# Get the image info
file_info = imp.getOriginalFileInfo()
if file_info is None:
    print("Error: Could not get file information.")
    quit()

filename = file_info.fileName
if not filename.endswith(".tif"):
    print("Invalid file format. Please select a TIF image stack.")
    quit()

# Replace ".tif" with ".zip" to create the filename for the ZIP file
#zip_filename = filename.replace(".tif", ".zip")
#could add user tag here = prompt for text string to add = <user tag> + "_.zip"
# Create a dialog window for user input

gd = GenericDialog("Select a ROI-ZIP filename class")
gd.addCheckbox("classA", True)
gd.addCheckbox("classB", False)
gd.addCheckbox("classC", False)
gd.addCheckbox("User input", False)
gd.showDialog()

#for checkboxes:
# Get the selected options
classA_tag = gd.getNextBoolean()
classB_tag = gd.getNextBoolean()
classC_tag = gd.getNextBoolean()
manual_tag = gd.getNextBoolean()

#if manual_tag:
#Use this if user-input option used above
#gd = GenericDialog("Enter ZIP filename tag")
#gd.addStringField("Filename:", "")

## Check if OK button was clicked
#if gd.wasOKed():
#    # Get the user input
#    user_input = gd.getNextString()
#else:
#    print("Dialog was canceled.")
#    quit()

# Use the user input in the filename
#zip_filename = filename.replace(".tif", "_ROIs_" + user_input + ".zip")
####

# Use the user or "class" input in the filename checkbox above
if classA_tag:
	user_input = "classA"
if classB_tag:
	user_input = "classB"
if classC_tag:
	user_input = "classC"
#if manual_tag:	
	# could make flexible number of classes? or >3 use the user input option

zip_fullpath = stack_path.replace(".tif", "_ROIs_" + user_input + ".zip")

# Save the ROI manager data as a ZIP file
#save_roi_zip(zip_filename)
save_roi_zip(zip_fullpath)

# Print the path where the ZIP file is saved
print("ROI data saved as:", zip_fullpath)

# CLOSE ALL
# Close ROI manager if it's open
if roi_manager is not None:
    roi_manager.close()

# Close contrast adjustment window if it's open
contrast_window = getWindow("B&C")
if contrast_window is not None:
    contrast_window.close()

# Close all other open windows
IJ.run("Close All")
