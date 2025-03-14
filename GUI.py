import PySimpleGUI as sg
import json
import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


"""
GUI for displaying computed balance data
- Reads user inputs (Disk Radius, Shaft Weight, Trial Weights)
- Loads computed values (Weight 1, Weight 2, Angle 1, Angle 2) from `data2.json`
- Provides a "Refresh Data" button to update values dynamically
"""

DATA_FILE = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\data.json"
DATA_FILE_2 = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\DOSYALAR\\data2.json"
BALANCE_SCRIPT = r"C:\\Users\\alita\\OneDrive\\Masaüstü\\INS_SON_KOD\\Deico_balance_S2_rev2.py"

def Unbal_func(a1, a2):
    try:
        a1 = float(a1)  # Convert to float
        a2 = float(a2)  # Convert to float
        return a1 * a2
    except ValueError:
        print(f"Invalid data: A1={a1}, A11={a2}")
        return "Error"

# Function to load computed balance data
def load_balance_data():
    if not os.path.exists(DATA_FILE_2):
        return {"magnitude_Wbal1": "N/A", "magnitude_Wbal2": "N/A", "angle_Wbal1_deg": "N/A", "angle_Wbal2_deg": "N/A", "A1": "N/A", "A11": "N/A", "a_x1": "N/A", "a_x2": "N/A","a_y1": "N/A", "a_y2": "N/A","a_x11": "N/A", "a_x12": "N/A","a_y11": "N/A", "a_y12": "N/A","a_x21": "N/A", "a_x22": "N/A","a_y21": "N/A", "a_y22": "N/A"}
    
    try:
        with open(DATA_FILE_2, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"magnitude_Wbal1": "N/A", "magnitude_Wbal2": "N/A", "angle_Wbal1_deg": "N/A", "angle_Wbal2_deg": "N/A", "A1": "N/A", "A11": "N/A", "a_x1": "N/A", "a_x2": "N/A","a_y1": "N/A", "a_y2": "N/A","a_x11": "N/A", "a_x12": "N/A","a_y11": "N/A", "a_y12": "N/A","a_x21": "N/A", "a_x22": "N/A","a_y21": "N/A", "a_y22": "N/A"}
    

def create_polar_plot():
    balance_data = load_balance_data()
    
    if balance_data["magnitude_Wbal1"] == "N/A" or balance_data["magnitude_Wbal2"] == "N/A":
        return plt.figure()
    
    angles = [
        np.deg2rad(float(balance_data["angle_Wbal1_deg"])),
        np.deg2rad(float(balance_data["angle_Wbal2_deg"]))
    ]
    magnitudes = [
        float(balance_data["magnitude_Wbal1"]),
        float(balance_data["magnitude_Wbal2"])
    ]
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_title("Correction Weights on Disk")
    ax.set_ylim(0, max(magnitudes) * 1.2)
    ax.scatter(angles, magnitudes, color='red', s=100, label="Correction Weights")
    for angle, mag in zip(angles, magnitudes):
        ax.text(angle, mag, f"{mag:.2f}g", fontsize=12, ha='left', va='bottom')
    ax.legend()
    return fig

# Function to draw the polar plot
def draw_polar_plot(canvas, fig):
    for widget in canvas.winfo_children():
        widget.destroy()
    canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack(fill='both', expand=True)


# Function to save values to JSON
def save_data(disk_radius, shaft_weight, trial_weight1, trial_weight2):
    data = {
        "Disk Radius": disk_radius,
        "Shaft Weight": shaft_weight,
        "Trial Weight 1": trial_weight1,
        "Trial Weight 2": trial_weight2
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

# Function to load existing data (if available)
def load_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"Disk Radius": "", "Shaft Weight": "", "Trial Weight 1": "", "Trial Weight 2": ""}

# Function to run the balance script and wait for `data2.json`
def run_balance_script():
    try:
        subprocess.Popen(["python", BALANCE_SCRIPT], shell=True)  # Run script in background
        sg.popup("Balance calculation started. Please wait...")
    except Exception as e:
        sg.popup_error(f"Error starting balance calculation:\n{e}")

# Load existing data
stored_data = load_data()
balance_data = load_balance_data()

# Define the table headers and data
table_headers = ["No Weight Added", "Weight on Plane 1", "Weight on Plane 2"]
# Initialize table with "N/A" (will be updated on Refresh)
table_data = [
    ["N/A", "N/A", "N/A"],
    ["N/A", "N/A", "N/A"],
    ["N/A", "N/A", "N/A"],
    ["N/A", "N/A", "N/A"],
]

# Create the table layout
table_layout = [
    [sg.Text("Amplitude Data", font=("Arial", 12, "bold"))],
    [sg.Table(values=table_data, headings=table_headers, auto_size_columns=True,
              justification="center", key="-TABLE-", num_rows=4)]
]


# Define layouts for each segment
segment1_layout = [
    [
        sg.Column([      # Left-side input and results
            [sg.Text("Disk Radius:", size=(15, 1)), sg.InputText(stored_data["Disk Radius"], key="-DISK_RADIUS-")],
            [sg.Text("Shaft Weight:", size=(15, 1)), sg.InputText(stored_data["Shaft Weight"], key="-SHAFT_WEIGHT-")],
            [sg.Text("Trial Weight 1:", size=(15, 1)), sg.InputText(stored_data["Trial Weight 1"], key="-TRIAL_WEIGHT_1-")],
            [sg.Text("Trial Weight 2:", size=(15, 1)), sg.InputText(stored_data["Trial Weight 2"], key="-TRIAL_WEIGHT_2-")],
            
            [sg.Text("_" * 60, size=(60, 1))],  # Separator
            [sg.Text("Computed Balance Data", font=("Arial", 12, "bold"))],
            [sg.Text("Weight 1:", size=(15, 1)), sg.Text(balance_data["magnitude_Wbal1"], key="-WEIGHT1-")],
            [sg.Text("Angle 1 (°):", size=(15, 1)), sg.Text(balance_data["angle_Wbal1_deg"], key="-ANGLE1-")],
            [sg.Text("Weight 2:", size=(15, 1)), sg.Text(balance_data["magnitude_Wbal2"], key="-WEIGHT2-")],
            [sg.Text("Angle 2 (°):", size=(15, 1)), sg.Text(balance_data["angle_Wbal2_deg"], key="-ANGLE2-")],
            
            # New section to display processed value
            [sg.Text("Processed Amplitude Data:", size=(25, 1)), sg.Text("N/A", key="-PROCESSED-")],

            [sg.Button("Submit", key="-SUBMIT-"), sg.Button("Refresh Data", key="-REFRESH-")]
        ]),
        sg.VSeparator(),     #Separator between columns
        sg.Column(table_layout)
    ]
]

# Segment 2 (Placeholder)
segment2_layout = [
    [sg.Canvas(key="-CANVAS-")],
]

# Define main layout with segment buttons
layout = [
    [sg.Button("Segment 1", key="-SEG1-", size=(10, 1)), sg.Button("Segment 2", key="-SEG2-", size=(10, 1))],
    [sg.Column(segment1_layout, key="-SEGMENT1-", visible=True)],
    [sg.Column(segment2_layout, key="-SEGMENT2-", visible=False)]
]

# Create the window
window = sg.Window("GUI with Two Segments", layout, finalize=True)

# Event loop
while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED:
        break
    elif event == "-SEG1-":
        window["-SEGMENT1-"].update(visible=True)
        window["-SEGMENT2-"].update(visible=False)
    elif event == "-SEG2-":
        window["-SEGMENT1-"].update(visible=False)
        window["-SEGMENT2-"].update(visible=True)
        fig = create_polar_plot()
        draw_polar_plot(window["-CANVAS-"].Widget, fig)

    elif event == "-SUBMIT-":
        disk_radius = values["-DISK_RADIUS-"]
        shaft_weight = values["-SHAFT_WEIGHT-"]
        trial_weight1 = values["-TRIAL_WEIGHT_1-"]
        trial_weight2 = values["-TRIAL_WEIGHT_2-"]

        # Save entered values
        save_data(disk_radius, shaft_weight, trial_weight1, trial_weight2)
        time.sleep(1)

        # Run the balance calculation script
        run_balance_script()
        
        sg.popup(f"Entered Values:\nDisk Radius: {disk_radius}\nShaft Weight: {shaft_weight}\nTrial Weight 1: {trial_weight1}\nTrial Weight 2: {trial_weight2}")
    
    elif event == "-REFRESH-":
        # Wait for `data2.json` to be created
        timeout = 275  # 4.5 minutes max wait time
        while not os.path.exists(DATA_FILE_2) and timeout > 0:
            time.sleep(1)  # Wait 1 second
            timeout -= 1
        if os.path.exists(DATA_FILE_2):
            Unbalance = Unbal_func(balance_data["A1"], balance_data["A11"])
            # Reload balance data and update display
            balance_data = load_balance_data()

            # Update computed balance values
            window["-WEIGHT1-"].update(balance_data["magnitude_Wbal1"])
            window["-ANGLE1-"].update(balance_data["angle_Wbal1_deg"])
            window["-WEIGHT2-"].update(balance_data["magnitude_Wbal2"])
            window["-ANGLE2-"].update(balance_data["angle_Wbal2_deg"])
            window["-PROCESSED-"].update(Unbalance)  # Update processed value

            # **Update the Table Data**
            table_data = [
                [balance_data["a_x1"], balance_data["a_x11"], balance_data["a_x12"]],
                [balance_data["a_x2"], balance_data["a_x21"], balance_data["a_x22"]],
                [balance_data["a_y1"], balance_data["a_y11"], balance_data["a_y12"]],
                [balance_data["a_y2"], balance_data["a_y21"], balance_data["a_y22"]],
            ]
            window["-TABLE-"].update(values=table_data)  # Update table in GUI
            
            sg.popup("✅ Data updated successfully!")
        else:
            sg.popup_error("⚠ Data not available. Calculation might have failed.")

# Close the window
window.close()