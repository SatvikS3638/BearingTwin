import adsk.core, adsk.fusion, adsk.cam, traceback
import threading
import os
import csv
import time
import math
#THIS SCRIPT IS FOR FUSION 360 ADD-IN
app = adsk.core.Application.get()
ui = app.userInterface

running = False
timer = None
row = 1
H = 0
status = "Good"
previous_stat = "Good"

# Function to read health index H from CSV
def read_health(current_row):
    try:
        with open(r"D:\Digital_twin_final\dashboard\all_features_copied_from_pi.csv", 'r') as file:
            csvreader = csv.reader(file)
            rows = list(csvreader)
            
            # Wait until enough data is available
            while len(rows) <= current_row:
                print("Waiting for enough data...")
                time.sleep(1)
                file.seek(0)
                rows = list(csv.reader(file))

            H = float(rows[current_row][15])  # Column 15 is the H value
            return H

    except Exception as e:
        ui.messageBox('Error in file read:\n{}'.format(traceback.format_exc()))
        return None


# Change appearance based on status using appearances already in the document
def update_appearance():
    global status, previous_stat
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        root_comp = design.rootComponent

        bearing_occ = root_comp.occurrences.itemByName("Bearing:1")
        if not bearing_occ:
            ui.messageBox("Component 'Bearing' not found")
            return

        body1 = bearing_occ.bRepBodies.item(0)
        body2 = bearing_occ.bRepBodies.item(1)

        if status in ['Good'] and H > 0.03 and H < 0.1:
            status = "Slight"
        elif status in ['Slight'] and H > 0.1 and H < 0.25:
            status = "Warning"
        elif status in ['Warning'] and H > 0.25 and H < 0.55:
            status = "Measures"
        elif status in ['Measures'] and H > 0.55 and H < 0.8:
            status = "Risky"
        elif status in ['Risky'] and H > 0.8:
            status = "Failure"

        appearance_names = {
            'Good': 'Powder Coat (Green)',
            'Slight':'Powder Coat (Mid-Green)',
            'Warning': 'Powder Coat (Yellow)',
            'Precaution': 'Powder Coat (Orange)',
            'Risky': 'Powder Coat (Orange-Red)',
            'Failure': 'Powder Coat (Red)'
        }

        appearance_name = appearance_names.get(status)
        if not appearance_name:
            ui.messageBox(f"No appearance defined for status: {status}")
            return

        appearance = design.appearances.itemByName(appearance_name)

        if not appearance:
            ui.messageBox(f"Appearance '{appearance_name}' not found in document")
            return

        body1.appearance = appearance
        body2.appearance = appearance

    except:
        ui.messageBox('Error in update_appearance:\n{}'.format(traceback.format_exc()))


# Scale the health bar and apply appearance

def update_health_bar():
    global row, H, status, previous_stat
    app = adsk.core.Application.get()
    ui = app.userInterface
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent

    try:
        appearance_names = {
            'Good': 'Powder Coat (Green)',
            'Slight':'Powder Coat (Mid-Green)',
            'Warning': 'Powder Coat (Yellow)',
            'Precaution': 'Powder Coat (Orange)',
            'Risky': 'Powder Coat (Orange-Red)',
            'Failure': 'Powder Coat (Red)'
        }

        bearing_occ = root.occurrences.itemByName("Statusbar:1")
        if not bearing_occ:
            ui.messageBox("Occurrence 'Statusbar:1' not found.")
            return

        body = bearing_occ.bRepBodies.item(0)

        sketch_name = "Sketch3"
        sketch = next((s for s in root.sketches if s.name == sketch_name), None)
        if not sketch:
            ui.messageBox(f"Sketch '{sketch_name}' not found.")
            return

        basePt = sketch.sketchPoints.item(0)

        scale_map = {
            'Good': 1,
            'Slight':0.8,
            'Warning': 0.65,
            'Precaution': 0.45,
            'Risky': 0.25,
            'Failure': 0.15
        }

        x_scale_val = scale_map.get(status, 1.0)
        bbox = body.boundingBox
        x_length = bbox.maxPoint.x - bbox.minPoint.x
        x_scale_val = (x_scale_val * 10) / x_length

        input_coll = adsk.core.ObjectCollection.create()
        input_coll.add(body)

        scales = root.features.scaleFeatures
        scaleInput = scales.createInput(input_coll, basePt, adsk.core.ValueInput.createByReal(1))

        xScale = adsk.core.ValueInput.createByReal(x_scale_val)
        yScale = adsk.core.ValueInput.createByReal(1)
        zScale = adsk.core.ValueInput.createByReal(1)
        scaleInput.setToNonUniform(xScale, yScale, zScale)

        if row == 1:
            scales.add(scaleInput)
        else:
            if status != previous_stat:
                scales.add(scaleInput)
                previous_stat = status

        appearance_name = appearance_names.get(status)
        if appearance_name:
            appearance = design.appearances.itemByName(appearance_name)
            if appearance:
                body.appearance = appearance
            else:
                ui.messageBox(f"Appearance '{appearance_name}' not found in document.")

        H_value = read_health(row)
        if H_value is not None:
            H = H_value
            row += 1

    except Exception as e:
        ui.messageBox('Failed in update_health_bar:\n{}'.format(traceback.format_exc()))


def simple_timer():
    global timer, running
    if not running:
        return
    update_appearance()
    update_health_bar()
    timer = threading.Timer(0.1, simple_timer)
    timer.start()


def run(context):
    try:
        global running
        running = True
        ui.messageBox("BearingTwin Add-In Started")
        simple_timer()
    except Exception as e:
        ui.messageBox(f"Failed:\n{traceback.format_exc()}")


def stop(context):
    try:
        global timer, running
        running = False
        if timer:
            timer.cancel()
        ui.messageBox("BearingTwin Add-In Stopped")
    except Exception as e:
        ui.messageBox(f"Failed to stop:\n{traceback.format_exc()}")