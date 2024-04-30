import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the Excel file
filename = input("Enter the name of the Excel file (without extension): ") + ".xlsx"
gaze_df = pd.read_excel(filename)
saccades = []
fixations = []
# Convert timestamp column to datetime objects
gaze_df['Timestamp'] = pd.to_datetime(gaze_df['Timestamp'])
fixation_time = 0
fixation_started = False
saccade_time =0 
saccade_started = False
# Initialize previous coordinates
prev_x, prev_y = None, None

# Calculate and print the duration and distance between consecutive frames
prev_timestamp = None
for index, row in gaze_df.iterrows():
    timestamp = row['Timestamp']
    x, y = row['X'], row['Y']
    milliseconds = 0
    distance = 0
    # Calculate duration
    if prev_timestamp is not None:
        duration = timestamp - prev_timestamp
        total_seconds = duration.total_seconds()
        milliseconds = int((total_seconds) * 1000)
        #print("Duration between frame", index-1, "and frame", index, ":", f"{milliseconds} milliseconds")
    
    # Calculate distance
    if prev_x is not None and prev_y is not None:
        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        if distance > 20:
            print("g 20")
            print(milliseconds)
            print(distance)
            if(fixation_started == True):
                fixations.append(fixation_time)
                fixation_time = 0
                fixation_started = False
            if(saccade_started == False):
                saccade_started = True
            saccade_time += milliseconds
        else:
            print("l 20")
            print(milliseconds)
            print(distance)

            if(fixation_started == False):
                fixation_started = True
            fixation_time += milliseconds
            if(saccade_started == True):
                saccades.append(saccade_time)
                saccade_started = False
                saccade_time = 0
        #print("Distance between frame", index-1, "and frame", index, ":", distance)
    # Update previous values
    prev_timestamp = timestamp
    prev_x, prev_y = x, y



print(fixations)
print(saccades)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(fixations, bins=100, color='blue', alpha=0.7)
plt.title('Fixations')
plt.xlabel('Duration (milliseconds)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(saccades, bins=100, color='red', alpha=0.7)
plt.title('Saccades')
plt.xlabel('Duration (milliseconds)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
#[8, 6669, 4446, 624, 10, 12, -16, 1018, 203, 232]
#[44, 46, 174, 202, 210, 195, 208, 195, 207, 197, 203, 206, 199, 206, 200, 199, 203, 205, 198, 210, 199, 201, 201, 231, 201, 201, 206, 198, 195, 283, 156, 202, 196, 214, 196, 206, 195, 204, 185, 205]