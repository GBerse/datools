"""
K-NET Data Attribute techer Module

This module provides functionality to extract Kik-NET (Kyoshin Network) ground motions,
extract their metadata properties, and organize them into a structured DataFrame.

Functions include reading fixed-width format files, calculating geographical distances,
and processing entire directories of K-NET data files.
"""

import pandas as pd
from glob import glob
import os
from geopy.distance import geodesic
import re

def calculate_and_format(expression):
    # Use regular expression to extract the numbers
    numbers = re.findall(r'\d+', expression)
    if len(numbers) == 2:
        numerator = int(numbers[0])
        denominator = int(numbers[1])
        result = numerator / denominator
        return f"{result:.8f}"
    else:
        return None

def read_properties(fname):
    # Reading fixed width file (assuming the structure is known and consistent)
    widths = [18, 100]  # Assuming these are the column widths
    nrows = 16  # Number of rows to read for properties
    df = pd.read_fwf(fname, widths=widths, header=None, nrows=nrows)

    base_name, extension = os.path.splitext(os.path.basename(fname))

    # Extract properties from the DataFrame
    properties = {
        'FileName': base_name,
        'Fileextension': extension,
        'OriginTime': df.iloc[0, 1],
        'OriginLat': float(df.iloc[1, 1]),
        'OriginLong': float(df.iloc[2, 1]),
        'Depth': float(df.iloc[3, 1]),
        'Magnitude': float(df.iloc[4, 1]),
        'StationCode': df.iloc[5, 1],
        'StationLat': float(df.iloc[6, 1]),
        'StationLong': float(df.iloc[7, 1]),
        'StationHeight': float(df.iloc[8, 1]),
        'RecordTime': df.iloc[9, 1],
        'SamplingFreq': int(df.iloc[10, 1].replace('Hz', '')),
        'DurationTime': float(df.iloc[11, 1]),
        'Direction': df.iloc[12, 1],
        'ScaleFactor': df.iloc[13, 1],
        'MaxAcc': float(df.iloc[14, 1]),
        'LastCorrection': df.iloc[15, 1],
        'Calibration': calculate_and_format(df.iloc[13, 1])  # Assuming the expression is in this cell
    }

    # Calculate Hypocentral and Epicentral Distances
    origin = (properties['OriginLat'], properties['OriginLong'])
    station = (properties['StationLat'], properties['StationLong'])
    properties['EpicentralDist'] = geodesic(origin, station).kilometers

    #channel = extension.lstrip('.')
    #properties['UniqueID'] = f"{properties['Calibration']}_{properties['MaxAcc']}_{properties['Magnitude']}_{channel}"
    properties['UniqueID'] = f"{properties['FileName']}{properties['Fileextension']}"
    return properties

def turntolist(folder):
    # takes a folder and breaks it up into path directories
    new_flist = [os.path.join(folder, filename) for filename in os.listdir(folder) if
                 filename.endswith(('.EW', '.NS', '.UD', '.EW1', '.NS1', '.UD1', '.EW2', '.NS2', '.UD2'))]
    return new_flist

def KnetDataListing(flist):
    """Reads a K-NET fixed-width format file and extracts its metadata properties.
    
    Parameters:
    ---
        fname (str): Path to the K-NET data file to be processed.
        
    Returns:
    ---
        dict: A dictionary containing all extracted properties from the file including:
            - File information (name, extension)
            - Event information (time, location, magnitude)
            - Station information (code, location, height)
            - Recording information (time, sampling frequency, duration)
            - Calculated values (distances, calibration factor)
            - Unique identifier for the record
            
    The function also calculates:
        - Epicentral distance using geopy's geodesic function
        - Calibration factor from the scale factor expression
    """
    # If the input is a folder
    if os.path.isdir(flist):
        flist = turntolist(flist)

    # Initialize an empty list for storing properties of each file
    prop_list = []

    # Filter file list for specific extensions
    filtered_flist = [f for f in flist if
                      f.endswith(('.EW', '.NS', '.UD', '.EW1', '.NS1', '.UD1', '.EW2', '.NS2', '.UD2'))]

    for fname in filtered_flist:
        properties = read_properties(fname)
        prop_list.append(properties)

    # Convert list of dictionaries to DataFrame
    properties_df = pd.DataFrame(prop_list)

    return properties_df

# Example usage:
# flist = ['path_to_file1.EW', 'path_to_file2.NS']
# properties_df = KnetDataListing(flist)
# print(properties_df)