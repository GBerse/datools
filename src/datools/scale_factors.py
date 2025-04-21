import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import linregress
import pykooh
import pandas as pd
import re
from pystrata.site import DarendeliSoilType
import pystrata
from scipy.integrate import cumulative_trapezoid
import os
from obspy import read
import obspy
from datools import site_response_small

def plot_FAS(trace,ax,ko=None):
    # Compute Fourier Amplitude Spectrum
    fourier_amps = trace.stats["delta"] * np.abs(np.fft.rfft(trace.data))
    freqs = np.fft.rfftfreq(len(trace.data), d=trace.stats["delta"])

    if ko is not None:
            fourier_amps = pykooh.smooth(freqs, freqs, fourier_amps, ko)
    
    ax.plot(freqs, fourier_amps,color='k')
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_title('Fourier Amplitude Spectra')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('FAS', fontsize=12)
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.25)
    ax.set_xlim(0.1, 100)
    #ax.set_ylim(0.000001, .01)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)

def calculate_kappa_and_plot(trace, ax, low_freq, high_freq, norm_freq=None, color='grey', label=None, color_2='blue', ko=40):
    # Compute Fourier Amplitude Spectrum
    fourier_amps = trace.stats["delta"] * np.abs(np.fft.rfft(trace.data))
    freqs = np.fft.rfftfreq(len(trace.data), d=trace.stats["delta"])

    # Smooth the spectrum
    smoothed_amps = pykooh.smooth(freqs, freqs, fourier_amps, ko)
    
    # Normalize (if wanted)
    if norm_freq is not None:
        fas_at_norm_freq = np.interp(norm_freq, freqs, smoothed_amps)
        amps = smoothed_amps / fas_at_norm_freq
    else:
        amps = smoothed_amps
    # Plot normalized spectrum
    ax.plot(freqs, amps, color=color, label=label)
    
    freq_min = low_freq
    freq_max = high_freq
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    
    if np.any(mask):  # Proceed only if the mask has valid points
        freqs_log = freqs[mask]
        amps_log = np.log(amps[mask])
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(freqs_log, amps_log)
        
        # Calculate kappa (negative slope divided by π)
        kappa = -slope / np.pi
        
        # Plot the fitted line for visual confirmation
        fitted_line = np.exp(intercept + slope * freqs_log)
        ax.plot(freqs_log, fitted_line, color=color_2, linestyle='--', label=f'Kappa: {kappa:.4f}')

        if norm_freq is not None:
            ax.axvline(x=norm_freq, label=f'{norm_freq}Hz', linestyle='--', color='red')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.set_title(f'Fourier Amplitude Spectra - Kappa Frequency Bounds: {low_freq} - {high_freq} Hz')
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('FAS', fontsize=12)
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.grid(which='both', color='gray', linestyle='-', linewidth=0.25)
        ax.set_xlim(0.1, 40)
        ax.set_ylim(0.0001, 100)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)



        
        return kappa
    else:
        return None

def preprocess_df(df):
    """
    Improved filename parser that handles:
    - Format: TCGH160311150344EW1.MSEED
    - Format: something.NS1
    - And other common seismic naming conventions
    """
    # First try to extract with regex for formats like TCGH160311150344EW1.MSEED
    pattern = r'^(.*?)([EN][SW][12]|Z[12]|UD[12])(?:\..*)?$'
    matches = df['Filename'].str.extract(pattern)
    
    # For matched cases
    df['BaseFilename'] = matches[0].where(~matches[0].isna(), df['Filename'].str.split('.').str[0])
    df['Component'] = matches[1].where(~matches[1].isna(), df['Filename'].str.split('.').str[1])
    
    # Clean up any remaining NaN values
    df['BaseFilename'] = df['BaseFilename'].fillna(df['Filename'])
    df['Component'] = df['Component'].fillna('UNKNOWN')
    
    return df

def generate_kappa_df(base, surf, drop_na=True):

    """
    Generate kappa comparison DataFrame with option to keep/drop rows with missing values
    
    Parameters:
    base (list): List of (kappa, filename) tuples for base motion
    surf (list): List of (kappa, filename) tuples for surface motion
    drop_na (bool): Whether to drop rows with missing values (default: True)
    """
    # Convert to DataFrames
    df_surf = pd.DataFrame(surf, columns=['Kappa', 'Filename'])
    df_base = pd.DataFrame(base, columns=['Kappa', 'Filename'])
    
    # Preprocess both dataframes
    df_base = preprocess_df(df_base)
    df_surf = preprocess_df(df_surf)

    # Combine dataframes
    df_all = pd.concat([df_base, df_surf], ignore_index=True)

    # Find all available components
    available_components = df_all['Component'].unique()
    
    # Find matching pairs (more flexible approach)
    valid_filenames = set()
    for component_pair in [('NS1', 'NS2'), ('EW1', 'EW2'), ('UD1', 'UD2')]:
        if all(c in available_components for c in component_pair):
            comp1_files = set(df_all[df_all['Component'] == component_pair[0]]['BaseFilename'])
            comp2_files = set(df_all[df_all['Component'] == component_pair[1]]['BaseFilename'])
            valid_filenames.update(comp1_files & comp2_files)

    # Filter for valid filenames
    if not valid_filenames:
        raise ValueError("No matching component pairs found in the data")
    
    df_matched = df_all[df_all['BaseFilename'].isin(valid_filenames)]

    # Pivot with all found components
    df_pivot = df_matched.pivot(index='BaseFilename', columns='Component', values='Kappa').reset_index()
    
    # Clean up
    df_pivot.columns.name = None

    # Calculate deltas for available pairs
    for comp1, comp2 in [('EW1', 'EW2'), ('NS1', 'NS2'), ('UD1', 'UD2')]:
        if comp1 in df_pivot.columns and comp2 in df_pivot.columns:
            df_pivot[f'{comp1[:2]}_delta'] = (df_pivot[comp2] - df_pivot[comp1]).abs()

    # Conditionally drop NA rows
    if drop_na:
        # Only drop rows where all delta columns are NA
        delta_cols = [col for col in df_pivot.columns if '_delta' in col]
        if delta_cols:
            df_pivot = df_pivot.dropna(subset=delta_cols, how='all')

    return df_pivot

def calc_damp_min(row):
    darendeli_soil = DarendeliSoilType(
        unit_wt=row['Unit_Weight'],
        plas_index=row['PI'],
        ocr=row['OCR'],
        stress_mean=row['Mean_Stress']
    )
    return darendeli_soil.damping_min

def calculate_sf(Vs, H, Dmin, kappa_delta):

    """
    Calculate the scaling factor (SF) based on Vs, H, Dmin, and kappa_delta.

    Parameters:
    Vs (list of floats): Shear wave velocities
    H (list of floats): Thicknesses
    Dmin (list of floats): Minimum distances (in percent)
    kappa_delta (float): Given delta kappa value

    Returns:
    float: The scaling factor (SF)
    """

    # Calculate the sum part of the equation with adjusted Dmin
    sum_value = sum(2 * H[i] * Dmin[i] / Vs[i] for i in range(len(Vs)))

    # Calculate the scaling factor (SF) with adjusted Dmin
    SF = kappa_delta / sum_value

    return SF


import pandas as pd
import obspy
from obspy import Stream
from copy import deepcopy
import numpy as np
def create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, df=None):
    """
    Create a soil profile using pystrata based on specified parameters.

    Parameters:
    uw : float
        Unit weight of soil in kN/m³.
    PI : list
        Plasticity index for each layer.
    OCR : list
        Over consolidation ratio for each layer.
    mean_stress : list
        Mean effective stress for each layer in kPa.
    thick : list
        Thickness of each layer in meters.
    vs : list
        Shear wave velocity for each layer in m/s.
    sf : float
        Scaling factor for damping.
    df : DataFrame, optional
        Data frame containing damping minimum values for each layer.

    Returns:
    pystrata.site.Profile
        Constructed soil profile.
    """
    layers = []
    num_layers = len(PI)  # Assuming all lists have the same length

    for i in range(num_layers):
        # Create kwargs dictionary for optional damping_min parameter
        kwargs = {
            'unit_wt': uw,
            'plas_index': PI[i],
            'ocr': OCR[i],
            'stress_mean': mean_stress[i]
        }
        
        # Add damping_min if DataFrame is provided
        if df is not None:
            kwargs['damping_min'] = sf * df['damp_min'].iloc[i]
        
        # Create soil type
        soil_type = pystrata.site.DarendeliSoilType(**kwargs)
        
        # Create layer and add to list
        layer = pystrata.site.Layer(soil_type, thick[i], vs[i])
        layers.append(layer)
    
    # Append a base layer (half-space) using the properties of the last layer
    base_soil_type = pystrata.site.SoilType("Soil", uw, None, 0.01)
    base_layer = pystrata.site.Layer(base_soil_type, 0, vs[-1])
    layers.append(base_layer)

    profile = pystrata.site.Profile(layers)
    return profile



def generate_identifier(trace):
    unique_identifier = trace.stats.filename
    return unique_identifier

def log_pga(stream):
    ids = pd.DataFrame(columns=['UniqueID', f'pga'])
    for trace in stream:
        id = generate_identifier(trace)
        pga = np.max(np.abs(trace.data))
        new_row = pd.DataFrame({'UniqueID': [id], f'pga': [pga]})
        ids = pd.concat([ids, new_row], ignore_index=True)
    return ids

def log_pgv(stream):
    stream = calc_vels(stream)
    ids = pd.DataFrame(columns=['UniqueID', f'pgv'])
    for trace in stream:
        id = generate_identifier(trace)
        pgv = np.max(np.abs(trace.data))
        new_row = pd.DataFrame({'UniqueID': [id], f'pgv': [pgv]})
        ids = pd.concat([ids, new_row], ignore_index=True)
    return ids


def log_Ia(stream):
    ia_table = pd.DataFrame(columns=['UniqueID', 'AriasIntensity'])
    for trace in stream:
        id = generate_identifier(trace) 
        dt = trace.stats.delta  
        squared_acc = trace.data ** 2  
        integral = np.trapz(squared_acc, dx=dt)  
        arias_intensity =9.81 * (np.pi / (2)) * integral 
        new_row = pd.DataFrame({'UniqueID': [id], 'AriasIntensity': [arias_intensity]})
        ia_table = pd.concat([ia_table, new_row], ignore_index=True)

    return ia_table



def calc_vels(input_stream):
    input_stream_copy = deepcopy(input_stream)
    vels = Stream()
    for trace in input_stream_copy:
        trace.data = trace.data * 981
        vel = trace.integrate()
        vels.append(vel)
    return vels


def consolidate_parameters(stream):
    pga_df = log_pga(stream)
    pgv_df = log_pgv(stream)
    ia_df = log_Ia(stream)
    merged_df = pga_df.merge(pgv_df, on='UniqueID').merge(ia_df, on='UniqueID')
    parts = merged_df['UniqueID'].str.split('.', expand=True)  # Split into two columns
    name = parts[0]  # Get the part before the dot
    merged_df['adjusted_filename'] = name.str[:-1]

    return merged_df


def intermediate_search_linear_pga(initial_sf, emp_pga, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.1, num_steps=10):
    """
    Search leftward from initial scaling factor, finding the nearest ratio under 1.
    Used for intermediate searches where we want to approach from below.
    """
    best_sf = initial_sf
    best_ratio = 0
    best_pga = None
    
    # Search leftward, looking for highest ratio that's still under 1
    for i in range(num_steps):
        sf = initial_sf - i * step_size
        profile = create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, dmin_df)

        calc = pystrata.propagation.LinearElasticCalculator()
        outputs = pystrata.output.OutputCollection(
            [pystrata.output.AccelerationTSOutput(
                pystrata.output.OutputLocation("outcrop", index=0)
            )]
        )
        calc(m, profile, profile.location("within", index=-1))
        outputs(calc)

        accel_output = outputs[0]
        pga = np.nanmax(abs(accel_output.values))
        ratio = pga / emp_pga

        if ratio >= 1:
            continue  # Skip ratios >= 1
            
        if ratio > best_ratio:  # Keep highest ratio under 1
            best_sf = sf
            best_ratio = ratio
            best_pga = pga

    return best_sf, best_ratio, best_pga

def final_search_linear_pga(initial_sf, emp_pga, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.01, num_steps=10):
    """
    Final search that finds the scaling factor giving ratio closest to 1.
    """
    best_sf = initial_sf
    best_ratio = float('inf')
    best_pga = None
    
    for i in range(num_steps):
        sf = initial_sf - i * step_size
        profile = create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, dmin_df)

        calc = pystrata.propagation.LinearElasticCalculator()
        outputs = pystrata.output.OutputCollection(
            [pystrata.output.AccelerationTSOutput(
                pystrata.output.OutputLocation("outcrop", index=0)
            )]
        )
        calc(m, profile, profile.location("within", index=-1))
        outputs(calc)

        accel_output = outputs[0]
        pga = np.nanmax(abs(accel_output.values))
        ratio = pga / emp_pga

        if abs(ratio - 1) < abs(best_ratio - 1):  # Keep closest to 1
            best_sf = sf
            best_ratio = ratio
            best_pga = pga

    return best_sf, best_ratio, best_pga

def intermediate_search_linear_pgv(initial_sf, emp_pgv, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.1, num_steps=10):
    """
    Search leftward from initial scaling factor, finding the nearest ratio under 1 for PGV.
    """
    best_sf = initial_sf
    best_ratio = 0
    best_pgv = None
    
    for i in range(num_steps):
        sf = initial_sf - i * step_size
        profile = create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, dmin_df)

        calc = pystrata.propagation.LinearElasticCalculator()
        outputs = pystrata.output.OutputCollection(
            [pystrata.output.AccelerationTSOutput(
                pystrata.output.OutputLocation("outcrop", index=0)
            )]
        )
        calc(m, profile, profile.location("within", index=-1))
        outputs(calc)

        accel_output = outputs[0]
        acceleration = accel_output.values
        times = accel_output.times
        velocity = cumulative_trapezoid(acceleration * 981, times, initial=0)  # cm/s
        pgv = np.nanmax(abs(velocity))
        ratio = pgv / emp_pgv

        if ratio >= 1:
            continue
            
        if ratio > best_ratio:
            best_sf = sf
            best_ratio = ratio
            best_pgv = pgv

    return best_sf, best_ratio, best_pgv

def final_search_linear_pgv(initial_sf, emp_pgv, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.01, num_steps=10):
    """
    Final search that finds the scaling factor giving PGV ratio closest to 1.
    """
    best_sf = initial_sf
    best_ratio = float('inf')
    best_pgv = None
    
    for i in range(num_steps):
        sf = initial_sf - i * step_size
        profile = create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, dmin_df)

        calc = pystrata.propagation.LinearElasticCalculator()
        outputs = pystrata.output.OutputCollection(
            [pystrata.output.AccelerationTSOutput(
                pystrata.output.OutputLocation("outcrop", index=0)
            )]
        )
        calc(m, profile, profile.location("within", index=-1))
        outputs(calc)

        accel_output = outputs[0]
        acceleration = accel_output.values
        times = accel_output.times
        velocity = cumulative_trapezoid(acceleration * 981, times, initial=0)  # cm/s
        pgv = np.nanmax(abs(velocity))
        ratio = pgv / emp_pgv

        if abs(ratio - 1) < abs(best_ratio - 1):
            best_sf = sf
            best_ratio = ratio
            best_pgv = pgv

    return best_sf, best_ratio, best_pgv

def intermediate_search_linear_ia(initial_sf, emp_ia, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.1, num_steps=10):
    """
    Search leftward from initial scaling factor, finding the nearest ratio under 1 for Ia.
    """
    best_sf = initial_sf
    best_ratio = 0
    best_ia = None
    
    for i in range(num_steps):
        sf = initial_sf - i * step_size
        profile = create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, dmin_df)

        calc = pystrata.propagation.LinearElasticCalculator()
        outputs = pystrata.output.OutputCollection(
            [
                pystrata.output.AriasIntensityTSOutput(
                    pystrata.output.OutputLocation("outcrop", index=0)
                ),
            ]
        )
        calc(m, profile, profile.location("within", index=-1))
        outputs(calc)

        ia_values = outputs[0]
        max_ia = np.nanmax(ia_values.values) #in m/s
        ratio = max_ia / emp_ia

        if ratio >= 1:
            continue
            
        if ratio > best_ratio:
            best_sf = sf
            best_ratio = ratio
            best_ia = max_ia

    return best_sf, best_ratio, best_ia

def final_search_linear_ia(initial_sf, emp_ia, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.01, num_steps=10):

    """
    Final search that finds the scaling factor giving Ia ratio closest to 1.
    """
    best_sf = initial_sf
    best_ratio = float('inf')
    best_ia = None
    
    for i in range(num_steps):
        sf = initial_sf - i * step_size
        profile = create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, dmin_df)

        calc = pystrata.propagation.LinearElasticCalculator()
        outputs = pystrata.output.OutputCollection(
            [
                pystrata.output.AriasIntensityTSOutput(
                    pystrata.output.OutputLocation("outcrop", index=0)
                ),
            ]
        )
        calc(m, profile, profile.location("within", index=-1))
        outputs(calc)

        ia_values = outputs[0]
        max_ia = np.nanmax(ia_values.values) #in m/s
        ratio = max_ia / emp_ia

        if abs(ratio - 1) < abs(best_ratio - 1):
            best_sf = sf
            best_ratio = ratio
            best_ia = max_ia
            
    return best_sf, best_ratio, best_ia


def pga_gmp_sf(motions,profile_df,emp_attr,sf_list,unit_weight=20):
    uw=unit_weight
    PI=profile_df['PI'].to_numpy()
    OCR=profile_df['OCR'].to_numpy()
    mean_stress=profile_df['Mean_Stress'].to_numpy()
    thick=profile_df['Thickness'].to_numpy()
    vs=profile_df['Vs'].to_numpy()
    dmin_df=pd.DataFrame(profile_df['damp_min'])
    sf = 1

    calc_pga_df = pd.DataFrame(columns=['UniqueID', 'pga', 'pga_ratio', 'pga_SF','emp_pga'])
    for m in motions:
        ratio =  float('inf')
        ii =0
        """
        Finding the respective empirical values for the motion
        """
        file_name = m.filename
        part = file_name.split('.')[0]  # Get the part before the dot
        adjusted_filename = part[:-1]

        emp_pga = emp_attr.loc[emp_attr['adjusted_filename'] == adjusted_filename]['pga'].iloc[0]
        
        while ratio > 1 and ii < len(sf_list):
            ii += 1
            sf = sf_list[ii-1]
            profile = site_response_small.create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, dmin_df)

            calc = pystrata.propagation.LinearElasticCalculator()
            outputs = pystrata.output.OutputCollection(
                [
                    pystrata.output.AccelerationTSOutput(
                        pystrata.output.OutputLocation("outcrop", index=0)
                    ),
                ]
            )
            calc(m, profile, profile.location("within", index=-1))
            outputs(calc)


            accel_output = outputs[0] 
            pga = np.nanmax(abs(accel_output.values))  # in g (1g = 9.81m/s²)
            ratio = pga / emp_pga

        initial_sf = sf
        next_sf, next_ratio, next_pga = intermediate_search_linear_pga(initial_sf, emp_pga, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.1, num_steps=10)
        best_sf, best_ratio, best_pga = final_search_linear_pga(next_sf, emp_pga, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.01, num_steps=10)


        new_row = pd.DataFrame({'UniqueID': [m.filename], 'pga': [best_pga], 'pga_ratio': [best_ratio],'pga_SF': [best_sf], 'emp_pga': [emp_pga]})
        calc_pga_df = pd.concat([calc_pga_df, new_row], ignore_index=True)
    return calc_pga_df

def pgv_gmp_sf(motions,profile_df,emp_attr,sf_list,unit_weight=20):
    uw=unit_weight
    PI=profile_df['PI'].to_numpy()
    OCR=profile_df['OCR'].to_numpy()
    mean_stress=profile_df['Mean_Stress'].to_numpy()
    thick=profile_df['Thickness'].to_numpy()
    vs=profile_df['Vs'].to_numpy()
    dmin_df=pd.DataFrame(profile_df['damp_min'])
    sf = 1

    calc_pgv_df = pd.DataFrame(columns=['UniqueID', 'pgv', 'pgv_ratio', 'pgv_SF','emp_pgv'])
    for m in motions:
        ratio =  float('inf')
        ii =0
        """
        Finding the respective empirical values for the motion
        """
        file_name = m.filename
        part = file_name.split('.')[0]  # Get the part before the dot
        adjusted_filename = part[:-1]

        emp_pgv = emp_attr.loc[emp_attr['adjusted_filename'] == adjusted_filename]['pgv'].iloc[0]
        
        while ratio > 1 and ii < len(sf_list):
            ii += 1
            sf = sf_list[ii-1]
            profile = create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, dmin_df)

            calc = pystrata.propagation.LinearElasticCalculator()
            outputs = pystrata.output.OutputCollection(
                [
                    pystrata.output.AccelerationTSOutput(
                        pystrata.output.OutputLocation("outcrop", index=0)
                    ),
                ]
            )
            calc(m, profile, profile.location("within", index=-1))
            outputs(calc)
            
            accel_output = outputs[0]
            acceleration = accel_output.values
            times = accel_output.times
            velocity = cumulative_trapezoid(acceleration * 981, times, initial=0)  #in cm/s
            pgv = np.nanmax(abs(velocity))

            ratio = pgv / emp_pgv

        initial_sf = sf
        next_sf, next_ratio, next_pgv = intermediate_search_linear_pgv(initial_sf, emp_pgv, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.1, num_steps=10)
        best_sf, best_ratio, best_pgv = final_search_linear_pgv(next_sf, emp_pgv, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.01, num_steps=10)


        new_row = pd.DataFrame({'UniqueID': [m.filename], 'pgv': [best_pgv], 'pgv_ratio': [best_ratio],'pgv_SF': [best_sf], 'emp_pgv': [emp_pgv]})
        calc_pgv_df = pd.concat([calc_pgv_df, new_row], ignore_index=True)

    return calc_pgv_df

def ia_gmp_sf(motions,profile_df,emp_attr,sf_list,unit_weight=20):
    uw=unit_weight
    PI=profile_df['PI'].to_numpy()
    OCR=profile_df['OCR'].to_numpy()
    mean_stress=profile_df['Mean_Stress'].to_numpy()
    thick=profile_df['Thickness'].to_numpy()
    vs=profile_df['Vs'].to_numpy()
    dmin_df=pd.DataFrame(profile_df['damp_min'])
    sf = 1

    calc_ia_df = pd.DataFrame(columns=['UniqueID', 'ia', 'ia_ratio', 'ia_SF','emp_ia'])
    for m in motions:
        ratio =  float('inf')
        ii =0
        """
        Finding the respective empirical values for the motion
        """
        file_name = m.filename
        part = file_name.split('.')[0]  # Get the part before the dot
        adjusted_filename = part[:-1]

        emp_ia = emp_attr.loc[emp_attr['adjusted_filename'] == adjusted_filename]['AriasIntensity'].iloc[0] 
        while ratio > 1 and ii < len(sf_list):
            ii += 1
            sf = sf_list[ii-1]
            profile = create_soil_profile(uw, PI, OCR, mean_stress, thick, vs, sf, dmin_df)

            calc = pystrata.propagation.LinearElasticCalculator()
            outputs = pystrata.output.OutputCollection(
                [
                    pystrata.output.AriasIntensityTSOutput(
                        pystrata.output.OutputLocation("outcrop", index=0)
                    ),
                ]
            )
            calc(m, profile, profile.location("within", index=-1))
            outputs(calc)
            
            ia_values = outputs[0]
            max_ia = np.nanmax(ia_values.values) #in m/s
            ratio = max_ia / emp_ia


        initial_sf = sf
        next_sf, next_ratio, next_ia = intermediate_search_linear_ia(initial_sf, emp_ia, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.1, num_steps=10)
        best_sf, best_ratio, best_ia = final_search_linear_ia(next_sf, emp_ia, m, emp_attr, uw, PI, OCR, mean_stress, thick, vs, dmin_df, step_size=0.01, num_steps=10)


        new_row = pd.DataFrame({'UniqueID': [m.filename], 'ia': [best_ia], 'ia_ratio': [best_ratio],'ia_SF': [best_sf], 'emp_ia': [emp_ia]})
        calc_ia_df = pd.concat([calc_ia_df, new_row], ignore_index=True)

    return calc_ia_df