import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import linregress
import pykooh
import pandas as pd
import re
from pystrata.site import DarendeliSoilType

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
        ax.set_title(f'Fourier Amplitude Spectra - Kappa Frequency Bounds: {low} - {high} Hz')
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