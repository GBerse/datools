import numpy as np
import pystrata.motion
import pystrata.propagation
import pystrata.output
import pystrata.site
import matplotlib.pyplot as plt
import os
from obspy import read, Stream

import pandas as pd
import re



# ------------------
# Building profiles and motions
# ------------------

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

def parse_site_data(file_path):
    """
    Parses site data from a text file and returns a dictionary of DataFrames.
    
    Args:
        file_path (str): Path to the text file containing site data
        
    Returns:
        dict: A dictionary where keys are site names and values are DataFrames
              containing the site's geotechnical data
    """
    # Read the content of the file
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the file into sections for each site
    sites = re.split(r'\n\n+', data.strip())
    
    site_dataframes = {}
    
    for site in sites:
        lines = site.splitlines()
        if not lines:
            continue
            
        site_name = lines[0].strip()
        try:
            # Extract values
            vs = eval(re.search(r'vs\s*=\s*(\[.*?\])', site).group(1))
            ocr = eval(re.search(r'OCR\s*=\s*(\[.*?\])', site).group(1))
            pi = eval(re.search(r'PI\s*=\s*(\[.*?\])', site).group(1))
            unit_weight = float(re.search(r'uw\s*=\s*(\d+\.\d+)', site).group(1))
            
            # Adjusted parsing for `thick`
            thick_match = re.search(r'thick\s*=\s*(\[.*?\])', site)
            if thick_match:
                thickness = eval(thick_match.group(1))
            else:
                # Handle cases where brackets are not properly closed
                thickness = eval(re.search(r'thick\s*=\s*(\[.*)', site).group(1) + ']')
            
            mean_stress = eval(re.search(r'mean_stress\s*=\s*(\[.*?\])', site).group(1))
            
            # Create the DataFrame
            df = pd.DataFrame({
                'Layer': range(1, len(vs) + 1),
                'Vs': vs,
                'Mean_Stress': mean_stress,
                'OCR': ocr,
                'PI': pi,
                'Unit_Weight': [unit_weight] * len(vs),
                'Thickness': thickness
            })
            
            # Store the DataFrame in the dictionary
            site_dataframes[site_name] = df
            
        except Exception as e:
            print(f"Error processing site {site_name}: {e}")
    
    return site_dataframes

    """
    Example 
    ------------
    site_data = parse_site_data('combined_site_data_2025.txt')

    site1_df = site_data['Site1']  # Replace 'Site1' with actual site name
    print(site1_df.head())
    """

def extract_base_filename(filename):
    # Extract the base part of the filename before the component
    return filename.split('.')[0]

def print_layer_properties(profile):
    """Print G, G*, and damping ratio for each layer."""
    for i, layer in enumerate(profile):
        # Get complex shear modulus (may be frequency-dependent)
        G_star = layer.comp_shear_mod  # This is what the solver uses
        
        # Extract real (G) and imaginary (loss modulus) components
        G_real = np.real(G_star)
        G_imag = np.imag(G_star)
        
        # Compute damping ratio (ξ)
        damping_ratio = G_imag / (2 * G_real)
        
        print(
            f"Layer {i} (Thickness: {layer.thickness:.1f} m):\n"
            f"  - Small-strain G₀: {layer.shear_mod:.2f} Pa\n"
            f"  - Complex G*: {G_real:.2f} + {G_imag:.2f}j Pa\n"
            f"  - Damping ratio (ξ): {damping_ratio:.2%}\n"
        )

# Function to combine traces for a given site, ensuring only EW1 matches with EW2 and NS1 with NS2
def load_site_motions(path,sc,type):
    if type not in ['large', 'small']:
        raise ValueError("type must be either 'large' or 'small'")
    
    path = rf'{path}/{sc}/{type}_strain/'

    
    ew_1, ew_2, ns_1, ns_2 = Stream(), Stream(), Stream(), Stream()
    
    # Get the list of base names from EW1 and EW2, and NS1 and NS2 for matching
    ew1_files = set([extract_base_filename(f) for f in os.listdir(rf'{path}EW1') if f.endswith('.MSEED')])
    ew2_files = set([extract_base_filename(f) for f in os.listdir(rf'{path}EW2') if f.endswith('.MSEED')])

    ns1_files = set([extract_base_filename(f) for f in os.listdir(rf'{path}NS1') if f.endswith('.MSEED')])
    ns2_files = set([extract_base_filename(f) for f in os.listdir(rf'{path}NS2') if f.endswith('.MSEED')])


    # Process EW1, only if a counterpart exists in EW2
    comp_path_ew1 = rf'{path}EW1'
    for filename in os.listdir(comp_path_ew1):
        if filename.endswith('.MSEED'):
            base_filename = extract_base_filename(filename)
            if base_filename in ew2_files:  # Match based on the base filename
                file_path = os.path.join(comp_path_ew1, filename)
                current_stream = read(file_path)
                for trace in current_stream:
                    trace.stats.filename = base_filename
                ew_1 += current_stream
    print(f"Total EW1 traces for {sc} combined: {len(ew_1)}")
    
    # Process EW2, only if a counterpart exists in EW1
    comp_path_ew2 = rf'{path}EW2'
    for filename in os.listdir(comp_path_ew2):
        if filename.endswith('.MSEED'):
            base_filename = extract_base_filename(filename)
            if base_filename in ew1_files:  # Match based on the base filename
                file_path = os.path.join(comp_path_ew2, filename)
                current_stream = read(file_path)
                for trace in current_stream:
                    trace.stats.filename = base_filename
                ew_2 += current_stream
    print(f"Total EW2 traces for {sc} combined: {len(ew_2)}")

    # Process NS1, only if a counterpart exists in NS2
    comp_path_ns1 = rf'{path}NS1'
    for filename in os.listdir(comp_path_ns1):
        if filename.endswith('.MSEED'):
            base_filename = extract_base_filename(filename)
            if base_filename in ns2_files:  # Match based on the base filename
                file_path = os.path.join(comp_path_ns1, filename)
                current_stream = read(file_path)
                for trace in current_stream:
                    trace.stats.filename = base_filename
                ns_1 += current_stream
    print(f"Total NS1 traces for {sc} combined: {len(ns_1)}")
    
    # Process NS2, only if a counterpart exists in NS1
    comp_path_ns2 = rf'{path}NS2'
    for filename in os.listdir(comp_path_ns2):
        if filename.endswith('.MSEED'):
            base_filename = extract_base_filename(filename)
            if base_filename in ns1_files:  # Match based on the base filename
                file_path = os.path.join(comp_path_ns2, filename)
                current_stream = read(file_path)
                for trace in current_stream:
                    trace.stats.filename = base_filename
                ns_2 += current_stream
    print(f"Total NS2 traces for {sc} combined: {len(ns_2)}")

    return ew_1, ew_2, ns_1, ns_2

# Function to combine base and surface motions for both small and large strains
def combine_strains(mp, sc):
    # Load large strain motions
    ew_1_large, ew_2_large, ns_1_large, ns_2_large = load_site_motions(mp, sc, 'large')
    
    rec_surf_large = ew_2_large + ns_2_large
    rec_base_large = ew_1_large + ns_1_large
    
    print(f"Total large surface traces combined for {sc}: {len(rec_surf_large)}")
    print(f"Total large base traces combined for {sc}: {len(rec_base_large)}")

    # Load small strain motions
    ew_1_small, ew_2_small, ns_1_small, ns_2_small = load_site_motions(mp, sc, 'small')
    
    rec_surf_small = ew_2_small + ns_2_small
    rec_base_small = ew_1_small + ns_1_small
    
    print(f"Total small surface traces combined for {sc}: {len(rec_surf_small)}")
    print(f"Total small base traces combined for {sc}: {len(rec_base_small)}")
    
    return rec_base_small, rec_surf_small, rec_base_large, rec_surf_large

    """
    Example
    -------------
    site_codes = ['KMMH14','KSRH07','FKSH11']
    for sc in site_codes:
        rec_base_small, rec_surf_small, rec_base_large, rec_surf_large = combine_strains(mp, sc)
    """

def convert_to_motion(traces, sc):
    motions = []
    for trace in traces:
        dt = trace.stats.delta
        accelerations = trace.data
        motion = pystrata.motion.TimeSeriesMotion(
            filename=trace.stats.filename,
            description=f'{sc}', 
            time_step=dt,
            accels=accelerations
        )
        motions.append(motion)
    return motions

# ------------------
# Empirical Transfer Function Creations
# ------------------

def create_output_collection(freqs=None):
    if freqs is None:
        freqs = np.logspace(-1, 2, num=500)
    return pystrata.output.OutputCollection([
        pystrata.output.FourierAmplitudeSpectrumOutput(
            freqs=freqs,
            location=pystrata.output.OutputLocation("within", index=-1),
            ko_bandwidth=30
        )
    ])

def process_profile(profile_name, profile, mp, freqs=None):
    if freqs is None:
        freqs = np.logspace(-1, 2, num=500)

    # Load strains
    rec_base_small, rec_surf_small, rec_base_large, rec_surf_large = combine_strains(mp, profile_name)
    
    # Convert to motions
    motions_base_small = convert_to_motion(rec_base_small, profile_name)
    motions_surf_small = convert_to_motion(rec_surf_small, profile_name)
    motions_base_large = convert_to_motion(rec_base_large, profile_name)
    motions_surf_large = convert_to_motion(rec_surf_large, profile_name)
    
    # Create output collections
    base_small = create_output_collection(freqs)
    surface_small = create_output_collection(freqs)
    base_large = create_output_collection(freqs)
    surface_large = create_output_collection(freqs)
    
    # Create calculator
    calc = pystrata.propagation.LinearElasticCalculator()
    
    # Calculate for small strain motions
    for motion in motions_base_small:
        calc(motion, profile, profile.location("within", index=-1))
        base_small(calc)
    
    for motion in motions_surf_small:
        calc(motion, profile, profile.location("within", index=-1))
        surface_small(calc)
    
    # Calculate for large strain motions
    for motion in motions_base_large:
        calc(motion, profile, profile.location("within", index=-1))
        base_large(calc)
    
    for motion in motions_surf_large:
        calc(motion, profile, profile.location("within", index=-1))
        surface_large(calc)
    
    # Convert to DataFrames
    base_small_df = base_small[-1].to_dataframe()
    surface_small_df = surface_small[-1].to_dataframe()
    base_large_df = base_large[-1].to_dataframe()
    surface_large_df = surface_large[-1].to_dataframe()
    
    return base_small_df, surface_small_df, base_large_df, surface_large_df

def plot_transfer_functions(base_small_df, surface_small_df, base_large_df, surface_large_df, profile_name, freqs=None):
    if freqs is None:
        freqs = np.logspace(-1, 2, num=500)

    # Plot for small strains only
    plt.figure(figsize=(10, 6))

    # Small-Strain Transfer Functions (Grey)
    small_strain_ratio = surface_small_df / base_small_df
    for col in small_strain_ratio.columns:
        plt.plot(freqs, small_strain_ratio[col], color='grey', alpha=0.3)

    # Log-normal median of small strain transfer functions
    log_all_transfer_functions = np.log(small_strain_ratio.values)
    median_tf = np.exp(np.mean(log_all_transfer_functions, axis=1))
    plt.plot(freqs, median_tf, color='k', label=f'Median TF (Small-Strain) - {profile_name}', linewidth=2)

    # Annotate the maximum amplitude for the small-strain median
    valid_indices = (freqs >= 0.5) & (freqs <= 20)
    limited_freqs = freqs[valid_indices]
    limited_amplitudes = median_tf[valid_indices]

    # Find the index of the maximum amplitude within this range
    max_amp_index = np.argmax(limited_amplitudes)
    max_amp_freq = limited_freqs[max_amp_index]
    max_amp_value = limited_amplitudes[max_amp_index]

    # Annotate on plot
    plt.annotate(f'Max Amp: {max_amp_freq:.2f} Hz', xy=(max_amp_freq, max_amp_value),
                xytext=(max_amp_freq * 1.5, max_amp_value * 1.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    # Configure plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Transfer Function')
    plt.xlim(0.2, 40)
    plt.ylim(0.1, 100)
    plt.title(f'Small-Strain Empirical Transfer Functions - {profile_name}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    
    # Plot for both small and large strains
    plt.figure(figsize=(10, 6))
    # Small-Strain Transfer Functions (Grey)
    for col in small_strain_ratio.columns:
        plt.plot(freqs, small_strain_ratio[col], color='grey', alpha=0.3)
    # Log-normal median of small strain transfer functions
    plt.plot(freqs, median_tf, color='k', label=f'Median TF (Small-Strain) - {profile_name}', linewidth=2)
    # Large-Strain Transfer Functions (Blue)
    large_strain_ratio = surface_large_df / base_large_df
    for i, col in enumerate(large_strain_ratio.columns):
        if i == 0:
            plt.plot(freqs, large_strain_ratio[col], color='blue', label='Large-Strain Transfer Functions')
        else:
            plt.plot(freqs, large_strain_ratio[col], color='blue')
    # Configure plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Transfer Function')
    plt.xlim(0.2, 40)
    plt.ylim(0.1, 100)
    plt.title(f'Small and Large-Strain Empirical Transfer Functions - {profile_name}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    """
    Example
    -------------
    freqs = np.logspace(-1, 2, num=500)
    for profile_name, profile in profiles.items():
        base_small_df, surface_small_df, base_large_df, surface_large_df = process_profile(profile_name, profile, mp, freqs)
        plot_transfer_functions(base_small_df, surface_small_df, base_large_df, surface_large_df, profile_name, freqs)
    """