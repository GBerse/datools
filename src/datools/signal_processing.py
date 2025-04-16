# ------------------
# Tools to downlaod ground motions from Kik-Net
# ------------------

import os
import tarfile
import glob
from obspy import read, Stream, UTCDateTime, Trace
import os
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import pandas as pd
from scipy.signal import detrend, butter, filtfilt
from scipy import signal
from obspy.signal.util import _npts2nfft
from numpy.fft import fft
import pykooh
from scipy.interpolate import interp1d

def grabASCIIs(tarfile_path, exdir):
    """
    Extract files from a tar.gz archive and remove specific files.

    Parameters:
    ----------
    - tarfile_path: path to the compressed .tar.gz folder.
    - exdir: directory for output.
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(exdir):
        os.makedirs(exdir)

    # Extract .tar or .tar.gz files
    if tarfile_path.endswith('.tar'):
        with tarfile.open(tarfile_path, 'r') as tar:
            tar.extractall(path=exdir)
    elif tarfile_path.endswith('.tar.gz'):
        with tarfile.open(tarfile_path, "r:gz") as tar:
            tar.extractall(path=exdir)

    # Look for remaining .tar.gz files in the output directory
    lftar_gz = glob.glob(os.path.join(exdir, '*.tar.gz'))
    while len(lftar_gz) != 0:
        for tar_gz_path in lftar_gz:
            with tarfile.open(tar_gz_path, "r:gz") as tar:
                tar.extractall(path=exdir)
            # Remove the .tar.gz file after its contents have been extracted
            os.remove(tar_gz_path)
        lftar_gz = glob.glob(os.path.join(exdir, '*.tar.gz'))

    # Remove '.gz', '.wave.ps.gz', and '.rsp.ps.gz' files considered as 'trash'
    for gz_pattern in ['*.gz', '*.wave.ps.gz', '*.rsp.ps.gz']:
        for gz_path in glob.glob(os.path.join(exdir, gz_pattern)):
            os.remove(gz_path)

    # Example usage:
    # grabASCII('path_to_your_tarfile.tar.gz', 'output_directory')

# ------------------
# Signal Processing; Obspy Stream/Trace Class utilized
# ------------------

def genstream(dir, ext):
    """
    Creates an Obspy Stream object from ground motions in a given directory

    Parameters:
    ----------
    - dir: path folder w/motions.
    - ext: extension name of the files you are creating a Stream from.

     Returns
    ----------
    - stream: constructed 
    """
        
    # Empty Obspy stream
    stream = Stream()
    # let's go through and see what relevant files
    # First, what files are in there
    for filename in os.listdir(dir):
        # what are their paths
        filepath = os.path.join(dir, filename)

        # making sure it's the right file (defined by ext)
        if filepath.endswith(f'.{ext}'):
            file_stream = read(filepath)

            # I want to store unique id for later
            filename = os.path.basename(filepath)

            # Add filename (not path) to trace stats and append trace to stream
            for trace in file_stream:
                trace.stats.filename = filename
                stream += trace
    print(f"Total traces combined: {len(stream)}")

    return stream

    # Example usage:
    # genstream('path_to_your_folder', 'EW2')

def detrend_demean(input_stream):
    """
    This code:
    (1) Detrends the data with a fitted line (linear detrend)
    (2) Demeans the data
    (3) Converts machine units in KikNet file to usable engineering units

    Parameters:
    - input_stream: Obspy stream object
    """

    for ii in range(len(input_stream)):
        st = input_stream.traces[ii]
        calibfactor = st.stats.calib * 100
        st.data = st.data * calibfactor
        st.data = st.data * 0.0010197162129779  # gal to g
        st = st.detrend("demean")  # Subtracting the mean of the time series from each value
        st = st.detrend("linear")  # Fitting a straight line to the data, and then  --> real data - point on the line
        input_stream.traces[ii] = st

    return(input_stream)

    #NOTE: This is function is specific to KikNet Calibrations

def narrowstream(stream, before, after, trigger_adjusted):
    """
    Code that uses recursive STA/LTA picking algorithm to find the trigger times and narrow the overall stream.
    More information on picker is avaliable here: https://gfzpublic.gfz-potsdam.de/rest/items/item_4097/component/file_4098/content

    The process in this code is as follows:
    This function processes seismic waveform data to:
    1. Detect triggers (earthquake signals) using the recursive STA/LTA algorithm.
    2. Trim each trace around detected triggers with user-defined time windows.
    3. Return trimmed traces, trigger metadata, and traces w/out triggers.
    *** traces w/out triggers failed the STA/LTA trigger detection algorithm

    Parameters
    ----------
    stream : obspy.Stream
        Input stream containing seismic traces to analyze.
    before : float
        Seconds to retain **before** the detected trigger.
    after : float
        Seconds to retain **after** the detected trigger.
    trigger_adjusted : float
        Time adjustment (in seconds) applied to the detected trigger time.

    Returns
    ---------
    new_stream : obspy.Stream
        Stream containing traces trimmed around detected triggers.
    trace_data : pandas.DataFrame
        DataFrame with trigger metadata.
    no_trigger_stream : obspy.Stream
        Stream containing traces where no triggers were detected.
    
    """

    # Setting up the figure for original data plotting
    n_subplots = len(stream)
    n_cols = int(np.ceil(np.sqrt(n_subplots)))
    n_rows = int(np.ceil(n_subplots / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))

    if n_rows > 1 or n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    trace_data = pd.DataFrame(columns=["station", "event_time", "start", "end", "trigger_time", "relative_trigger_time"])
    new_stream = Stream()  # New stream for adjusted time series
    no_trigger_stream = Stream()  # Stream for traces without PGA-triggering windows

    for ii in range(n_subplots):
        trace = stream[ii]
        start_time = trace.stats.starttime
        site_id = trace.stats.station
        date_str = start_time.strftime('%Y-%m-%d')

        axs[ii].set_title(f'{site_id} - {date_str}', fontsize=10)

        sampling_rate = trace.stats.sampling_rate
        cft = recursive_sta_lta(trace.data, int(6 * sampling_rate), int(10 * sampling_rate))
        on_of = trigger_onset(cft, 1.5, 0.5)

        axs[ii].set_xlabel('Time [s]', fontsize=8)
        axs[ii].set_ylabel('Acceleration [g]', fontsize=8)

        axs[ii].plot(trace.times(), trace.data, 'k')
        ymin, ymax = axs[ii].get_ylim()

        pga_index = np.argmax(np.abs(trace.data))
        relevant_trigger = [trigger for trigger in on_of if trigger[0] <= pga_index <= trigger[1]]

        if not relevant_trigger:
            no_trigger_stream.append(trace)
        else:
            for start, end in relevant_trigger:
                pre_trigger = before  # seconds before trigger start
                post_trigger = after  # seconds after trigger end
                start_sample = max(0, (start / sampling_rate - pre_trigger))
                end_sample = min(len(trace.data), end / sampling_rate + post_trigger)
                new_segment = trace.slice(starttime=start_time + start_sample,
                                          endtime=start_time + end_sample)
                new_stream.append(new_segment)

                original_trigger_time = start / sampling_rate
                adjusted_trigger_time = original_trigger_time - trigger_adjusted
                if start_sample <= adjusted_trigger_time <= end_sample:
                    trigger_index = adjusted_trigger_time
                else:
                    adjusted_trigger_time = original_trigger_time
                    trigger_index = adjusted_trigger_time

                relative_trigger = trigger_index - start_sample
                relative_duration = end_sample - start_sample

                new_row = pd.DataFrame({
                    "station": [site_id],
                    "event_time": [date_str],
                    "start": [start_sample],
                    "end": [end_sample],
                    "trigger_time": [trigger_index],
                    "relative_trigger_time": [relative_trigger],
                    "relative_duration": [relative_duration]
                })
                trace_data = pd.concat([trace_data, new_row], ignore_index=True)

                axs[ii].vlines(trigger_index, ymin, ymax, color='r', linewidth=2, label='Trigger Start')
                axs[ii].vlines(end_sample, ymin, ymax, color='b', linewidth=2, label='Trigger End')

                break  # Only the first relevant trigger

    plt.tight_layout()
    plt.show()

    # Plotting the newly narrowed time histories on their own plot
    n_new_subplots = len(new_stream)
    n_new_cols = int(np.ceil(np.sqrt(n_new_subplots)))
    n_new_rows = int(np.ceil(n_new_subplots / n_new_cols))
    fig_new, axs_new = plt.subplots(n_new_rows, n_new_cols, figsize=(15, 10))

    if n_new_rows > 1 or n_new_cols > 1:
        axs_new = axs_new.flatten()
    else:
        axs_new = [axs_new]

    for ii in range(n_new_subplots):
        new_trace = new_stream[ii]
        new_start_time = new_trace.stats.starttime
        new_site_id = new_trace.stats.station
        new_date_str = new_start_time.strftime('%Y-%m-%d')

        axs_new[ii].set_title(f'{new_site_id} - {new_date_str}', fontsize=10)

        axs_new[ii].set_xlabel('Time [s]', fontsize=8)
        axs_new[ii].set_ylabel('Acceleration [g]', fontsize=8)

        axs_new[ii].plot(new_trace.times(), new_trace.data, 'k')

        trigger_time = trace_data.loc[ii, 'trigger_time'] - trace_data.loc[ii, 'start']
        ymin_new, ymax_new = axs_new[ii].get_ylim()
        axs_new[ii].vlines(trigger_time, ymin_new, ymax_new, color='r', linewidth=2, label='Trigger Time')

    plt.tight_layout()
    plt.show()

    # Plotting the traces without PGA-triggering windows
    n_no_trigger_subplots = len(no_trigger_stream)
    if n_no_trigger_subplots > 0:
        n_no_trigger_cols = int(np.ceil(np.sqrt(n_no_trigger_subplots)))
        n_no_trigger_rows = int(np.ceil(n_no_trigger_subplots / n_no_trigger_cols))
        fig_no_trigger, axs_no_trigger = plt.subplots(n_no_trigger_rows, n_no_trigger_cols, figsize=(15, 10))

        if n_no_trigger_rows > 1 or n_no_trigger_cols > 1:
            axs_no_trigger = axs_no_trigger.flatten()
        else:
            axs_no_trigger = [axs_no_trigger]

        for ii in range(n_no_trigger_subplots):
            no_trigger_trace = no_trigger_stream[ii]
            no_trigger_start_time = no_trigger_trace.stats.starttime
            no_trigger_site_id = no_trigger_trace.stats.station
            no_trigger_date_str = no_trigger_start_time.strftime('%Y-%m-%d')

            axs_no_trigger[ii].set_title(f'{no_trigger_site_id} - {no_trigger_date_str}', fontsize=10)
            axs_no_trigger[ii].set_xlabel('Time [s]', fontsize=8)
            axs_no_trigger[ii].set_ylabel('Acceleration [g]', fontsize=8)
            axs_no_trigger[ii].plot(no_trigger_trace.times(), no_trigger_trace.data, 'k')

        plt.tight_layout()
        plt.show()
    else:
        print("No traces without triggers to display.")

    return new_stream, trace_data, no_trigger_stream

# -------------------------------------------------------------------------------------------------

# ------------------
# Signal Processing (con.); SNR Calculations
# ------------------

    """
    There are three codes below, each doing the same mathematics, but give slightly different outputs.

    (1). snr()
        Returns both SNR plots and data:
        +SNR Plots       +SNR Data
    (2). snr_comp()
        Returns constituent plots/data (i.e., signal and noise seperately):
        +Signal Plots   +Noise Plots    +Signal Data    +Noise Data
    (3). snr_no_plots()
        Returns SNR data only
        +SNR Data
    """

def snr(trimmed_stream, pwave_arrival):
    """
    Calculate Signal-to-Noise Ratio (SNR) for seismic traces and plot frequency-dependent SNR.

    Processes trimmed seismic traces to:
    1. Extract noise (pre-trigger) and signal (post-trigger) windows.
    2. Compute FFT for noise and signal segments.
    3. Calculate SNR as the ratio of signal FFT to noise FFT.
    4. Identify frequencies where SNR drops below 3 (used for corner frequency estimation).
    5. Plot smoothed SNR vs. frequency for each trace.

    Parameters
    ----------
    trimmed_stream : obspy.Stream
        Stream containing trimmed seismic traces (centered on triggers).
    pwave_arrival : pandas.DataFrame
        DataFrame with metadata including:
        - `relative_trigger_time` (float): Trigger time relative to trace start (seconds).
        - `relative_duration` (float): Duration of the trimmed trace (seconds).
        - `station` (str): Station ID.
        - `event_time` (str): Date of the event.

    Returns
    -------
    pandas.DataFrame
        DataFrame with SNR analysis results for each trace:
        - `Site ID` (str): Station identifier.
        - `Date` (UTCDateTime): Trace start time.
        - `Trace Number` (int): Index of the trace in the stream.
        - `SNR=3 Frequencies (upper)` (float): First frequency >15Hz where SNR <3 (Hz).
        - `SNR=3 Frequencies` (float): First frequency <15Hz where SNR <3 (Hz).
        - `SNR < 1` (str): 'Yes' if SNR <1 above 15Hz, else 'No'.
        - `SNR < 3` (str): 'Yes' if SNR <3 below 15Hz, else 'No'.
        - `UniqueID` (str): Filename or unique trace identifier.
    """

    #Initialising a table wo wir koennen frequencies stellen
    SNR_corner_freq = []

    #Creating one big plot with everything
    n_subplots = len(trimmed_stream)

    n_cols = int(np.ceil(np.sqrt(n_subplots)))
    n_rows = int(np.ceil(n_subplots / n_cols))

    base_size = 12  # Base font size for one plot
    scale_factor = max(1, np.sqrt(n_subplots / 4))  # Scale factor increases as the number of plots increases
    font_size = max(6, int(base_size / scale_factor))  # Prevent font size from becoming too small

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))  # Make the figure size square

    #Flattening :)
    if n_rows > 1 and n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs] #iterable

    for ii in range(len(trimmed_stream)):
        # Define noise and signal windows
        noise_start = 0
        noise_end = pwave_arrival.loc[ii, 'relative_trigger_time'] - 1  # 1 seconds before the trigger time
        signal_start = pwave_arrival.loc[ii, 'relative_trigger_time']
        signal_end = pwave_arrival.loc[ii, 'relative_duration']

        # Slice noise and signal segments from the trace
        noise = trimmed_stream[ii].slice(starttime=trimmed_stream[ii].stats.starttime + noise_start,
                                         endtime=trimmed_stream[ii].stats.starttime + noise_end)
        signal = trimmed_stream[ii].slice(starttime=trimmed_stream[ii].stats.starttime + signal_start,
                                          endtime=trimmed_stream[ii].stats.starttime + signal_end)

        # Compute the number of points for FFT
        nfft = _npts2nfft(max(len(noise.data), len(signal.data)))

        # Calculate FFT for noise and signal
        noise_fft = trimmed_stream[ii].stats["delta"] * np.abs(fft(noise.data, n=nfft))
        signal_fft = trimmed_stream[ii].stats["delta"] * np.abs(fft(signal.data, n=nfft))

        # Calculate SNR at each frequency
        snr_frequency = signal_fft / noise_fft

        # Frequency axis
        freqs = np.fft.fftfreq(nfft, d=trimmed_stream[ii].stats.delta)[:nfft // 2]

        # Smooth the FFT and SNR data
        b = 40  # Smoothing coefficient
        noise_amps = pykooh.smooth(freqs, freqs, noise_fft[:nfft // 2], b)
        signal_amps = pykooh.smooth(freqs, freqs, signal_fft[:nfft // 2], b)
        amps = pykooh.smooth(freqs, freqs, snr_frequency, b)

        # Plotting the smoothed data
        axs[ii].plot(freqs, amps, color='purple', label='SNR')

        # Making Subplot
        site_id = pwave_arrival.loc[ii, 'station']
        date_str = pwave_arrival.loc[ii, 'event_time']
        axs[ii].set_title(f'{site_id} - {date_str}', fontsize=font_size)
        axs[ii].set_xlabel('Frequency (Hz)', fontsize=font_size)
        axs[ii].set_ylabel('SNR', fontsize=font_size)
        axs[ii].tick_params(axis='both', which='major', labelsize=font_size - 1)

        # Setting log scale for both axes
        axs[ii].set_xscale('log')
        axs[ii].set_yscale('log')

        # gridlines
        axs[ii].grid(which='both', color='gray', linestyle='-', linewidth=0.25)

        # Limits
        axs[ii].set_xlim(0.1, 50)
        axs[ii].set_ylim(0.01, 1000000)

        # SNR cutoffs
        axs[ii].axhline(y=3, color='green', linestyle='--')

        # Check if SNR ever drops below 3 at frequencies above 15Hz
        high_freq_indices = freqs > 15
        snr_below_3_above_15Hz = np.any(amps[high_freq_indices] < 3)

        snr_below_3_upper_first = None  # Default value indicating not set or not applicable

        if snr_below_3_above_15Hz:
            # Interpolation for more precise detection of SNR=3 upper level crossing
            interp_func = interp1d(freqs, amps, kind='linear', fill_value="extrapolate")
            fine_freqs = np.linspace(freqs.min(), freqs.max(), num=10000)
            fine_amps = interp_func(fine_freqs)
            crossing_indices_below_3_upper = np.where(np.diff(np.sign(fine_amps[fine_freqs > 15] - 3)))[0]
            if crossing_indices_below_3_upper.size > 0:
                snr_below_3_upper_first = fine_freqs[fine_freqs > 15][crossing_indices_below_3_upper[0]]

        # Check if SNR is below 3 below 15Hz
        low_freq_indices = (freqs > 0.01) & (freqs < 15)
        snr_below_3_below_15Hz = np.any(amps[low_freq_indices] < 3)

        snr3_freqs = None

        # More accurate
        if snr_below_3_below_15Hz:
            # Interpolation for more precise detection of SNR=1 crossing
            interp_func = interp1d(freqs, amps, kind='linear', fill_value="extrapolate")
            fine_freqs = np.linspace(freqs.min(), freqs.max(), num=10000)
            fine_amps = interp_func(fine_freqs)
            crossing_indices = np.where(np.diff(np.sign(fine_amps - 3)))[0]
            if crossing_indices.size > 0:
                snr3_freqs = fine_freqs[crossing_indices[0]]  # First crossing frequency

        # Retrieve site ID and date from the trace metadata
        site_id = trimmed_stream[ii].stats.station
        date = trimmed_stream[ii].stats.starttime
        filename = trimmed_stream[ii].stats.filename

        SNR_corner_freq.append({
            'Site ID': site_id,
            'Date': date,
            'Trace Number': ii + 1,
            'SNR=3 Frequencies (upper)': snr_below_3_upper_first,
            'SNR=3 Frequencies': snr3_freqs,
            'SNR < 1': 'Yes' if snr_below_3_above_15Hz else 'No',
            'SNR < 3 ': 'Yes' if snr_below_3_below_15Hz else 'No',
            'UniqueID': filename
        })

    SNR_corner_freq = pd.DataFrame(SNR_corner_freq)

    for ax in axs[n_subplots:]:
        ax.set_visible(False)
    plt.tight_layout
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)

    return SNR_corner_freq

def snr_comp(trimmed_stream,pwave_arrival):
    """
    Plot smoothed Fourier Amplitude Spectra (FAS) for noise and signal segments.

    Similar to `snr()`, but visualizes noise and signal FAS instead of SNR. Useful for 
    comparing raw spectral content of noise vs. signal.

    Parameters
    ----------
    trimmed_stream : obspy.Stream
        Stream of trimmed seismic traces.
    pwave_arrival : pandas.DataFrame
        DataFrame with metadata (same structure as in `snr()`).

    Notes
    -----
    - Generates a grid of log-log plots showing noise (orange) and signal (blue) FAS.
    - No return value; this is a visualization-only function.
    """

    # Creating one big plot with everything
    n_subplots = len(trimmed_stream)

    n_cols = int(np.ceil(np.sqrt(n_subplots)))
    n_rows = int(np.ceil(n_subplots / n_cols))

    base_size = 12  # Base font size for one plot
    scale_factor = max(1, np.sqrt(n_subplots / 4))  # Scale factor increases as the number of plots increases
    font_size = max(6, int(base_size / scale_factor))  # Prevent font size from becoming too small

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))  # Make the figure size square

    # Flattening :)
    if n_rows > 1 and n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]  # iterable

    for ii in range(len(trimmed_stream)):
        # Define noise and signal windows
        noise_start = 0
        noise_end = pwave_arrival.loc[ii, 'relative_trigger_time'] - 1  # 1 seconds before the trigger time
        signal_start = pwave_arrival.loc[ii, 'relative_trigger_time']
        signal_end = pwave_arrival.loc[ii, 'relative_duration']

        # Slice noise and signal segments from the trace
        noise = trimmed_stream[ii].slice(starttime=trimmed_stream[ii].stats.starttime + noise_start,
                                         endtime=trimmed_stream[ii].stats.starttime + noise_end)
        signal = trimmed_stream[ii].slice(starttime=trimmed_stream[ii].stats.starttime + signal_start,
                                          endtime=trimmed_stream[ii].stats.starttime + signal_end)

        # Compute the number of points for FFT
        nfft = _npts2nfft(max(len(noise.data), len(signal.data)))

        # Calculate FFT for noise and signal
        noise_fft = trimmed_stream[ii].stats["delta"] * np.abs(fft(noise.data, n=nfft))
        signal_fft = trimmed_stream[ii].stats["delta"] * np.abs(fft(signal.data, n=nfft))

        # Frequency axis
        freqs = np.fft.fftfreq(nfft, d=trimmed_stream[ii].stats.delta)[:nfft // 2]

        # Smooth the FFT and SNR data
        b = 40  # Smoothing coefficient
        noise_amps = pykooh.smooth(freqs, freqs, noise_fft[:nfft // 2], b)
        signal_amps = pykooh.smooth(freqs, freqs, signal_fft[:nfft // 2], b)

        # Plotting the smoothed data
        axs[ii].plot(freqs, noise_amps, color='orange', label='Noise')
        axs[ii].plot(freqs, signal_amps, color='blue', label='Signal')

        # Making Subplot
        site_id = pwave_arrival.loc[ii, 'station']
        date_str = pwave_arrival.loc[ii, 'event_time']

        axs[ii].set_title(f'{site_id} - {date_str}', fontsize=font_size)
        axs[ii].set_xlabel('Frequency (Hz)',fontsize=font_size - 2)
        axs[ii].set_ylabel('FAS',fontsize=font_size)

        axs[ii].tick_params(axis='both', which='major', labelsize=font_size - 1)


        # Setting log scale for both axes
        axs[ii].set_xscale('log')
        axs[ii].set_yscale('log')

        # gridlines
        axs[ii].grid(which='both', color='gray', linestyle='-', linewidth=0.25)

        # Limits
        axs[ii].set_xlim(0.1, 50)

    for ax in axs[n_subplots:]:
        ax.set_visible(False)
    plt.tight_layout
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)

    handles, labels = axs[0].get_legend_handles_labels()  #
    fig.legend(handles, labels, loc='lower right')

def snr_no_plots(trimmed_stream, pwave_arrival):
    """
    Calculate SNR metrics without generating plots (lightweight version of `snr()`).

    Useful for batch processing where visualization is not needed.

    Parameters
    ----------
    trimmed_stream : obspy.Stream
        Stream of trimmed seismic traces.
    pwave_arrival : pandas.DataFrame
        DataFrame with metadata (same structure as in `snr()`).

    Returns
    -------
    pandas.DataFrame
        Identical to `snr()` output, but without plotting overhead.

    See Also
    --------
    snr : For the plotting version of this function.
    """

    #Initialising a table wo wir koennen frequencies stellen
    SNR_corner_freq = []

    for ii in range(len(trimmed_stream)):
        # Define noise and signal windows
        noise_start = 0
        noise_end = pwave_arrival.loc[ii, 'relative_trigger_time'] - 1  # 1 seconds before the trigger time
        signal_start = pwave_arrival.loc[ii, 'relative_trigger_time']
        signal_end = pwave_arrival.loc[ii, 'relative_duration']

        # Slice noise and signal segments from the trace
        noise = trimmed_stream[ii].slice(starttime=trimmed_stream[ii].stats.starttime + noise_start,
                                         endtime=trimmed_stream[ii].stats.starttime + noise_end)
        signal = trimmed_stream[ii].slice(starttime=trimmed_stream[ii].stats.starttime + signal_start,
                                          endtime=trimmed_stream[ii].stats.starttime + signal_end)

        # Compute the number of points for FFT
        nfft = _npts2nfft(max(len(noise.data), len(signal.data)))

        # Calculate FFT for noise and signal
        noise_fft = trimmed_stream[ii].stats["delta"] * np.abs(fft(noise.data, n=nfft))
        signal_fft = trimmed_stream[ii].stats["delta"] * np.abs(fft(signal.data, n=nfft))

        # Calculate SNR at each frequency
        snr_frequency = signal_fft / noise_fft

        # Frequency axis
        freqs = np.fft.fftfreq(nfft, d=trimmed_stream[ii].stats.delta)[:nfft // 2]

        # Smooth the FFT and SNR data
        b = 40  # Smoothing coefficient
        noise_amps = pykooh.smooth(freqs, freqs, noise_fft[:nfft // 2], b)
        signal_amps = pykooh.smooth(freqs, freqs, signal_fft[:nfft // 2], b)
        amps = pykooh.smooth(freqs, freqs, snr_frequency, b)

        # Check if SNR ever drops below 3 at frequencies above 15Hz
        high_freq_indices = freqs > 15
        snr_below_3_above_15Hz = np.any(amps[high_freq_indices] < 3)

        snr_below_3_upper_first = None  # Default value indicating not set or not applicable

        if snr_below_3_above_15Hz:
            # Interpolation for more precise detection of SNR=3 upper level crossing
            interp_func = interp1d(freqs, amps, kind='linear', fill_value="extrapolate")
            fine_freqs = np.linspace(freqs.min(), freqs.max(), num=10000)
            fine_amps = interp_func(fine_freqs)
            crossing_indices_below_3_upper = np.where(np.diff(np.sign(fine_amps[fine_freqs > 15] - 3)))[0]
            if crossing_indices_below_3_upper.size > 0:
                snr_below_3_upper_first = fine_freqs[fine_freqs > 15][crossing_indices_below_3_upper[0]]

        # Check if SNR is below 3 below 15Hz
        low_freq_indices = (freqs > 0.01) & (freqs < 15)
        snr_below_3_below_15Hz = np.any(amps[low_freq_indices] < 3)

        snr3_freqs = None

        # More accurate
        if snr_below_3_below_15Hz:
            # Interpolation for more precise detection of SNR=1 crossing
            interp_func = interp1d(freqs, amps, kind='linear', fill_value="extrapolate")
            fine_freqs = np.linspace(freqs.min(), freqs.max(), num=10000)
            fine_amps = interp_func(fine_freqs)
            crossing_indices = np.where(np.diff(np.sign(fine_amps - 3)))[0]
            if crossing_indices.size > 0:
                snr3_freqs = fine_freqs[crossing_indices[0]]  # First crossing frequency

        # Retrieve site ID and date from the trace metadata
        site_id = trimmed_stream[ii].stats.station
        date = trimmed_stream[ii].stats.starttime
        filename = trimmed_stream[ii].stats.filename

        SNR_corner_freq.append({
            'Site ID': site_id,
            'Date': date,
            'Trace Number': ii + 1,
            'SNR=3 Frequencies (upper)': snr_below_3_upper_first,
            'SNR=3 Frequencies': snr3_freqs,
            'SNR < 1': 'Yes' if snr_below_3_above_15Hz else 'No',
            'SNR < 3 ': 'Yes' if snr_below_3_below_15Hz else 'No',
            'UniqueID': filename
        })

    SNR_corner_freq = pd.DataFrame(SNR_corner_freq)
    return SNR_corner_freq

# ------------------
# Signal Processing (con.); Filter Frequency Functions
# ------------------

def corner_freq(trimmed_stream,SNR_corner_freq, min):
    """
   Garners the upper and lower bound frequencies used in an impending Buttersworth bandpass.
   Combines SNR-based frequency limits with UCLA's `get_fchp` method to determine optimal filter bounds.

    Parameters
    ----------
    trimmed_stream : obspy.Stream
        Stream of trimmed seismic traces (centered on triggers).
    SNR_corner_freq : pandas.DataFrame
        DataFrame from `snr()` containing SNR-derived frequency limits:
        - `SNR=3 Frequencies` (float): Lower frequency bound (Hz).
        - `SNR=3 Frequencies (upper)` (float): Upper frequency bound (Hz).
    min : float
        Minimum allowed high-pass frequency (Hz). Used as `fchp_min` in UCLA's method.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with new columns:
        - `UCLAhf` (float): High-pass freq from UCLA's method.
        - `lf` (float): Low-pass freq (80% Nyquist).
        - `corner_filter_frequency_hp` (float): Final high-pass (max of SNR/UCLA).
        - `corner_filter_frequency_lp` (float): Final low-pass (min of SNR/Nyquist).

    Notes
    -----
    - High-pass: Max of SNR=3 frequency or UCLA's `get_fchp`.
    - Low-pass: Min of SNR=3 upper frequency or 80% Nyquist (anti-aliasing).
    """

    from datools import UCLA as uc

    window = SNR_corner_freq
    fchp_results_list = []
    fclp_results_list = []

    window = window.fillna(value=np.nan)
    for trace in trimmed_stream:
        dt = trace.stats.delta
        acc = trace.data

        fchp_result = uc.get_fchp(dt=dt, acc=acc, fchp_min=min)

        fchp_results_list.append(fchp_result)
    window['UCLAhf'] = fchp_results_list

    for trace in trimmed_stream:
        dt = trace.stats.sampling_rate #the same in kiknet
        fclp = 0.8 * (dt /2)
        fclp_results_list.append(fclp)
    window['lf'] = fclp_results_list

    highest_values = window.apply(lambda row: np.nanmax([row['SNR=3 Frequencies'], row['UCLAhf']]), axis=1)
    lowest_values = window.apply(lambda row: np.nanmin([row['SNR=3 Frequencies (upper)'], row['lf']]), axis=1)
    window['corner_filter_frequency_hp'] = highest_values
    window['corner_filter_frequency_lp'] = lowest_values

    return window

def buttersworth(trimmed_stream,window,c=4):
    """
    Applies a Butterworth bandpass filter to the data stream with adjustable corners.
    
    Parameters:
    - trimmed_stream: Stream object to be filtered
    - window: DataFrame containing filter parameters
    - c: Number of corners for the filter (default=4)
    
    Returns:
    - filtered_stream: Filtered Stream object
    """

    filtered_stream = Stream()
    for ii in range(len(trimmed_stream)):
        corner_frequency_hp = window['corner_filter_frequency_hp'].iloc[ii]
        corner_frequency_lp = window['corner_filter_frequency_lp'].iloc[ii]

        filtered_trace = trimmed_stream[ii].filter("bandpass", freqmin=corner_frequency_hp, freqmax=corner_frequency_lp,
                                                   corners=c, zerophase=True)
        filtered_stream.append(filtered_trace)

    return filtered_stream

# ------------------
# Export Function
# ------------------

def save_seismic_data(stream, base_path, station_code, formats=("MSEED",), 
                     dir_prefix="i_love_science", extension_end=""):
    """
    Save seismic data in multiple formats with organized directory structure.
    
    Parameters:
        stream (Stream): ObsPy Stream object to save
        base_path (str): Base directory path
        station_code (str): Station code (e.g., 'EW1')
        formats (tuple): Tuple of output formats ('MSEED', 'SAC', 'SEED', etc.)
        dir_prefix (str): Prefix for output directory name
        extension_end (str): Optional suffix for directory name
    """
    if not isinstance(stream, Stream):
        raise ValueError("Input must be an ObsPy Stream object")
    
    # Create output directory path
    output_dir = os.path.join(
        base_path, 
        station_code, 
        f"{dir_prefix}{station_code}{extension_end}"
    )
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save in each requested format
    for fmt in formats:
        fmt = fmt.upper()  # Normalize format name
        
        if fmt == "MSEED":
            # Save each trace separately as MSEED
            for tr in stream:
                filename = f"{tr.stats.filename}.{fmt}"
                tr.write(os.path.join(output_dir, filename), format=fmt)
        
        elif fmt in ("SAC", "SEED", "GSE2", "ASCII"):
            # These formats typically save the whole stream together
            filename = f"{station_code}_combined.{fmt}"
            stream.write(os.path.join(output_dir, filename), format=fmt)
        
        else:
            print(f"Warning: Unsupported format '{fmt}' - skipping")
    
    print(f"Saved data for station {station_code} in {len(formats)} format(s) to:")
    print(f"  {output_dir}")

# ------------------
# Simple Plotting Functions
# ------------------

def plotstream(stream):
    if not stream:
        print("The stream is empty.")
        return

    # Creating one big plot with everything
    n_subplots = len(stream)

    # Breaking it up into a square of subplots. In other words, ceil looks at smallest value
    n_cols = int(np.ceil(np.sqrt(n_subplots)))
    n_rows = int(np.ceil(n_subplots / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))  # Make the figure size square

    # Flattening the matrix of subplots into a flat list for easier processing
    if n_rows > 1 or n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]  # Ensure axs is always an iterable

    # Initializing a blank list for the PGA values
    pgas = []

    # Actually plotting the data
    for ii in range(len(stream)):
        # Extract metadata
        start_time = stream[ii].stats.starttime
        site_id = stream[ii].stats.station  # Assuming the site ID is stored in the station attribute
        date_str = start_time.strftime('%Y-%m-%d')

        # Set title to site ID and date
        axs[ii].set_title(f'{site_id} - {date_str}', fontsize=10)  # Title with site ID and date

        # Logging the PGA
        pga = np.max(np.abs(stream[ii].data))
        pgas.append(pga)

        # Plotting time history
        axs[ii].plot(stream[ii].times(), stream[ii].data, label='Full Time History')

        # Making Subplot axis labels
        axs[ii].set_xlabel('Time [s]', fontsize=8)
        axs[ii].set_ylabel('Acceleration [g]', fontsize=8)

        # Gridlines
        axs[ii].grid(which='both', color='gray', linestyle='-', linewidth=0.25)

        # Adding the PGAs
        axs[ii].text(0.95, 0.95, f'Max: {pga:.4f} g', transform=axs[ii].transAxes,
                     fontsize=8, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))

    # Hide excess subplots
    for ax in axs[n_subplots:]:
        ax.set_visible(False)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for the suptitle
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.5)

    # Title
    plt.suptitle(f'Time Histories')

    # Show the plot
    plt.show()

def plotstream_vels(stream):
    if not stream:
        print("The stream is empty.")
        return

    # Creating one big plot with everything
    n_subplots = len(stream)

    # Breaking it up into a square of subplots. In other words, ceil looks at smallest value
    n_cols = int(np.ceil(np.sqrt(n_subplots)))
    n_rows = int(np.ceil(n_subplots / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))  # Make the figure size square

    # Flattening the matrix of subplots into a flat list for easier processing
    if n_rows > 1 or n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]  # Ensure axs is always an iterable

    # Initializing a blank list for the PGV values
    pgvs = []

    # Actually plotting the data
    for ii in range(len(stream)):
        # Extract metadata
        start_time = stream[ii].stats.starttime
        site_id = stream[ii].stats.station  # Assuming the site ID is stored in the station attribute
        date_str = start_time.strftime('%Y-%m-%d')
        time_str = start_time.strftime('%H:%M:%S')
        location = stream[ii].stats.location

        # Set title to site ID and date
        axs[ii].set_title(f'{site_id} - {date_str}', fontsize=10)  # Title with site ID and date

        # Logging the PGV
        pgv = np.max(np.abs(stream[ii].data))
        pgvs.append(pgv)

        # Plotting time history
        axs[ii].plot(stream[ii].times(), stream[ii].data, label='Full Time History')

        # Making Subplot axis labels
        axs[ii].set_xlabel('Time [s]', fontsize=8)
        axs[ii].set_ylabel('Velocity [cm/s]', fontsize=8)

        # Gridlines
        axs[ii].grid(which='both', color='gray', linestyle='-', linewidth=0.25)

        # Adding the PGVs
        axs[ii].text(0.95, 0.95, f'Max: {pgv:.2f} cm/s', transform=axs[ii].transAxes,
                     fontsize=8, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))

    # Hide excess subplots
    for ax in axs[n_subplots:]:
        ax.set_visible(False)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for the suptitle
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.5)

    # Title
    plt.suptitle(f'Velocity Time Histories')

    # Show the plot
    plt.show()

def plotstream_minimal(stream):
    if not stream:
        print("The stream is empty.")
        return

    n_subplots = len(stream)
    n_cols = int(np.ceil(np.sqrt(n_subplots)))
    n_rows = int(np.ceil(n_subplots / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))

    if n_rows > 1 or n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    pgas = []

    for ii in range(len(stream)):
        pga = np.max(np.abs(stream[ii].data))
        pgas.append(pga)

        axs[ii].plot(stream[ii].times(), stream[ii].data, label='Full Time History')

        axs[ii].text(0.95, 0.95, f'Max: {pga:.4f} g', transform=axs[ii].transAxes,
                     fontsize=8, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))
        axs[ii].set_xticks([])
        axs[ii].set_yticks([])

    for ax in axs[n_subplots:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.5)
    plt.suptitle(f'Time Histories; Demeaned and Linear Detrend')

    plt.show()

def plotstream_vels_minimal(stream):
    if not stream:
        print("The stream is empty.")
        return

    n_subplots = len(stream)
    n_cols = int(np.ceil(np.sqrt(n_subplots)))
    n_rows = int(np.ceil(n_subplots / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))

    if n_rows > 1 or n_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    pgvs = []

    for ii in range(len(stream)):
        pgv = np.max(np.abs(stream[ii].data))
        pgvs.append(pgv)

        axs[ii].plot(stream[ii].times(), stream[ii].data, label='Full Time History')

        axs[ii].text(0.95, 0.95, f'Max: {pgv:.2f} cm/s', transform=axs[ii].transAxes,
                     fontsize=8, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(facecolor='white', alpha=0.5))
        axs[ii].set_xticks([])
        axs[ii].set_yticks([])

    for ax in axs[n_subplots:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.5)
    plt.suptitle(f'Velocity Time Histories')

    plt.show()


