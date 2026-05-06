#!/usr/bin/env python3
"""
3D Spatial Audio Visualization Tool
Senior DSP Engineering Implementation
Handles ITU-R BS.2051 spatial audio with robust STFT analysis and Plotly 3D rendering
"""

import argparse
import warnings
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import stft
import plotly.graph_objects as go


class SpatialAudioVisualizer:
    """High-performance 3D spatial audio visualization engine."""
    
    # ITU-R BS.2051-3 7.1.4 Standard Layout (12 channels)
    CHANNEL_LAYOUT = {
        0: ("L", 30, 0),           # Left
        1: ("R", -30, 0),          # Right
        2: ("C", 0, 0),            # Center
        3: ("LFE", 0, -90),        # Low Frequency Effects
        4: ("Ls", 110, 0),         # Left Surround
        5: ("Rs", -110, 0),        # Right Surround
        6: ("Lrs", 135, 0),        # Left Rear Surround
        7: ("Rrs", -135, 0),       # Right Rear Surround
        8: ("Ltf", 45, 45),        # Left Top Front
        9: ("Rtf", -45, 45),       # Right Top Front
        10: ("Ltr", 135, 45),      # Left Top Rear
        11: ("Rtr", -135, 45),     # Right Top Rear
    }
    
    def __init__(self, wav_path, stft_nperseg=2048):
        """
        Initialize the spatial audio visualizer.
        
        Args:
            wav_path: Path to input WAV file
            stft_nperseg: STFT window length
        """
        self.wav_path = Path(wav_path)
        self.stft_nperseg = stft_nperseg
        self.fs = None
        self.audio_data = None
        self.num_channels = None
        
    def load_audio(self):
        """
        Robustly load WAV file using memory-mapped I/O.
        Safely ignores non-data chunks without ImportError.
        """
        print(f"Loading audio from: {self.wav_path}")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fs, self.audio_data = wavfile.read(self.wav_path, mmap=True)
        
        # Handle mono audio
        if self.audio_data.ndim == 1:
            self.audio_data = self.audio_data.reshape(-1, 1)
        
        self.num_channels = self.audio_data.shape[1]
        print(f"Sample rate: {self.fs} Hz | Channels: {self.num_channels} | Duration: {len(self.audio_data)/self.fs:.2f}s")
        
        return self
    
    def _spherical_to_cartesian(self, azimuth, elevation):
        """
        Convert spherical coordinates to 3D Cartesian coordinates.
        
        Args:
            azimuth: Angle in degrees (-180 to 180)
            elevation: Angle in degrees (-90 to 90)
        
        Returns:
            Tuple of (x, y, z) normalized to unit sphere
        """
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        x = np.cos(el_rad) * np.sin(az_rad)
        y = np.cos(el_rad) * np.cos(az_rad)
        z = np.sin(el_rad)
        
        return x, y, z
    
    def compute_stft_peaks(self):
        """
        Compute STFT for all channels and extract absolute peak energy (Max Hold).
        Collapses time domain completely. Returns exactly 1 data point per channel.
        """
        print("Computing STFT and extracting absolute peak energy per channel...")
        
        # Normalize audio to prevent overflow in STFT
        audio_normalized = self.audio_data.astype(np.float32) / np.iinfo(self.audio_data.dtype).max
        
        # Pre-allocate results
        spatial_coords = []
        peak_dbfs_values = []
        channel_labels = []
        
        # Process each channel
        for ch_idx in range(min(self.num_channels, 12)):  # Limit to 12 channels
            channel_name, azimuth, elevation = self.CHANNEL_LAYOUT.get(
                ch_idx, (f"CH{ch_idx}", 0, 0)
            )
            
            x, y, z = self._spherical_to_cartesian(azimuth, elevation)
            
            # Compute STFT
            f, t, Zxx = stft(
                audio_normalized[:, ch_idx],
                fs=self.fs,
                nperseg=self.stft_nperseg,
                noverlap=self.stft_nperseg // 2
            )
            
            # Magnitude spectrum
            magnitude = np.abs(Zxx)
            
            # Convert to dBFS (reference: 1.0 amplitude)
            epsilon = 1e-10
            dbfs = 20 * np.log10(magnitude + epsilon)
            
            # Absolute Max Hold: single maximum value across all time and frequency bins
            peak_energy = np.max(dbfs)
            
            # Clamp to noise floor [-60, 0] dBFS
            peak_energy_clamped = np.clip(peak_energy, -60, 0)
            
            spatial_coords.append((x, y, z))
            peak_dbfs_values.append(peak_energy_clamped)
            channel_labels.append(channel_name)
            
            print(f"  Channel {ch_idx:2d} ({channel_name:4s}): Azimuth={azimuth:4.0f}°, Elevation={elevation:+3.0f}°, Peak={peak_energy_clamped:+6.1f} dBFS")
        
        # Convert to numpy arrays
        spatial_coords = np.array(spatial_coords)
        peak_dbfs_values = np.array(peak_dbfs_values)
        
        print(f"Total visualization points: {len(peak_dbfs_values)} (1 per channel)")
        
        return spatial_coords, peak_dbfs_values, channel_labels
    
    def render_3d_visualization(self, spatial_coords, dbfs_values, channel_labels):
        """
        Create 3D Plotly visualization with unified coloraxis.
        Renders massive radiating energy spheres with persistent channel labels.
        
        Args:
            spatial_coords: Nx3 array of (x, y, z) coordinates (N=12)
            dbfs_values: N-length array of peak dBFS values [-60, 0]
            channel_labels: N-length array of channel names
        """
        print("Rendering 3D visualization...")
        
        # Normalize dBFS from [-60, 0] to [0, 1] for marker size scaling
        normalized_sizes = (dbfs_values - (-60)) / (0 - (-60))
        # Massive marker scaling: 20 to 120 pixels
        marker_sizes = 20 + normalized_sizes * 100
        
        # Extract spatial coordinates
        x_coords = spatial_coords[:, 0]
        y_coords = spatial_coords[:, 1]
        z_coords = spatial_coords[:, 2]
        
        # Create figure with single unified coloraxis
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            text=channel_labels,
            textposition='top center',
            marker=dict(
                size=marker_sizes,
                color=dbfs_values,
                colorscale='magma',
                cmin=-60,
                cmax=0,
                showscale=True,
                colorbar=dict(
                    title="Peak dBFS",
                    thickness=15,
                    len=0.7,
                    x=1.02
                ),
                opacity=0.85,
                line=dict(width=0)
            ),
            hoverinfo='text',
            hovertext=[f"{label}<br>Peak Energy: {dbfs:.1f} dBFS" for label, dbfs in zip(channel_labels, dbfs_values)],
            showlegend=False
        ))
        
        # Configure layout
        fig.update_layout(
            title="ITU-R BS.2051 7.1.4 Spatial Audio Peak Energy Map",
            scene=dict(
                xaxis=dict(
                    title="X (Left-Right)",
                    range=[-1.5, 1.5],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                ),
                yaxis=dict(
                    title="Y (Front-Back)",
                    range=[-1.5, 1.5],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                ),
                zaxis=dict(
                    title="Z (Down-Up)",
                    range=[-1.5, 1.5],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1400,
            height=900,
            margin=dict(l=0, r=200, b=0, t=40),
            hovermode='closest',
            showlegend=False
        )
        
        # Save and display
        output_path = self.wav_path.stem + "_spatial_3d.html"
        fig.write_html(output_path)
        print(f"Visualization saved to: {output_path}")
        
        # Optionally open in browser
        try:
            fig.show()
        except Exception as e:
            print(f"Could not open browser: {e}")
    
    def process(self):
        """Execute full processing pipeline."""
        self.load_audio()
        spatial_coords, dbfs_values, channel_labels = self.compute_stft_peaks()
        self.render_3d_visualization(spatial_coords, dbfs_values, channel_labels)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="3D Spatial Audio Visualization Tool - ITU-R BS.2051 Rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py audio.wav
  python main.py audio.wav --stft-window 4096
        """
    )
    
    parser.add_argument(
        'wav_file',
        type=str,
        help='Path to input WAV file'
    )
    parser.add_argument(
        '--stft-window',
        type=int,
        default=2048,
        help='STFT window length in samples (default: 2048)'
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    wav_path = Path(args.wav_file)
    if not wav_path.exists():
        print(f"Error: File not found: {wav_path}")
        return 1
    
    if not wav_path.suffix.lower() == '.wav':
        print("Warning: File does not have .wav extension")
    
    # Process
    visualizer = SpatialAudioVisualizer(
        wav_path,
        stft_nperseg=args.stft_window
    )
    
    try:
        visualizer.process()
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
