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
    
    def __init__(self, wav_path, stft_nperseg=2048, temporal_downsample=4):
        """
        Initialize the spatial audio visualizer.
        
        Args:
            wav_path: Path to input WAV file
            stft_nperseg: STFT window length
            temporal_downsample: Temporal downsampling factor to reduce point count
        """
        self.wav_path = Path(wav_path)
        self.stft_nperseg = stft_nperseg
        self.temporal_downsample = temporal_downsample
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
    
    def _normalize_dbfs(self, dbfs_values, floor_db=-60):
        """
        Normalize negative dBFS values to positive linear range for marker sizes.
        Maps [floor_db, 0] dB to [0, 1].
        
        Args:
            dbfs_values: Array of dBFS values (typically -60 to 0)
            floor_db: Noise floor in dB (default -60)
        
        Returns:
            Normalized positive values [0, 1]
        """
        clipped = np.clip(dbfs_values, floor_db, 0)
        normalized = (clipped - floor_db) / (-floor_db)
        return normalized
    
    def compute_stft_peaks(self):
        """
        Compute STFT for all channels and extract peak energy per spatial coordinate.
        Returns a heavily downsampled dataset suitable for 3D visualization.
        """
        print("Computing STFT for all channels...")
        
        # Normalize audio to prevent overflow in STFT
        audio_normalized = self.audio_data.astype(np.float32) / np.iinfo(self.audio_data.dtype).max
        
        # Pre-allocate results
        spatial_coords = []
        dbfs_values = []
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
            
            # Temporal downsampling to reduce point count
            dbfs_downsampled = dbfs[:, ::self.temporal_downsample]
            
            # Extract peak energy across frequency bins at each time step
            peak_dbfs = np.max(dbfs_downsampled, axis=0)
            
            # Duplicate spatial coordinates for each temporal point
            num_time_points = peak_dbfs.shape[0]
            spatial_coords.extend([(x, y, z)] * num_time_points)
            dbfs_values.extend(peak_dbfs.tolist())
            channel_labels.extend([channel_name] * num_time_points)
            
            print(f"  Channel {ch_idx:2d} ({channel_name:4s}): Azimuth={azimuth:4.0f}°, Elevation={elevation:+3.0f}°")
        
        # Convert to numpy arrays
        spatial_coords = np.array(spatial_coords)
        dbfs_values = np.array(dbfs_values)
        
        print(f"Total visualization points: {len(dbfs_values)}")
        
        return spatial_coords, dbfs_values, channel_labels
    
    def render_3d_visualization(self, spatial_coords, dbfs_values, channel_labels):
        """
        Create 3D Plotly visualization with unified coloraxis.
        
        Args:
            spatial_coords: Nx3 array of (x, y, z) coordinates
            dbfs_values: N-length array of dBFS values
            channel_labels: N-length array of channel names
        """
        print("Rendering 3D visualization...")
        
        # Normalize dBFS to positive range for marker sizes
        normalized_sizes = self._normalize_dbfs(dbfs_values, floor_db=-60)
        # Scale sizes for visibility (range 2 to 12)
        marker_sizes = 2 + normalized_sizes * 10
        
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
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=dbfs_values,  # Raw dBFS values for colorbar
                colorscale='magma',  # Aggressive heatmap
                showscale=True,
                colorbar=dict(
                    title="dBFS",
                    thickness=15,
                    len=0.7,
                    x=1.02
                ),
                opacity=0.8,
                line=dict(width=0)
            ),
            text=[f"{label}<br>dBFS: {dbfs:.1f}" for label, dbfs in zip(channel_labels, dbfs_values)],
            hoverinfo='text',
            name='Spatial Audio Energy'
        ))
        
        # Configure layout
        fig.update_layout(
            title="ITU-R BS.2051 7.1.4 Spatial Audio Visualization",
            scene=dict(
                xaxis=dict(
                    title="X (Left-Right)",
                    range=[-1.2, 1.2],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                ),
                yaxis=dict(
                    title="Y (Front-Back)",
                    range=[-1.2, 1.2],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                ),
                zaxis=dict(
                    title="Z (Down-Up)",
                    range=[-1.2, 1.2],
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
            showlegend=True
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
  python main.py audio.wav --stft-window 4096 --temporal-downsample 8
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
    parser.add_argument(
        '--temporal-downsample',
        type=int,
        default=4,
        help='Temporal downsampling factor (default: 4)'
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
        stft_nperseg=args.stft_window,
        temporal_downsample=args.temporal_downsample
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
