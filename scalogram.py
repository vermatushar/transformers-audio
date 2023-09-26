import os
import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import librosa
from scipy.io import wavfile
import librosa.display


DATA_DIR = "/Users/kayle/Projects/Python/audio/data_dir/all"
OUTPUT_DIR = "/Users/kayle/Projects/Python/audio/data_dir/cwt_scalograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_cwt_scalogram(audio_file_path, output_image_path, scales=np.arange(1, 128), dpi=100):
    """
    Create a CWT scalogram from an audio file and save it as a PNG image.

    Args:
        audio_file_path (str): Path to the audio file.
        output_image_path (str): Path to save the scalogram image as a PNG.
        scales (array-like): List of scales for the CWT.
        dpi (int): DPI (dots per inch) for the saved image.

    Returns:
        None
    """
    audio_data, _ = librosa.load(audio_file_path)
    cwt_coefficients, frequencies = pywt.cwt(audio_data, scales, 'morl')

    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    ax = plt.axes()

    # Display the CWT scalogram
    plt.imshow(np.abs(cwt_coefficients), extent=[0, len(audio_data), frequencies[-1], frequencies[0]],
               cmap=cm.viridis, aspect='auto', interpolation='bilinear')

    # Customize the plot
    #plt.title('Continuous Wavelet Transform Scalogram')
    #plt.xlabel('Time (samples)')
    #plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Magnitude')
    plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight')

    # Close the plot to release resources
    plt.close()


audio_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
for audio_file in audio_files:
    audio_file_path = os.path.join(DATA_DIR, audio_file)
    output_image_path = os.path.join(OUTPUT_DIR, f'{os.path.splitext(audio_file)[0]}.png')

    create_cwt_scalogram(audio_file_path, output_image_path)

print(f'Scalograms generated for {len(audio_files)} audio files.')