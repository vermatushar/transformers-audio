import os
import librosa
from scipy.io import wavfile
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = "/Users/kayle/Projects/Python/audio/data_dir/all"
OUTPUT_DIR = "/Users/kayle/Projects/Python/audio/data_dir/spectrograms"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_spectrogram(audio_file, output_image_path, dpi = 100):
    freq, data = wavfile.read(audio_file)
    data = data.astype(float)

    # create spectrogram
    D = librosa.amplitude_to_db(librosa.stft(data, hop_length=128), ref=np.max)
    # create figure
    plt.figure(figsize=(10, 6))
    ax = plt.axes()

    librosa.display.specshow(D, y_axis='linear', sr=freq, hop_length=128, bins_per_octave=24)
    plt.colorbar(format='%+2.0f dB')
    # plt.title('Linear-frequency power spectrogram')
    # plt.show()
    plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    try:
        audio_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
        for audio_file in audio_files:
            audio_file_path = os.path.join(DATA_DIR, audio_file)
            output_image_path = os.path.join(OUTPUT_DIR, f'{os.path.splitext(audio_file)[0]}.png')

            create_spectrogram(audio_file_path, output_image_path)

        print(f'Spectrograms generated for {len(audio_files)} audio files.')
    except Exception as ex:
        pass

