import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import soundfile as sf

from SpectogramReconstructorParameter import *


def LoadAudio(AudioFilePath):
    return librosa.load(AudioFilePath)


def GenerateMelSpectrogram(Audio, SampleRate):
    Spectrogram = librosa.feature.melspectrogram(y=Audio, sr=SampleRate)
    return librosa.power_to_db(Spectrogram, ref=np.max)


def SaveSpectrogramImage(SpectrogramDB, SampleRate, OutputPath):
    plt.figure(figsize=FigureSize)
    librosa.display.specshow(SpectrogramDB, sr=SampleRate, x_axis='time', y_axis='mel')
    plt.colorbar(format=ColorMapFormat)
    plt.title(SpectrogramTitle)
    plt.tight_layout()
    plt.savefig(OutputPath)
    plt.close()


def ApplyImageFilters(SpectrogramDB):
    NormalizedImage = cv2.normalize(SpectrogramDB, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    BlurredImage = cv2.GaussianBlur(NormalizedImage, GaussianKernelSize, 0)
    SharpenedImage = cv2.filter2D(BlurredImage, -1, SharpenKernel)
    EdgeImage = cv2.Canny(SharpenedImage, CannyThreshold1, CannyThreshold2)
    return SharpenedImage, EdgeImage


def ReconstructAudioFromSpectrogram(SpectrogramDBModified, SampleRate):
    PowerSpectrogram = librosa.db_to_power(SpectrogramDBModified)
    return librosa.feature.inverse.mel_to_audio(PowerSpectrogram, sr=SampleRate)


def EnsureOutputDirectory(AudioFileName):
    OutputDirectory = os.path.join(OutputFolder, AudioFileName)
    os.makedirs(OutputDirectory, exist_ok=True)
    return OutputDirectory


def SaveOutputFiles(Audio, SampleRate, AudioFileName, OriginalAudioPath, SpectrogramDB, ModifiedSpectrogram, Edges):
    OutputDirectory = EnsureOutputDirectory(AudioFileName)

    OriginalSpectrogramPath = os.path.join(OutputDirectory, "original_spectrogram.png")
    SaveSpectrogramImage(SpectrogramDB, SampleRate, OriginalSpectrogramPath)

    cv2.imwrite(os.path.join(OutputDirectory, "modified_spectrogram.png"), ModifiedSpectrogram)
    cv2.imwrite(os.path.join(OutputDirectory, "edges_spectrogram.png"), Edges)

    OriginalAudioCopyPath = os.path.join(OutputDirectory, "original_audio.wav")
    sf.write(OriginalAudioCopyPath, Audio, SampleRate)

    ReconstructedAudio = ReconstructAudioFromSpectrogram(SpectrogramDB, SampleRate)
    ReconstructedAudioPath = os.path.join(OutputDirectory, "reconstructed_audio.wav")
    sf.write(ReconstructedAudioPath, ReconstructedAudio, SampleRate)

    print(f"Files has been generated in path {OutputDirectory}")


def Main():
    if len(sys.argv) < 2:
        print("The audio input file must be inside the folder 'Input'")
        sys.exit(1)

    AudioFileName = sys.argv[1]
    AudioFilePath = os.path.join(InputFolder, AudioFileName)

    if not os.path.exists(AudioFilePath):
        print(f"File '{AudioFileName}' does not found.")
        sys.exit(1)

    Audio, SampleRate = LoadAudio(AudioFilePath)
    SpectrogramDB = GenerateMelSpectrogram(Audio, SampleRate)
    ModifiedSpectrogram, Edges = ApplyImageFilters(SpectrogramDB)
    SaveOutputFiles(Audio, SampleRate, os.path.splitext(AudioFileName)[0], AudioFilePath, SpectrogramDB,
                    ModifiedSpectrogram, Edges)


if __name__ == '__main__':
    Main()
