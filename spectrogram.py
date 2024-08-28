import argparse
import sys
import numpy as np 
import sounddevice as sd
import librosa
from sklearn.preprocessing import StandardScaler
from tflite_runtime.interpreter import Interpreter
import tensorflow as tf


def int_or_str(text):
    try:
        return int(text)
    except:
        return text
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action= 'store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser =argparse.ArgumentParser(
    description = __doc__,
    formatter_class = argparse.RawDescriptionHelpFormatter,
    parents = [parser])
parser.add_argument(
    'channels', type = int, default=[1], nargs='*', metavar = 'CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type = int_or_str,
    help = 'input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=2, metavar='DURATION',
    help='visile time slot (default: %(default)s, ms)')
parser.add_argument(
    '-i','--interval', type=float, default=30,
    help= 'minimum time between plot updates(default: %(default)s ms)')
parser.add_argument(
    '-b','--blocksize', type=int,
    help= 'block size (in samples)')
parser.add_argument(
    '-r','--samplerate', type=float, default= 44100, help= 'sampling rate of audio device')
parser.add_argument(
    '-n','--downsample', type=float, default=10, metavar = 'N',
    help= 'display Nth sample (default: %(default)s s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >=1')
sc = StandardScaler()
spectrogram = None
model_path = "snoring.tflite"
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
row, col = interpreter.get_input_details()[0]['shape']
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
features = np.load("features_new.npy")
features = features[:10]
sc.fit(features)

def processAudio(X, sc):
    stft = np.abs(librosa.stft(X))
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=args.samplerate, n_mfcc = 8).T)
    chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=args.samplerate).T)
    mel = np.array(librosa.feature.melspectrogram(y=X, sr=args.samplerate).T)
    contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=args.samplerate).T)
    tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=args.samplerate).T)
    features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    input  = np.array(sc.transform(features))
    return input

def runInference(interpreter, input):
    prob_out = np.empty((0,11))
    for i in range(len(input)):
        interpreter.set_tensor(input_details[0]['index'], tf.cast(input[i,:].reshape(1,161), tf.float32))
        interpreter.invoke()
        prob_out = np.vstack([prob_out, interpreter.get_tensor(output_details[0]['index'])[0]])
        cls_out = np.bincount(np.argmax(prob_out, axis=1)).argmax()
    return cls_out
    

def audio_callback(indata, frames, time, status):
    # global spectrogram
    global window
    if status:
        print(status, file=sys.stderr)

    indata = np.squeeze(indata)
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = indata

    ############################OPTION1###################################
    # spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=window, sr=args.samplerate,hop_length =512))
    # spectrogram = abs(spectrogram)
    # means = np.mean(spectrogram, axis = 0, keepdims = True)
    # stddevs = np.std(spectrogram, axis = 0, keepdims = True)
    # spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    ######################################################################

    ############################OPTION2###################################
    #Process spectrogram in the stream loop.

def process_audio(window):
    spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=window, sr=args.samplerate,hop_length =512))
    spectrogram = abs(spectrogram)
    means = np.mean(spectrogram, axis = 0, keepdims = True)
    stddevs = np.std(spectrogram, axis = 0, keepdims = True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    return spectrogram 
    
try: 
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']
    if args.window is None: 
        device_info = sd.query_devices(args.device, 'input')
        args.window = device_info['default_window']
    window = np.zeros(int(args.window*args.samplerate)*2)
    stream = sd.InputStream(
        device = args.device, channels = max(args.channels), 
        samplerate = args.samplerate, callback = audio_callback, blocksize = int(args.samplerate *args.window))

    with stream:
        while True:
            ##########################OPTION1#####################################
            # if spectrogram is not None:
            #     pass
            #     #run inference 
            ######################################################################

        ##############################OPTION2######################################
            input = processAudio(window, sc)
            output = runInference(interpreter, input)
            print(output)

            #run inference 
            

except Exception as e:
    parser.exit(type(e).__name__+': '+ str(e))