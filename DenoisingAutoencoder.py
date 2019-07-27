PREPROCESS = 1
TRAIN_DENOISE = 0
DENOISE = 1
CLEAR = 0


import sys
sys.path.append('lib')
import AVHandler as avh
import AVPreprocess as avp
from scipy import fft, arange
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/dot.exe'
import wave
import math
import cv2
import librosa
from sklearn.model_selection import train_test_split
np.seterr(divide='ignore', invalid='ignore')
import gc

if PREPROCESS:

    def frequency_sepectrum(x, sf):
        """
        Derive frequency spectrum of a signal from time domain
        :param x: signal in the time domain
        :param sf: sampling frequency
        :returns frequencies and their content distribution
        """
        x = x - np.average(x)  # zero-centering

        n = len(x)
        print(n)
        k = arange(n)
        tarr = n / float(sf)
        frqarr = k / float(tarr)  # two sides frequency range

        frqarr = frqarr[range(n // 2)]  # one side frequency range

        x = fft(x) / n  # fft computing and normalization
        x = x[range(n // 2)]

        return frqarr, abs(x)

    def psnr(signal1, signal2):
        mse = np.mean((signal1 - signal2) ** 2)
        if mse == 0:
            return 100
    
        maxval = 65536
        return 20 * math.log10(maxval / math.sqrt(mse))

    def batch(l, group_size):
        """
        :param l:           list
        :param group_size:  size of each group
        :return:            Yields successive group-sized lists from l.
        """
        for i in range(0, len(l), group_size):
            yield l[i:i+group_size]


    here_path = os.path.dirname(os.path.realpath(__file__))
    wav_file_name_1 = 'bowstring/clean/clean (13).wav'
    wav_file_name_2 = 'bowstring/noise only/noise.wav'
    wave_file_path_1 = os.path.join(here_path, wav_file_name_1)
    wave_file_path_2 = os.path.join(here_path, wav_file_name_2)
    sr, signal1 = wavfile.read(wave_file_path_1)
    sr, signal2 = wavfile.read(wave_file_path_2)

    y1 = signal1[:, 0]  # use the first channel (or take their average, alternatively)
    y2 = signal2[:, 0]
    t1 = np.arange(len(y1)) / float(sr)
    t2 = np.arange(len(y2)) / float(sr)

    for yy1 in batch(y1, group_size = 22050):
        print(yy1)
    for yy2 in batch(y2, group_size = 22050):
        print(yy2)


    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t1, y1, t2, y2)
    plt.xlabel('t')
    plt.ylabel('y')
    mse = np.mean((yy1[:, np.newaxis] - yy2) ** 2)
    print('mse = ', mse)
    #rmse = math.sqrt(mse)
    #print('rmse = ', rmse)
    d = psnr(yy1[:, np.newaxis], yy2)
    print('psnr = ', d, 'dB')
    print('Sample Rate = ',sr)
    frq, X = frequency_sepectrum(y1, sr)

    # mix clean audio and noise audio
    with open('bowstring/clean/clean (13).wav', 'rb') as f:
        clean_data, clean_sr = librosa.load('bowstring/clean/clean (13).wav', sr=None)  # time series data,sample rate
    with open('bowstring/noise only/noise.wav', 'rb') as f:
        noise_data, noise_sr = librosa.load('bowstring/noise only/noise.wav', sr=None)  # time series data,sample rate

    # normalize expand the noise
    noise_max = np.max(y2)
    expand_rate = 1/noise_max
    noise_data = noise_data*expand_rate

    assert clean_sr == noise_sr
    mix_data = clean_data*0.8 + noise_data*0.2
    mix_sr = clean_sr

    # fft windowing parameter #
    fft_size = 1024
    step_size = fft_size // 3 # distance to slide along the window

    # fequency to mel parameter #
    n_mels = 40 # number of mel frequency
    start_freq = 0.0
    end_freq = 8000.0

    #split data
    mel_mix_data = avp.time_to_mel(mix_data,mix_sr,fft_size,n_mels,step_size)
    D_X = avp.real_imag_expand(mel_mix_data)

    mel_clean_data = avp.time_to_mel(clean_data,clean_sr,fft_size,n_mels,step_size,fmax=8000)
    D_y = avp.real_imag_expand(mel_clean_data)

    # separate data to train test sets
    D_X_train = avp.min_max_norm(D_X[:int(D_X.shape[0]*0.9),:])
    D_y_train = D_y[:int(D_y.shape[0]*0.9),:] / D_X[:int(D_X.shape[0]*0.9),:]
    G_max = np.max(D_y_train)
    D_y_train = D_y_train/G_max

    X_test = avp.min_max_norm(D_X[int(D_X.shape[0]*0.9):,:])
    y_test = D_y[int(D_y.shape[0]*0.9):,:] / D_X[int(D_X.shape[0]*0.9):,:]
    y_test = y_test/G_max

    X_train, X_val, y_train, y_val = train_test_split(D_X_train, D_y_train, test_size=0.15, random_state=87)

    #Train and Denoise

from keras.layers import BatchNormalization,Dropout,Dense,Input,LeakyReLU
from keras import backend as K
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.models import Model
from keras.utils import plot_model
from keras.initializers import he_normal
from keras.models import model_from_json
from keras import optimizers


if TRAIN_DENOISE:
    n_input_dim = X_train.shape[1]
    n_output_dim = y_train.shape[1]

    n_hidden1 = 2049
    n_hidden2 = 500
    n_hidden3 = 180

    InputLayer1 = Input(shape=(n_input_dim,), name="InputLayer")
    InputLayer2 = BatchNormalization(axis=1, momentum=0.6)(InputLayer1)

    HiddenLayer1_1 = Dense(n_hidden1, name="H1", activation='relu', kernel_initializer=he_normal(seed=27))(InputLayer2)
    HiddenLayer1_2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer1_1)
    HiddenLayer1_3 = Dropout(0.1)(HiddenLayer1_2)

    HiddenLayer2_1 = Dense(n_hidden2, name="H2", activation='relu', kernel_initializer=he_normal(seed=42))(HiddenLayer1_3)
    HiddenLayer2_2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer2_1)

    HiddenLayer3_1 = Dense(n_hidden3, name="H3", activation='relu', kernel_initializer=he_normal(seed=65))(HiddenLayer2_2)
    HiddenLayer3_2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer3_1)

    HiddenLayer2__1 = Dense(n_hidden2, name="H2_R", activation='relu', kernel_initializer=he_normal(seed=42))(HiddenLayer3_2)
    HiddenLayer2__2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer2__1)

    HiddenLayer1__1 = Dense(n_hidden1, name="H1_R", activation='relu', kernel_initializer=he_normal(seed=27))(HiddenLayer2__2)
    HiddenLayer1__2 = BatchNormalization(axis=1, momentum=0.6)(HiddenLayer1__1)
    HiddenLayer1__3 = Dropout(0.1)(HiddenLayer1__2)

    OutputLayer = Dense(n_output_dim, name="OutputLayer", kernel_initializer=he_normal(seed=62))(HiddenLayer1__3)

    model = Model(inputs=[InputLayer1], outputs=[OutputLayer])
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0001, amsgrad=False)
    #loss = p_loss(OutputLayer,K.placeholder())
    model.compile(loss='mse', optimizer=opt)

   # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.summary()

    tensorboard = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
    # fit the model
    hist = model.fit(X_train, y_train, batch_size=512, epochs=100, verbose=0, validation_data=([X_val], [y_val]),
                     callbacks=[tensorboard])

    #plt.figure(figsize=(10, 8))
    #plt.plot(hist.history['loss'], label='Loss')
    #plt.plot(hist.history['val_loss'], label='Val_Loss')
    #plt.legend(loc='best')
    #plt.title('Training Loss and Validation Loss')
    #plt.show()

    results = model.evaluate(X_test, y_test, batch_size=len(y_test))
    print('Test loss:%3f' % results)

    # serialize model to JSON
    model_json = model.to_json()
    avh.mkdir('model')
    with open("model/model clean/model clean (35).json", 'w') as f:
        f.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model clean/model (35).h5")
    print("Saved model to disk")
    

if DENOISE:
    # load josn and create model
    with open('model/model clean/model clean (13).json','r') as f:
        loaded_model_json = f.read()
    denoise_model = model_from_json(loaded_model_json)
    denoise_model.load_weights("model/model clean/model clean (13).h5")
    print("Loaded model from disk")

    gain = denoise_model.predict(D_X) * G_max
    M_gain = gain[:,::2]+1j*gain[:,1::2]
    F_gain = avp.mel2freq(M_gain,mix_sr,fft_size,n_mels)

    F = F_gain * avp.stft(mix_data,fft_size,step_size)
    #ratio[np.isnan(ratio)] = 0.0
    print("shape of F_out:",F.shape)
    T = avp.istft(F,fft_size,step_size)

    # write the result
    Tint = T/np.max(T)*32767
    avh.mkdir("Reconstructed")
    wavfile.write("Reconstructed/Denoise_reconstruction_clean (13).wav",mix_sr,Tint.astype('int16'))

if CLEAR:
    gc.collect()