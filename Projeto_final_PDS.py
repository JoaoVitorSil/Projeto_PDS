# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch
from scipy import fftpack

#Passo a)

#Carrega o arquivo
samplerate, data = wavfile.read('257210__xtrgamr__stereo-mic-test-guitar.wav')

#Carrega o arquivo em dois canais (audio estereo)
print(f"Número de canais = {data.shape[1]}")
x = data.shape[0]
#Tempo total = numero de amostras / fs
length = data.shape[0] / samplerate
print(f"Duração = {length}s")

#Interpola para determinar eixo do tempo
time = np.linspace(0., length, data.shape[0])

#Plota os canais esquerdo e direito
plt.figure(1)
plt.plot(time, data[:, 0])
plt.title("Conteúdo temporal canal esquerdo")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.show()

plt.figure(2)
plt.plot(time, data[:, 1])
plt.title("Conteúdo temporal canal direito")
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.show()

#Passo B

#Plota espectro usando funcao FFT (CANAL ESQUERDO)

nfft=512
fft_esq = abs(fftpack.fft(data[:,0]))
freqfft_esq = fftpack.fftfreq(nfft, (1/samplerate))

plt.figure(3)
#plt.plot(freqfft_esq[range(nfft//2)], np.log10(fft_esq[range(nfft//2)])*20)
plt.plot(freqfft_esq[range(-nfft//2,nfft//2)], np.abs(fftpack.fftshift(np.log10(fft_esq[range(-nfft//2,nfft//2)])*20)))
plt.title('FFT canal esquerdo')
plt.xlabel('frequencia [Hz]')
plt.ylabel('Amplitudes [dB]')
plt.show()

#Plota espectro usando funcao FFT (CANAL DIREITO)

nfft=512
fft_dir = abs(fftpack.fft(data[:,1]))
freqfft_dir = fftpack.fftfreq(nfft, (1/samplerate))

plt.figure(4)
plt.plot(freqfft_dir[range(-nfft//2,nfft//2)], np.abs(fftpack.fftshift(np.log10(fft_dir[range(-nfft//2,nfft//2)])*20)))
plt.title('FFT canal direito')
plt.xlabel('frequencia [Hz]')
plt.ylabel('Amplitudes [dB]')
plt.show()


#Passo D

#Carregando filtro FIR
b = np.genfromtxt('FilterFIR_LP.csv', delimiter=',')

def pad_zeros_to(x, new_length):
    output = np.zeros((new_length,))
    output[:x.shape[0]] = x
    return output

def next_power_of_2(n):
    return 1 << (int(np.log2(n - 1)) + 1)

def fft_convolution(x, h, K=None):
    Nx = x.shape[0]
    Nh = h.shape[0]
    Ny = Nx + Nh - 1 

    if K is None:
        K = next_power_of_2(Ny)

    X = np.fft.fft(pad_zeros_to(x, K))
    H = np.fft.fft(pad_zeros_to(h, K))

    Y = X*H
    
    y = np.real(np.fft.ifft(Y))

    return y[:Ny]

def overlap_add_convolution(x, h, B, K=None):
    M = len(x)
    N = len(h)

    num_input_blocks = np.ceil(M / B).astype(int)

    xp = np.zeros((num_input_blocks*B,))
    xp[:x.shape[0]] = x

    output_size = num_input_blocks * B + N - 1
    y = np.zeros((output_size,))
    
    for n in range(num_input_blocks):
        
        xb = xp[n*B:(n+1)*B]
        
        u = fft_convolution(xb, h, K)

        y[n*B:n*B+len(u)] += u

    return  y[:M+N-1]


#Realiza o cálculo da convolução
convolucao_esq = overlap_add_convolution(data[:,0], b, 1000)
convolucao_dir = overlap_add_convolution(data[:,1], b, 1000)

#Estima o espectro do sinal utilizando a funcao welch
f_esq, Pxx_esq = welch(data[:,0], samplerate, 'flattop', 1024, scaling='spectrum')
f_esq_filtered, Pxx_esq_filtered = welch(
    convolucao_esq, samplerate, 'flattop', 1024, scaling='spectrum')

# Plota o espectro do sinal para frequencias normalizadas entre 0 e pi
plt.figure(5)
plt.semilogy(f_esq, Pxx_esq, 'b-', label='Original')
plt.semilogy(f_esq_filtered, Pxx_esq_filtered, 'y-', label='Filtrado')
plt.title('Espectro de frequência do lado esquerdo')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()

#Estima o espectro do sinal utilizando a funcao welch
f_dir, Pxx_dir = welch(data[:,1], samplerate, 'flattop', 1024, scaling='spectrum')
f_dir_filtered, Pxx_dir_filtered = welch(
    convolucao_dir, samplerate, 'flattop', 1024, scaling='spectrum')

# Plota o espectro do sinal para frequencias normalizadas entre 0 e pi
plt.figure(6)
plt.semilogy(f_dir, Pxx_dir, 'b-', label='Original')
plt.semilogy(f_dir_filtered, Pxx_dir_filtered, 'y-', label='Filtrado')
plt.title('Espectro de frequência do lado direito')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()

#Passo E

#Interpolação dos canais

#Expansão dos sinais por L

L = 2

exp_esq = np.zeros(len(data[:,0]))
exp_dir = np.zeros(len(data[:,1]))

aux1 = 0
aux2 = 0


for i in range(0,len(data[:,0]),L):
    exp_esq[i] = data[:,0][aux1]
    aux1+=1
    
for i in range(0,len(data[:,1]),L):
    exp_dir[i] = data[:,1][aux2]
    aux2+=1


#Estima o espectro do sinal utilizando a funcao welch
f_esq, Pxx_esq = welch(data[:,0], samplerate, 'flattop', 1024, scaling='spectrum')
f_esq_exp, Pxx_esq_exp = welch(
    exp_esq, samplerate, 'flattop', 1024, scaling='spectrum')

# Plota o espectro do sinal para frequencias normalizadas entre 0 e pi
plt.figure(7)
plt.semilogy(f_esq, Pxx_esq, 'b-', label='Original')
plt.semilogy(f_esq_exp, Pxx_esq_exp, 'r-', label='Expandido')
plt.title('Espectro de frequência do lado esquerdo')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()

#Estima o espectro do sinal utilizando a funcao welch
f_dir, Pxx_dir = welch(data[:,1], samplerate, 'flattop', 1024, scaling='spectrum')
f_dir_exp, Pxx_dir_exp = welch(
    exp_dir, samplerate, 'flattop', 1024, scaling='spectrum')

# Plota o espectro do sinal para frequencias normalizadas entre 0 e pi
plt.figure(8)
plt.semilogy(f_dir, Pxx_dir, 'b-', label='Original')
plt.semilogy(f_dir_exp, Pxx_dir_exp, 'r-', label='Expandido')
plt.title('Espectro de frequência do lado direito')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()

#Realizando a interpolação
interpolacao_esq = overlap_add_convolution(exp_esq, L*b, 1000)
interpolacao_dir = overlap_add_convolution(exp_dir, L*b, 1000)

#Estima o espectro do sinal utilizando a funcao welch
f_esq, Pxx_esq = welch(data[:,0], samplerate, 'flattop', 1024, scaling='spectrum')
f_esq_inter, Pxx_esq_inter = welch(
    interpolacao_esq, samplerate, 'flattop', 1024, scaling='spectrum')

# Plota o espectro do sinal para frequencias normalizadas entre 0 e pi
plt.figure(9)
plt.semilogy(f_esq, Pxx_esq, 'b-', label='Original')
plt.semilogy(f_esq_inter, Pxx_esq_inter, 'g-', label='Interpolado')
plt.title('Espectro de frequência do lado esquerdo')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()

#Estima o espectro do sinal utilizando a funcao welch
f_dir, Pxx_dir = welch(data[:,1], samplerate, 'flattop', 1024, scaling='spectrum')
f_dir_inter, Pxx_dir_inter = welch(
    interpolacao_dir, samplerate, 'flattop', 1024, scaling='spectrum')

# Plota o espectro do sinal para frequencias normalizadas entre 0 e pi
plt.figure(10)
plt.semilogy(f_dir, Pxx_dir, 'b-', label='Original')
plt.semilogy(f_dir_inter, Pxx_dir_inter, 'g-', label='Interpolado')
plt.title('Espectro de frequência do lado direito')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()

#Passo F

#Plote no domínio do tempo
lengthstem = np.linspace(0,0.009,40)


plt.figure(11)
plt.stem(lengthstem, data[:,0][range(0,40)])
plt.title('Conteúdo no tempo do lado esquerdo original')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.show()

plt.figure(12)
plt.stem(lengthstem, exp_esq[range(0,40)])
plt.title('Conteúdo no tempo do lado esquerdo expandido')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.show()

plt.figure(13)
plt.stem(lengthstem, data[:,1][range(0,40)])
plt.title('Conteúdo no tempo do lado direito original')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.show()

plt.figure(14)
plt.stem(lengthstem, exp_dir[range(0,40)])
plt.title('Conteúdo no tempo do lado direito expandido')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.show()

plt.figure(12)
plt.stem(lengthstem, interpolacao_esq[range(0,40)])
plt.title('Conteúdo no tempo do lado esquerdo interpolado')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.show()

plt.figure(14)
plt.stem(lengthstem, interpolacao_dir[range(0,40)])
plt.title('Conteúdo no tempo do lado direito interpolado')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.show()

length2_esq = exp_esq.shape[0] / samplerate
time2_esq = np.linspace(0,length2_esq,exp_esq.shape[0])

plt.figure(15)
plt.plot(time, data[:,1], 'b-', label='Original')
plt.plot(time2_esq, exp_esq, 'r-', label='Expandido')
plt.title('Contéudo no tempo do lado direito')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()


length2_dir = exp_dir.shape[0] / samplerate
time2_dir = np.linspace(0,length2_dir,exp_dir.shape[0])

plt.figure(16)
plt.plot(time, data[:,1], 'b-', label='Original')
plt.plot(time2_dir, exp_dir, 'r-', label='Expandido')
plt.title('Contéudo no tempo do lado direito')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

length3_esq = interpolacao_esq.shape[0] / samplerate
time3_esq = np.linspace(0,length3_esq,interpolacao_esq.shape[0])

plt.figure(17)
plt.plot(time, data[:,0], 'b-', label='Original')
plt.plot(time3_esq, interpolacao_esq, 'g-', label='Interpolado')
plt.title('Conteúdo no tempo do lado esquerdo')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

length3_dir = interpolacao_dir.shape[0] / samplerate
time3_dir = np.linspace(0,length2_dir,interpolacao_dir.shape[0])

plt.figure(18)
plt.plot(time, data[:,1], 'b-', label='Original')
plt.plot(time3_dir, interpolacao_dir, 'g-', label='Interpolado')
plt.title('Conteúdo no tempo do lado direito')
plt.xlabel('Tempo [s]')
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()

#Plote no domínio da frequencia

plt.figure(19)
plt.semilogy(f_esq, Pxx_esq, 'b-', label='Original')
plt.semilogy(f_esq_exp, Pxx_esq_exp, 'r-', label='Expandido')
plt.semilogy(f_esq_inter, Pxx_esq_inter, 'g-', label='Interpolado')
plt.title('Espectro de frequência do lado direito')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()

plt.figure(20)
plt.semilogy(f_dir, Pxx_dir, 'b-', label='Original')
plt.semilogy(f_dir_exp, Pxx_dir_exp, 'r-', label='Expandido')
plt.semilogy(f_dir_inter, Pxx_dir_inter, 'g-', label='Interpolado')
plt.title('Espectro de frequência do lado direito')
plt.xlabel('Frequencia [rad/($\pi$)]')
plt.ylabel('Espectro')
plt.grid()
plt.legend()
plt.show()