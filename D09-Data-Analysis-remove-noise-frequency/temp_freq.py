import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from get_data import Data

# Load the data

mat = []
for i in range(1, 7):
    line = []
    for j in range(1, 7):
        line.append(Data(i, j))
    mat.append(line)


def plot_Bode(subject=0, condition=0):
    a, b = subject, condition
    # Load a condition
    ft = mat[a][b].ft
    u = mat[a][b].u
    t = mat[a][b].t


    # Get magnitude and phase of numbers in Hpe_FC
    Hpe_FC = mat[a][b].Hpe_FC
    hper = np.real(Hpe_FC)
    hpec = np.imag(Hpe_FC)
    magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
    phase_Hpe = np.angle(Hpe_FC, deg=True)
    # Get magnitude and phase of numbers in Hpxd_FC
    Hpxd_FC = mat[a][b].Hpxd_FC
    hpxdr = np.real(Hpxd_FC)
    hpxdc = np.imag(Hpxd_FC)
    magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
    phase_Hpxd = np.angle(Hpxd_FC, deg=True)


    for i in range(len(phase_Hpe)):
        if i != len(phase_Hpe) - 1:
            if abs(phase_Hpe[i] - phase_Hpe[i + 1]) >= 180:
                #for j in range(i + 1, len (phase_Hpe)):
                phase_Hpe[i + 1] -= 360

    for i in range(len(phase_Hpxd)):
        if i != len(phase_Hpxd) - 1:
            if abs(phase_Hpxd[i] - phase_Hpxd[i + 1]) >= 180:
                #for j in range(i + 1, len (phase_Hpe)):
                phase_Hpxd[i + 1] -= 360
                
            if phase_Hpxd[i] * phase_Hpxd[i + 1] < 0:
                print(phase_Hpxd[i], phase_Hpxd[i + 1])


    w_FC = mat[a][b].w_FC
    # Create a figure with two subplots
    fig, axs = plt.subplots(2)

    # Plot the magnitude of Hpe_FC in the first subplot
    axs[0].semilogx(w_FC, magnitude_Hpe, label='Hpe')
    axs[0].semilogx(w_FC, magnitude_Hpxd, label='Hpxd')
    axs[0].semilogx(w_FC, (magnitude_Hpe + magnitude_Hpxd) / 2, label='Average')
    axs[0].set_title('Magnitude of Hpe_FC')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid()
    axs[0].legend()

    # Plot the phase of Hpe_FC in the second subplot
    axs[1].semilogx(w_FC, phase_Hpe, label='Hpe')
    axs[1].semilogx(w_FC, phase_Hpxd, label='Hpxd')
    axs[1].semilogx(w_FC, (phase_Hpe + phase_Hpxd) / 2, label='Average')
    axs[1].set_title('Phase of Hpe_FC')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Phase (degrees)')
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def plot_FFT(subject=0, condition=0):
    a, b = subject, condition
    # Load a condition
    ft = mat[a][b].ft
    u = mat[a][b].u
    t = mat[a][b].t

    # Compute the FFT of both signals
    fft_result_ft = np.fft.fft(ft[:, 0])
    fft_result_u = np.fft.fft(u[:, 0])

    # Assuming uniform time sampling, compute the frequency axis
    dt = t[0, 1] - t[0, 0]
    N = len(ft)  # or len(u), assuming both have the same length
    frequencies = np.fft.fftfreq(N, dt)

    # Keep only positive frequencies and corresponding FFT coefficients
    positive_frequencies = frequencies[:N//2]
    positive_fft_result_ft = fft_result_ft[:N//2]
    positive_fft_result_u = fft_result_u[:N//2]

    # Compute the difference in frequencies
    difference_in_frequencies = positive_frequencies

    # Perform noise cancellation
    noise_cancelled_result = np.abs(positive_fft_result_u) - np.abs(positive_fft_result_ft)

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3)

    # Plot the Fourier transform of the original signal in the first subplot
    axs[0].plot(positive_frequencies, np.abs(positive_fft_result_u))
    axs[0].set_title('Original Fourier Transform')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid()
    axs[0].set_xlim(0, 5)

    # Plot the Fourier transform of the noise in the second subplot
    axs[1].semilogy(positive_frequencies, np.abs(positive_fft_result_ft))
    axs[1].set_title('Noise Fourier Transform')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude (log scale)')
    axs[1].grid()
    axs[1].set_xlim(0, 5)

    # Plot the Fourier transform of the noise-cancelled signal in the third subplot
    axs[2].plot(difference_in_frequencies, np.abs(noise_cancelled_result))
    axs[2].set_xlim(0, 5)
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.show()

def plot_6_pilots(condition=0):
    fig, axs = plt.subplots(2)
    w_FC = mat[0][condition].w_FC

    for i in range(0, 6):
        # Get magnitude and phase of numbers in Hpe_FC
        Hpe_FC = mat[i][condition].Hpe_FC
        hper = np.real(Hpe_FC)
        hpec = np.imag(Hpe_FC)
        magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
        phase_Hpe = np.angle(Hpe_FC, deg=True)
        # Get magnitude and phase of numbers in Hpxd_FC
        Hpxd_FC = mat[i][condition].Hpxd_FC
        hpxdr = np.real(Hpxd_FC)
        hpxdc = np.imag(Hpxd_FC)
        magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
        phase_Hpxd = np.angle(Hpxd_FC, deg=True)

        for j in range(len(phase_Hpe)):
            if j != len(phase_Hpe) - 1:
                if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                    phase_Hpe[j + 1] -= 360
                if abs(phase_Hpxd[j] - phase_Hpxd[j + 1]) >= 180:
                    phase_Hpxd[j + 1] -= 360
        
        axs[0].semilogx(w_FC, magnitude_Hpe, label=f'Hpe {i + 1}')
        axs[0].semilogx(w_FC, magnitude_Hpxd, label=f'Hpdx {i + 1}')
        #axs[0].semilogx(w_FC, (magnitude_Hpe + magnitude_Hpxd) / 2, label='Average')
        axs[0].set_title('Magnitude of Hpe_FC')
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Magnitude')
        axs[0].grid()
        axs[0].legend()

        # Plot the phase of Hpe_FC in the second subplot
        axs[1].semilogx(w_FC, phase_Hpe, label=f'Hpe {i + 1}')
        axs[1].semilogx(w_FC, phase_Hpxd, label=f'Hpxd {i + 1}')
        #axs[1].semilogx(w_FC, (phase_Hpe + phase_Hpxd) / 2, label='Average')
        axs[1].set_title('Phase of Hpe_FC')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Phase (degrees)')
        axs[1].grid()
        axs[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_motion_nomotion_average():
    fig, axs = plt.subplots(2, 2)
    w_FC = mat[0][0].w_FC

    # First get average magnitude and phase of all pilots in conditions 0-2, only use Hpe
    for condition in range(0, 3):
        for pilot in range(0, 6):
            Hpe_FC = mat[pilot][condition].Hpe_FC
            hper = np.real(Hpe_FC)
            hpec = np.imag(Hpe_FC)
            magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
            phase_Hpe = np.angle(Hpe_FC, deg=True)
            for j in range(len(phase_Hpe)):
                if j != len(phase_Hpe) - 1:
                    if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                        phase_Hpe[j + 1] -= 360


            if pilot == 0:
                magnitude_Hpe_avg = magnitude_Hpe
                phase_Hpe_avg = phase_Hpe
            else:
                magnitude_Hpe_avg += magnitude_Hpe
                phase_Hpe_avg += phase_Hpe
            
        average_magnitude_NM = magnitude_Hpe_avg / 6
        average_phase_NM = phase_Hpe_avg / 6
        # Plot the average magnitude of Hpe_FC in the first subplot [0, 0], plot average phase in [0, 1] 
        axs[0, 0].semilogx(w_FC, average_magnitude_NM, label=f'Condition {condition + 1}')
        axs[0, 1].semilogx(w_FC, average_phase_NM, label=f'Condition {condition + 1}')

    # Get average magnitude and phase of all pilots in conditions 3-5, use Hpe and Hpxd
    for condition in range(3, 6):
        for pilot in range(0, 6):
            Hpe_FC = mat[pilot][condition].Hpe_FC
            hper = np.real(Hpe_FC)
            hpec = np.imag(Hpe_FC)
            magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
            phase_Hpe = np.angle(Hpe_FC, deg=True)
            Hpxd_FC = mat[pilot][condition].Hpxd_FC
            hpxdr = np.real(Hpxd_FC)
            hpxdc = np.imag(Hpxd_FC)
            magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
            phase_Hpxd = np.angle(Hpxd_FC, deg=True)
            for j in range(len(phase_Hpe)):
                if j != len(phase_Hpe) - 1:
                    if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                        phase_Hpe[j + 1] -= 360
                    if abs(phase_Hpxd[j] - phase_Hpxd[j + 1]) >= 180:
                        phase_Hpxd[j + 1] -= 360
            
            if pilot == 0:
                magnitude_Hpe_avg = magnitude_Hpe
                magnitude_Hpxd_avg = magnitude_Hpxd
                phase_Hpe_avg = phase_Hpe
                phase_Hpxd_avg = phase_Hpxd
            else:
                magnitude_Hpe_avg += magnitude_Hpe
                magnitude_Hpxd_avg += magnitude_Hpxd
                phase_Hpe_avg += phase_Hpe
                phase_Hpxd_avg += phase_Hpxd
            
        average_magnitude_hpe_M = magnitude_Hpe_avg / 6
        average_magnitude_hpxd_M = magnitude_Hpxd_avg / 6
        average_phase_hpe_M = phase_Hpe_avg / 6
        average_phase_hpxd_M = phase_Hpxd_avg / 6
        # Plot the average magnitude of Hpe_FC in the first subplot [1, 0], plot average phase in [1, 1] 
        axs[1, 0].semilogx(w_FC, average_magnitude_hpe_M, label=f'Hpe {condition + 1}', color='red')
        axs[1, 0].semilogx(w_FC, average_magnitude_hpxd_M, label=f'Hpxd {condition + 1}', color='blue')
        axs[1, 1].semilogx(w_FC, average_phase_hpe_M, label=f'Hpe {condition + 1}', color='red')
        axs[1, 1].semilogx(w_FC, average_phase_hpxd_M, label=f'Hpxd {condition + 1}', color='blue')

    axs[0, 0].set_title('Magnitude NM')
    axs[0, 0].set_xlabel('Frequency (Hz)')
    axs[0, 0].set_ylabel('Magnitude')
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[0, 1].set_title('Phase NM')
    axs[0, 1].set_xlabel('Frequency (Hz)')
    axs[0, 1].set_ylabel('Phase (degrees)')
    axs[0, 1].grid()
    axs[0, 1].legend()

    axs[1, 0].set_title('Magnitude M')
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel('Magnitude')
    axs[1, 0].grid()
    axs[1, 0].legend()

    axs[1, 1].set_title('Phase M')
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_ylabel('Phase (degrees)')
    axs[1, 1].grid()
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

def compare_6_conditions():
    fig, axs = plt.subplots(2, 3)
    w_FC = mat[0][0].w_FC

    # Position NM
    condition = 0
    for pilot in range(0, 6):
        Hpe_FC = mat[pilot][condition].Hpe_FC
        hper = np.real(Hpe_FC)
        hpec = np.imag(Hpe_FC)
        magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
        phase_Hpe = np.angle(Hpe_FC, deg=True)
        for j in range(len(phase_Hpe)):
            if j != len(phase_Hpe) - 1:
                 if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                        phase_Hpe[j + 1] -= 360

            if pilot == 0:
                magnitude_Hpe_avg = magnitude_Hpe
                phase_Hpe_avg = phase_Hpe
            else:
                magnitude_Hpe_avg += magnitude_Hpe
                phase_Hpe_avg += phase_Hpe
            
    average_magnitude_NM = magnitude_Hpe_avg / 6
    average_phase_NM = phase_Hpe_avg / 6
    # Plot the average magnitude of Hpe_FC in the first subplot [0, 0], plot average phase in [0, 1] 
    # axs[0, 0].semilogx(w_FC, average_magnitude_NM, label=f'NM {condition + 1}')
    # axs[1, 0].semilogx(w_FC, average_phase_NM, label=f'NM {condition + 1}')
    axs[0, 0].set_title('Magnitude Position')
    axs[1, 0].set_title('Phase Position')

    # Velocity NM
    condition = 1
    for pilot in range(0, 6):
        Hpe_FC = mat[pilot][condition].Hpe_FC
        hper = np.real(Hpe_FC)
        hpec = np.imag(Hpe_FC)
        magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
        phase_Hpe = np.angle(Hpe_FC, deg=True)
        for j in range(len(phase_Hpe)):
            if j != len(phase_Hpe) - 1:
                 if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                        phase_Hpe[j + 1] -= 360

            if pilot == 0:
                magnitude_Hpe_avg = magnitude_Hpe
                phase_Hpe_avg = phase_Hpe
            else:
                magnitude_Hpe_avg += magnitude_Hpe
                phase_Hpe_avg += phase_Hpe
            
    average_magnitude_NM = magnitude_Hpe_avg / 6
    average_phase_NM = phase_Hpe_avg / 6
    # Plot the average magnitude of Hpe_FC in the first subplot [0, 0], plot average phase in [0, 1] 
    # axs[0, 1].semilogx(w_FC, average_magnitude_NM, label=f'NM {condition + 1}')
    # axs[1, 1].semilogx(w_FC, average_phase_NM, label=f'NM {condition + 1}')
    axs[0, 1].set_title('Magnitude Velocity')
    axs[1, 1].set_title('Phase Velocity')

    # Acceleration NM
    condition = 2
    for pilot in range(0, 6):
        Hpe_FC = mat[pilot][condition].Hpe_FC
        hper = np.real(Hpe_FC)
        hpec = np.imag(Hpe_FC)
        magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
        phase_Hpe = np.angle(Hpe_FC, deg=True)
        for j in range(len(phase_Hpe)):
            if j != len(phase_Hpe) - 1:
                 if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                        phase_Hpe[j + 1] -= 360

            if pilot == 0:
                magnitude_Hpe_avg = magnitude_Hpe
                phase_Hpe_avg = phase_Hpe
            else:
                magnitude_Hpe_avg += magnitude_Hpe
                phase_Hpe_avg += phase_Hpe
    # axs[0, 2].semilogx(w_FC, average_magnitude_NM, label=f'NM {condition + 1}')
    # axs[1, 2].semilogx(w_FC, average_phase_NM, label=f'NM {condition + 1}')
    axs[0, 2].set_title('Magnitude Acceleration')
    axs[1, 2].set_title('Phase Acceleration')

    # Position M
    condition = 3
    for pilot in range(0, 6):
        Hpe_FC = mat[pilot][condition].Hpe_FC
        hper = np.real(Hpe_FC)
        hpec = np.imag(Hpe_FC)
        magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
        phase_Hpe = np.angle(Hpe_FC, deg=True)
        Hpxd_FC = mat[pilot][condition].Hpxd_FC
        hpxdr = np.real(Hpxd_FC)
        hpxdc = np.imag(Hpxd_FC)
        magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
        phase_Hpxd = np.angle(Hpxd_FC, deg=True)
        for j in range(len(phase_Hpe)):
            if j != len(phase_Hpe) - 1:
                 if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                        phase_Hpe[j + 1] -= 360
                 if abs(phase_Hpxd[j] - phase_Hpxd[j + 1]) >= 180:
                        phase_Hpxd[j + 1] -= 360

        if pilot == 0:
            magnitude_Hpe_avg = magnitude_Hpe
            magnitude_Hpxd_avg = magnitude_Hpxd
            phase_Hpe_avg = phase_Hpe
            phase_Hpxd_avg = phase_Hpxd
        else:
            magnitude_Hpe_avg += magnitude_Hpe
            magnitude_Hpxd_avg += magnitude_Hpxd
            phase_Hpe_avg += phase_Hpe
            phase_Hpxd_avg += phase_Hpxd
    
    average_magnitude_hpe_M = magnitude_Hpe_avg / 6
    average_magnitude_hpxd_M = magnitude_Hpxd_avg / 6
    average_phase_hpe_M = phase_Hpe_avg / 6
    average_phase_hpxd_M = phase_Hpxd_avg / 6
    axs[0, 0].semilogx(w_FC, average_magnitude_hpe_M, label='Hpe',)
    axs[0, 0].semilogx(w_FC, average_magnitude_hpxd_M, label='Hpxd')
    axs[1, 0].semilogx(w_FC, average_phase_hpe_M, label='Hpe')
    axs[1, 0].semilogx(w_FC, average_phase_hpxd_M, label='Hpxd')
    axs[0, 0].legend()
    axs[1, 0].legend()

    # Velocity M
    condition = 4
    for pilot in range(0, 6):
        Hpe_FC = mat[pilot][condition].Hpe_FC
        hper = np.real(Hpe_FC)
        hpec = np.imag(Hpe_FC)
        magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
        phase_Hpe = np.angle(Hpe_FC, deg=True)
        Hpxd_FC = mat[pilot][condition].Hpxd_FC
        hpxdr = np.real(Hpxd_FC)
        hpxdc = np.imag(Hpxd_FC)
        magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
        phase_Hpxd = np.angle(Hpxd_FC, deg=True)
        for j in range(len(phase_Hpe)):
            if j != len(phase_Hpe) - 1:
                 if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                        phase_Hpe[j + 1] -= 360
                 if abs(phase_Hpxd[j] - phase_Hpxd[j + 1]) >= 180:
                        phase_Hpxd[j + 1] -= 360

        if pilot == 0:
            magnitude_Hpe_avg = magnitude_Hpe
            magnitude_Hpxd_avg = magnitude_Hpxd
            phase_Hpe_avg = phase_Hpe
            phase_Hpxd_avg = phase_Hpxd
        else:
            magnitude_Hpe_avg += magnitude_Hpe
            magnitude_Hpxd_avg += magnitude_Hpxd
            phase_Hpe_avg += phase_Hpe
            phase_Hpxd_avg += phase_Hpxd
    
    average_magnitude_hpe_M = magnitude_Hpe_avg / 6
    average_magnitude_hpxd_M = magnitude_Hpxd_avg / 6
    average_phase_hpe_M = phase_Hpe_avg / 6
    average_phase_hpxd_M = phase_Hpxd_avg / 6
    axs[0, 1].semilogx(w_FC, average_magnitude_hpe_M, label='Hpe')
    axs[0, 1].semilogx(w_FC, average_magnitude_hpxd_M, label='Hpxd')
    axs[1, 1].semilogx(w_FC, average_phase_hpe_M, label='Hpe')
    axs[1, 1].semilogx(w_FC, average_phase_hpxd_M, label='Hpxd')
    axs[0, 1].legend()
    axs[1, 1].legend()

    # Acceleration M
    condition = 5
    for pilot in range(0, 6):
        Hpe_FC = mat[pilot][condition].Hpe_FC
        hper = np.real(Hpe_FC)
        hpec = np.imag(Hpe_FC)
        magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
        phase_Hpe = np.angle(Hpe_FC, deg=True)
        Hpxd_FC = mat[pilot][condition].Hpxd_FC
        hpxdr = np.real(Hpxd_FC)
        hpxdc = np.imag(Hpxd_FC)
        magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
        phase_Hpxd = np.angle(Hpxd_FC, deg=True)
        for j in range(len(phase_Hpe)):
            if j != len(phase_Hpe) - 1:
                 if abs(phase_Hpe[j] - phase_Hpe[j + 1]) >= 180:
                        phase_Hpe[j + 1] -= 360
                 if abs(phase_Hpxd[j] - phase_Hpxd[j + 1]) >= 180:
                        phase_Hpxd[j + 1] -= 360

        if pilot == 0:
            magnitude_Hpe_avg = magnitude_Hpe
            magnitude_Hpxd_avg = magnitude_Hpxd
            phase_Hpe_avg = phase_Hpe
            phase_Hpxd_avg = phase_Hpxd
        else:
            magnitude_Hpe_avg += magnitude_Hpe
            magnitude_Hpxd_avg += magnitude_Hpxd
            phase_Hpe_avg += phase_Hpe
            phase_Hpxd_avg += phase_Hpxd
    
    average_magnitude_hpe_M = magnitude_Hpe_avg / 6
    average_magnitude_hpxd_M = magnitude_Hpxd_avg / 6
    average_phase_hpe_M = phase_Hpe_avg / 6
    average_phase_hpxd_M = phase_Hpxd_avg / 6
    axs[0, 2].semilogx(w_FC, average_magnitude_hpe_M, label='Hpe')
    axs[0, 2].semilogx(w_FC, average_magnitude_hpxd_M, label='Hpxd')
    axs[1, 2].semilogx(w_FC, average_phase_hpe_M, label='Hpe')
    axs[1, 2].semilogx(w_FC, average_phase_hpxd_M, label='Hpxd')
    axs[0, 2].legend()
    axs[1, 2].legend()

    plt.tight_layout()
    plt.show()


frequencies = np.linspace(0, 100, 100)
#print(frequencies)

#plot_Bode(0, 0)
#plot_FFT(0, 0)
#plot_6_pilots(0)
#plot_motion_nomotion_average()
compare_6_conditions()

