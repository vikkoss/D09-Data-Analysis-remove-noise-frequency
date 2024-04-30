import scipy.optimize as opt
import numpy as np
from get_data import Data
import matplotlib.pyplot as plt

# Data: subject, condition
subject = 1
condition = 5
run = 2
data = Data(subject, condition)

Hpe_data = data.Hpe_FC
Hpxd_data = data.Hpxd_FC
w_FC = data.w_FC

x_total = data.x
x_total = np.transpose(x_total)
x = x_total[run]
x = np.transpose(x)
t = data.t[0]

# Coefficients
K_pe = 1         # magnitude of magnitude graph
t_lead = 0.6     # moves phase graph up
t_lag = 0.1      # moves phase graph down
tau = 0.3        # decreases gradient of phase graph
omega_nm = 10    # location of magnitude peak
zeta_nm = 0.6    # height of magnitude peak

initial_parameters = [1, 0.2, 0.1, 0.45, 13, 0.6]
parameters = [K_pe, t_lead, t_lag, tau, omega_nm, zeta_nm]

"""
- Boxplots for parameters in no motion case
- New fits for motion model (conditions 4-6)
- Boxplots for total model (4-6)
"""

def H_pe(parameters, w):
    jw = w * 1j
    K_pe = parameters[0]
    t_lead = parameters[1]
    t_lag = parameters[2]
    tau = parameters[3]
    omega_nm = parameters[4]
    zeta_nm = parameters[5]
    return K_pe * (t_lead * jw + 1) / (t_lag * jw + 1) * np.exp(- jw * tau) * H_nm(parameters, jw)

def H_nm(parameters, w):
    jw = w
    K_pe = parameters[0]
    t_lead = parameters[1]
    t_lag = parameters[2]
    tau = parameters[3]
    omega_nm = parameters[4]
    zeta_nm = parameters[5]
    return omega_nm ** 2 / (jw ** 2 + 2 * zeta_nm * omega_nm * jw + omega_nm ** 2)

def cost_function_hpe(parameters, w, Hpe_data):
    cost = np.sum(np.sqrt(np.square(np.real(Hpe_data) - np.real(H_pe(parameters, w))) 
                        + np.square(np.imag(Hpe_data) - np.imag(H_pe(parameters, w)))))
    return cost

def plot():
    w_FC = data.w_FC
    response = H_pe(parameters, w_FC)
    opt_parameters = opt.fmin(cost_function_hpe, initial_parameters, args=(w_FC, Hpe_data,))
    opt_response = H_pe(opt_parameters, w_FC)

    # Get magnitude and phase of numbers in Hpe_FC
    Hpe_FC = Hpe_data
    hper = np.real(Hpe_FC)
    hpec = np.imag(Hpe_FC)
    magnitude_Hpe = np.sqrt(np.square(hper) + np.square(hpec))
    phase_Hpe = np.angle(Hpe_FC, deg=True)
    # Get magnitude and phase of numbers in Hpxd_FC
    Hpxd_FC = Hpxd_data
    hpxdr = np.real(Hpxd_FC)
    hpxdc = np.imag(Hpxd_FC)
    magnitude_Hpxd = np.sqrt(np.square(hpxdr) + np.square(hpxdc))
    phase_Hpxd = np.angle(Hpxd_FC, deg=True)
    # Get magnitude and phase of numbers in response
    hper_response = np.real(response)
    hpec_response = np.imag(response)
    magnitude_response = np.sqrt(np.square(hper_response) + np.square(hpec_response))
    phase_response = np.angle(response, deg=True)
    # Get magnitude and phase of numbers in opt_response
    hper_opt_response = np.real(opt_response)
    hpec_opt_response = np.imag(opt_response)
    magnitude_opt_response = np.sqrt(np.square(hper_opt_response) + np.square(hpec_opt_response))
    phase_opt_response = np.angle(opt_response, deg=True)

    for i in range(len(phase_Hpe)):
        if i != len(phase_Hpe) - 1:
            if abs(phase_Hpe[i] - phase_Hpe[i + 1]) >= 180:
                phase_Hpe[i + 1] -= 360
            if abs(phase_Hpxd[i] - phase_Hpxd[i + 1]) >= 180:
                phase_Hpxd[i + 1] -= 360
            if abs(phase_response[i] - phase_response[i + 1]) >= 180:
                phase_response[i + 1] -= 360
            if abs(phase_opt_response[i] - phase_opt_response[i + 1]) >= 180:
                phase_opt_response[i + 1] -= 360


    # Create a figure with two subplots
    fig, axs = plt.subplots(2)

    # Plot the magnitude of Hpe_FC in the first subplot
    axs[0].semilogx(w_FC, magnitude_Hpe, label='Hpe')
    axs[0].semilogx(w_FC, magnitude_Hpxd, label='Hpxd')
    axs[0].semilogx(w_FC, magnitude_response, label='Initial Guess')
    axs[0].semilogx(w_FC, magnitude_opt_response, label='Optimized')
    axs[0].set_title('Magnitude of Hpe_FC')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid()
    axs[0].legend()

    # Plot the phase of Hpe_FC in the second subplot
    axs[1].semilogx(w_FC, phase_Hpe, label='Hpe')
    axs[1].semilogx(w_FC, phase_Hpxd, label='Hpxd')
    axs[1].semilogx(w_FC, phase_response, label='Initial Guess')
    axs[1].semilogx(w_FC, phase_opt_response, label='Optimized')
    axs[1].set_title('Phase of Hpe_FC')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Phase (degrees)')
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()


#opt_parameters = opt.fmin(cost_function_hpe, initial_parameters, args=(w_FC, Hpe_data,))

#print(opt_parameters)

#plot()

def boxplots_hpe():
    K_pe_arr = [[], [], [], [], [], []]
    t_lead_arr = [[], [], [], [], [], []]
    t_lag_arr = [[], [], [], [], [], []]
    tau_arr = [[], [], [], [], [], []]
    omega_nm_arr = [[], [], [], [], [], []]
    zeta_nm_arr = [[], [], [], [], [], []]
    
    for condition in range(6):
        for subject in range(6):
            data = Data(subject + 1, condition + 1)
            Hpe_data = data.Hpe_FC
            w_FC = data.w_FC
            optimized_parameters = opt.fmin(cost_function_hpe, initial_parameters, args=(w_FC, Hpe_data,))
            K_pe_arr[condition].append(optimized_parameters[0])
            t_lead_arr[condition].append(optimized_parameters[1])
            t_lag_arr[condition].append(optimized_parameters[2])
            tau_arr[condition].append(optimized_parameters[3])
            omega_nm_arr[condition].append(optimized_parameters[4])
            zeta_nm_arr[condition].append(optimized_parameters[5])


    # make 6 plots, 1 for each parameter, with 6 boxplots, 1 for each condition
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Optimized Parameters for Hpe')
    axs[0, 0].boxplot(K_pe_arr, showfliers=False)
    axs[0, 0].set_title('K_pe')
    axs[0, 1].boxplot(t_lead_arr, showfliers=False)
    axs[0, 1].set_title('t_lead')
    axs[0, 2].boxplot(t_lag_arr, showfliers=False)
    axs[0, 2].set_title('t_lag')
    axs[1, 0].boxplot(tau_arr, showfliers=False)
    axs[1, 0].set_title('tau')
    axs[1, 1].boxplot(omega_nm_arr, showfliers=False)
    axs[1, 1].set_title('omega_nm')
    axs[1, 2].boxplot(zeta_nm_arr, showfliers=False)
    axs[1, 2].set_title('zeta_nm')
    print(omega_nm_arr)
    print(zeta_nm_arr)
    
    plt.show()

boxplots_hpe()

