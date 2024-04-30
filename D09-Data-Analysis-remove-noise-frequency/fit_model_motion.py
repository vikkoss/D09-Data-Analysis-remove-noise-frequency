import scipy.optimize as opt
import numpy as np
from get_data import Data
import matplotlib.pyplot as plt


K_m = 0.1
omega_nm = 10
zeta_nm = 0.6
tau_m = 0.3

initial_parameters = [0.1, 10, 0.6, 0.3]

parameters = [K_m, omega_nm, zeta_nm, tau_m]


def H_pm(parameters, w):
    jw = w * 1j
    K_m = parameters[0]
    return jw * H_scc(parameters, jw) * parameters[0] * H_nm(parameters, jw)

def H_scc(parameters, w):
    return (0.11 * w + 1) / (5.9 * w + 1)

def H_nm(parameters, w):
    jw = w
    omega_nm = parameters[1]
    zeta_nm = parameters[2]
    return omega_nm ** 2 / (jw ** 2 + 2 * zeta_nm * omega_nm * jw + omega_nm ** 2)

def cost_function_pm(parameters, w, Hpm_data, Hpxd_data):
    cost = np.sum(np.sqrt(np.square(np.real(Hpm_data) - np.real(H_pm(parameters, w))) + np.square(np.imag(Hpm_data) - np.imag(H_pm(parameters, w)))))
    cost += np.sum(np.sqrt(np.square(np.real(Hpxd_data) - np.real(H_scc(parameters, w))) + np.square(np.imag(Hpxd_data) - np.imag(H_scc(parameters, w)))))
    return cost

def boxplots_hpxd():
    K_m_arr = [[], [], []]
    omega_nm_arr = [[], [], []]
    zeta_nm_arr = [[], [], []]
    tau_m_arr = [[], [], []]

    for condition in range(3, 6):
        for subject in range(6):
            data = Data(subject + 1, condition + 1)
            H_pe_data = data.Hpe_FC
            H_pxd_data = data.Hpxd_FC
            w_FC = data.w_FC
            optimized_parameters = opt.fmin(cost_function_pm, initial_parameters, args=(w_FC, H_pe_data, H_pxd_data,), disp=False)
            K_m_arr[condition - 3].append(optimized_parameters[0])
            omega_nm_arr[condition - 3].append(optimized_parameters[1])
            zeta_nm_arr[condition - 3].append(optimized_parameters[2])
            tau_m_arr[condition - 3].append(optimized_parameters[3])

    # make 4 plots, 1 for each parameter, with 3 boxplots, 1 for each condition
    plt.figure(1)
    plt.subplot(221)
    plt.boxplot(K_m_arr, showfliers=False)
    plt.title('K_m')
    plt.subplot(222)
    plt.boxplot(omega_nm_arr, showfliers=False)
    plt.title('omega_nm')
    plt.subplot(223)
    plt.boxplot(zeta_nm_arr, showfliers=False)
    plt.title('zeta_nm')
    plt.subplot(224)
    plt.boxplot(tau_m_arr, showfliers=False)
    plt.title('tau_m')
    plt.show()

boxplots_hpxd()