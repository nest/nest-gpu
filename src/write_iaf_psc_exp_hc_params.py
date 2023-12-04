time_res = 0.1
tau_m = 10.0
C_m = 250.0
E_L = -65.0
Theta_rel = 15.0
V_reset_rel = 0.0
tau_syn = 0.5
t_ref = 2.0


import numpy as np


def propagator_32(tau_syn, tau, C, h):
    P32_linear = (
        1.0 / (2.0 * C * tau * tau) * h * h * (tau_syn - tau) * np.exp(-h / tau)
    )
    P32_singular = h / C * np.exp(-h / tau)
    P32 = (
        -tau
        / (C * (1 - tau / tau_syn))
        * np.exp(-h / tau_syn)
        * np.expm1(h * (1 / tau_syn - 1 / tau))
    )

    dev_P32 = abs(P32 - P32_singular)

    if tau == tau_syn or (abs(tau - tau_syn) < 0.1 and dev_P32 > 2.0 * abs(P32_linear)):
        return P32_singular
    else:
        return P32


h = time_res
P11 = np.exp(-h / tau_syn)
P22 = np.exp(-h / tau_m)
P21 = propagator_32(tau_syn, tau_m, C_m, h)
P20 = tau_m / C_m * (1.0 - P22)

n_refractory_steps = int(round(t_ref / time_res))

with open("iaf_psc_exp_hc_params.h", "w") as p_file:
    p_file.write("#define P11 " + "{:.7E}".format(P11) + "\n")
    p_file.write("#define P22 " + "{:.7E}".format(P22) + "\n")
    p_file.write("#define P21 " + "{:.7E}".format(P21) + "\n")
    p_file.write("#define P20 " + "{:.7E}".format(P20) + "\n")
    p_file.write("#define Theta_rel " + "{:.7E}".format(Theta_rel) + "\n")
    p_file.write("#define V_reset_rel " + "{:.7E}".format(V_reset_rel) + "\n")
    p_file.write("#define n_refractory_steps " + str(n_refractory_steps) + "\n")
