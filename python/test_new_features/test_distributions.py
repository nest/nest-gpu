import matplotlib.pyplot as plt
import nestgpu as ngpu

n=ngpu.Create('aeif_cond_beta', 10000, 3)
ngpu.SetStatus(n, 'V_m', {'distribution':'normal_clipped','mu':1.0, 'sigma':0.5, 'low':0.1, 'high':2.0})

ngpu.SetStatus(n, 'C_m', {'distribution':'normal_clipped','mu':100.0, 'sigma':50.0, 'low':30.0, 'high':200.0})

V_m = ngpu.GetStatus(n,'V_m')
plt.figure(1)
plt.hist(V_m, bins = 50)

C_m = ngpu.GetStatus(n,'C_m')
plt.figure(2)
plt.hist(C_m, bins = 50)

ngpu.SetStatus(n.ToList(), 'V_m', {'distribution':'normal_clipped','mu':2.0, 'sigma':1.0, 'low':0.2, 'high':4.0})

ngpu.SetStatus(n.ToList(), 'C_m', {'distribution':'normal_clipped','mu':200.0, 'sigma':100.0, 'low':60.0, 'high':400.0})

V_m = ngpu.GetStatus(n,'V_m')
plt.figure(3)
plt.hist(V_m, bins = 50)

C_m = ngpu.GetStatus(n,'C_m')
plt.figure(4)
plt.hist(C_m, bins = 50)

ngpu.SetStatus(n, 'tau_rise', {'distribution':'normal_clipped',
                               'mu':[10.0, 20.0, 30.0],
                               'sigma':[5.0, 10.0, 15.0],
                               'low':[5.0, 10.0, 15.0],
                               'high':[30.0, 60.0, 90.0]})

tau_rise = ngpu.GetStatus(n,'tau_rise')
tau_rise = list(map(list, zip(*tau_rise)))

plt.figure(5)
plt.hist(tau_rise[0], bins = 50)

plt.figure(6)
plt.hist(tau_rise[1], bins = 50)

plt.figure(7)
plt.hist(tau_rise[2], bins = 50)


plt.show()
