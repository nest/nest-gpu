import nestgpu as ngpu
import numpy as np

neuron = ngpu.Create("aeif_cond_beta_multisynapse",100000)

ngpu.SetStatus(neuron, {"V_m": {"distribution":"normal_clipped",
                                       "mu":-70.0, "low":-90.0,
                                       "high":-50.0,
                                       "sigma":5.0}})
l = ngpu.GetStatus(neuron, "V_m")
d=[]
for elem in l:
    d.append(elem[0])
    
print (len(d))
import matplotlib.pyplot as plt

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(d, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('V_m Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()
