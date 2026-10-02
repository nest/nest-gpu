import nestgpu as ngpu
import numpy as np

n_neurons = 5000

mu = 10.0
sigma = 5.0
low = 0.1
high = 100


neuron1 = ngpu.Create("aeif_cond_beta_multisynapse", n_neurons)
neuron2 = ngpu.Create("aeif_cond_beta_multisynapse", n_neurons)

dum_l = ngpu.DictToArray(
    {
        "distribution": "lognormal_clipped",
        "mu": mu,
        "sigma": sigma,
        "low": low,
        "high": high,
    },
    n_neurons,
)
l = np.zeros(n_neurons)

for i in range(n_neurons):
    l[i] = dum_l[i]


ngpu.Connect(
    neuron1,
    neuron2,
    {"rule": "one_to_one"},
    {
        "delay": {
            "distribution": "lognormal_clipped",
            "mu": mu,
            "sigma": sigma,
            "low": low,
            "high": high,
        },
        "weight": {
            "distribution": "lognormal_clipped",
            "mu": mu,
            "sigma": sigma,
            "low": low,
            "high": high,
        },
    },
)


ngpu.Calibrate()
ngpu.Simulate(1000)


conn_list = ngpu.GetConnections(neuron1, neuron2)
conn_status = ngpu.GetConnectionStatus(conn_list)

delays = np.zeros(n_neurons)
weights = np.zeros(n_neurons)

for c, conns in enumerate(conn_status):
    delays[c] = conns["delay"]
    weights[c] = conns["weight"]

rng = np.random.default_rng()

arr = []

baseline = rng.lognormal(mean=mu, sigma=sigma, size=n_neurons * 10)
# baseline = np.maximum(np.minimum(baseline, high), low)

for b in baseline:
    if low <= b <= high:
        arr.append(b)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = ax[0, 0].hist(
    l, bins="auto", color="coral", alpha=0.7, rwidth=0.85, density=True, label="Vm"
)
n, bins, patches = ax[0, 1].hist(
    delays,
    bins="auto",
    color="cornflowerblue",
    alpha=0.7,
    rwidth=0.85,
    density=True,
    label="Delays",
)
n, bins, patches = ax[1, 0].hist(
    weights,
    bins="auto",
    color="gold",
    alpha=0.7,
    rwidth=0.85,
    density=True,
    label="Weights",
)
n, bins, patches = ax[1, 1].hist(
    arr,
    bins="auto",
    color="lavender",
    alpha=0.7,
    rwidth=0.85,
    density=True,
    label="Baseline",
)


ax[0, 0].set_xlabel("Value")
ax[0, 1].set_xlabel("Value")
ax[1, 0].set_xlabel("Value")
ax[1, 1].set_xlabel("Value")

ax[0, 0].set_ylabel("Density")
ax[0, 1].set_ylabel("Density")
ax[1, 0].set_ylabel("Density")
ax[1, 1].set_ylabel("Density")
# plt.title('V_m Histogram')
# plt.text(23, 45, r'$\mu=15, b=3$')
ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()


# maxfreq = n.max()
# Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.savefig("lognormal_test.png")

# plt.show()
