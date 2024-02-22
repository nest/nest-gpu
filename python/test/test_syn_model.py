import sys
import nestgpu as ngpu

tolerance = 1.0e-6
dt_step = 0.1
N = 5
fact = 0.2
offset = 0.03

ngpu.SetKernelStatus({'spike_buffer_algo': 0})

syn_group = ngpu.CreateSynGroup("test_syn_model")
ngpu.SetSynGroupParam(syn_group, "fact", fact)
ngpu.SetSynGroupParam(syn_group, "offset", offset)

sg = ngpu.Create("spike_generator", N)
neuron = ngpu.Create("aeif_cond_beta_multisynapse", 2*N)
ngpu.SetStatus(neuron, {"t_ref": 10.0})
neuron0 = neuron[0:N]
neuron1 = neuron[N:2*N]
dt_list = []
for i in range(N):
    dt_list.append(dt_step*(-0.5*(N-1) + i))

spike_time = [50.0]
spike_mul = [1.0]
n_spikes = 1
time_diff = 10.0

# set spike times and multiplicity
ngpu.SetStatus(sg, {"spike_times": spike_time, "spike_gen_mul":spike_mul})
delay0 = 1.0
delay1 = delay0 + time_diff
weight_sg = 17.9
weight_test = 0.0

conn_dict={"rule": "one_to_one"}
syn_dict0={"weight":weight_sg, "delay":delay0, "receptor":0, "synapse_group":0}
syn_dict1={"weight":weight_sg, "delay":delay1, "receptor":0, "synapse_group":0}

ngpu.Connect(sg, neuron0, conn_dict, syn_dict0)
ngpu.Connect(sg, neuron1, conn_dict, syn_dict1)

for i in range(N):
    delay_test = time_diff - dt_list[i]
    syn_dict_test={"weight":weight_test, "delay":delay_test, "receptor":0, \
                   "synapse_group":syn_group}
    ngpu.Connect([neuron0[i]], [neuron1[i]], conn_dict, syn_dict_test)

ngpu.Simulate(200.0)

conn_id = ngpu.GetConnections(neuron0, neuron1)
conn_status_dict = ngpu.GetStatus(conn_id, ["weight", "delay"])
#print (conn_status_dict)
for i in range(N):
    #print dt_list[i], conn_status_dict[0][i]
    expect_w = dt_list[i]*fact + offset
    if abs(expect_w - conn_status_dict[0][i])>tolerance:
        print("Expected weight: ", expect_w, " simulated: ", \
              conn_status_dict[0][i])
        sys.exit(1)

sys.exit(0)
