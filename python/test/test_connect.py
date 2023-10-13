import ctypes
import nestgpu as ngpu

N = 5

neuron = ngpu.Create("aeif_cond_beta_multisynapse", 2*N)
neuron_even = []
neuron_odd = []
for i in range(N):
    neuron_even.append(neuron[2*i])
    neuron_odd.append(neuron[2*i+1])

even_to_odd_delay = []
even_to_odd_weight = []
odd_to_even_delay = []
odd_to_even_weight = []

for isrc in range(N):
    ise = 2*isrc
    iso = 2*isrc + 1
    
    for itgt in range(N):
        ite = 2*itgt
        ito = 2*itgt + 1
        even_to_odd_delay.append(2.0*N*ise + ito)
        even_to_odd_weight.append(100.0*(2.0*N*ise + ito))
        odd_to_even_delay.append(2.0*N*iso + ite)
        odd_to_even_weight.append(100.0*(2.0*N*iso + ite))


conn_dict={"rule": "all_to_all"}
even_to_odd_syn_dict={
    "weight_array":even_to_odd_weight,
    "delay_array":even_to_odd_delay}
  
odd_to_even_syn_dict={
    "weight_array":odd_to_even_weight,
    "delay_array":odd_to_even_delay}
  
ngpu.Connect(neuron_even, neuron_odd, conn_dict, even_to_odd_syn_dict);
ngpu.Connect(neuron_odd, neuron_even, conn_dict, odd_to_even_syn_dict);

ngpu.Calibrate()
# Even to all
conn_id = ngpu.GetConnections(neuron_even, neuron)
conn_status_dict = ngpu.GetStatus(conn_id)
print("########################################")
print("Even to all")
for i in range(len(conn_status_dict)):
    print (conn_status_dict[i])
print()
print()

# Even to all weight, delay
conn_status_dict = ngpu.GetStatus(conn_id, ["weight", "delay"])
print("########################################")
print("Even to all weight, delat")
for i in range(len(conn_status_dict)):
    print (conn_status_dict[i])
print()
print()

conn_id = ngpu.GetConnections(neuron, neuron_odd)
conn_status_dict = ngpu.GetStatus(conn_id)
print("########################################")
print("All to odd")
for i in range(len(conn_status_dict)):
    print (conn_status_dict[i])
print()
print()

# Even to 3,4,5,6
neuron_3_6 = neuron[3:7]
conn_id = ngpu.GetConnections(neuron_even, neuron_3_6)
conn_status_dict = ngpu.GetStatus(conn_id)
print("########################################")
print("Even to 3,4,5,6")
for i in range(len(conn_status_dict)):
    print (conn_status_dict[i])
print()
print()

 
# 3,4,5,6 to odd
conn_id = ngpu.GetConnections(neuron_3_6, neuron_odd)
conn_status_dict = ngpu.GetStatus(conn_id)
print("########################################")
print("3,4,5,6 to odd")
for i in range(len(conn_status_dict)):
    print (conn_status_dict[i])
print()
print()

 
