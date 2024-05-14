import sys
import nestgpu as ngpu

spike = ngpu.Create("spike_generator", 4)
spike0 = spike[0:2]
spike1 = spike[2:3]
spike2 = spike[3:4]

spike_time0 = [10.0, 400.0]
spike_mul0 = [1.0, 0.5]
spike_time1 = [4.0]
spike_mul1 = [2.0]
spike_time2 = [50.0, 20.0, 80.0]
spike_mul2 = [0.1, 0.3, 0.2]


# set spike times and multiplicities
ngpu.SetStatus(spike0, {"spike_times": spike_time0,
                        "spike_gen_mul":spike_mul0})

ngpu.SetStatus(spike1, {"spike_times": spike_time1,
                        "spike_gen_mul":spike_mul1})

ngpu.SetStatus(spike2, {"spike_times": spike_time2,
                        "spike_gen_mul":spike_mul2})

print(ngpu.GetStatus(spike0, "spike_times"))
print(ngpu.GetStatus(spike0, "spike_gen_mul"))
print()
print(ngpu.GetStatus(spike1, "spike_times"))
print(ngpu.GetStatus(spike1, "spike_gen_mul"))
print()
print(ngpu.GetStatus(spike2, "spike_times"))
print(ngpu.GetStatus(spike2, "spike_gen_mul"))

print()
print()
neuron_list = [spike[2], spike[3], spike[0], spike[1]]
print(ngpu.GetStatus(neuron_list, "spike_times"))
print(ngpu.GetStatus(neuron_list, "spike_gen_mul"))
print()
print()
print(ngpu.GetStatus(spike1))
print()
print()
print(ngpu.GetStatus(neuron_list))


      
      
