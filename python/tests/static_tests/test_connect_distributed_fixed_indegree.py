import sys
import nestgpu as ngpu

ngpu.SetKernelStatus("rnd_seed", 1234 + int(sys.argv[1]))

ngpu.ConnectMpiInit()

hg1 = ngpu.CreateHostGroup([0, 2, 4, 6])

neuron = ngpu.Create("iaf_psc_exp", 10)

syn_spec = {"weight": 2.0, "delay": 0.6}
indegree = 4
source_host_list = [0, 4, 6]
source_group_list = [[3, 1, 4, 5], [5, 8, 6], [3, 4, 7]]
target_host_list = [2, 6]
target_group_list = [[9, 2, 7], [0, 3, 1, 2]]
ngpu.ConnectDistributedFixedIndegree(
    source_host_list,
    source_group_list,
    target_host_list,
    target_group_list,
    indegree,
    hg1,
    syn_spec,
)

ngpu.MpiFinalize()
