import sys

import nestgpu as ngpu

syn_group = ngpu.CreateSynGroup("test_syn_model", {"fact": 1.0, "offset": 2.0})
print(ngpu.GetStatus(syn_group))
ngpu.SetStatus(syn_group, "fact", 3.0)
ngpu.SetStatus(syn_group, "offset", 4.0)
print(ngpu.GetStatus(syn_group))

ngpu.SetStatus(syn_group, {"fact": 5.0, "offset": 6.0})
print(ngpu.GetStatus(syn_group))

fact = ngpu.GetSynGroupParam(syn_group, "fact")
print("fact: ", fact)
offset = ngpu.GetSynGroupParam(syn_group, "offset")
print("offset: ", offset)

print(ngpu.GetStatus(syn_group, "fact"))

print(ngpu.GetStatus(syn_group, "offset"))

print(ngpu.GetStatus(syn_group, ["fact", "offset"]))

print(ngpu.GetStatus(syn_group, ["offset", "fact"]))
