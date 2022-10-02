import nestgpu as ngpu

n1 = ngpu.Create('iaf_psc_exp', 12345)
pg = ngpu.Create('poisson_generator', 6789)
n2 = ngpu.Create('iaf_psc_exp', 3141)

conn_dict={"rule": "all_to_all"}
syn_dict={"delay": {"distribution":"normal_clipped",
                    "mu":0.4, "low":0.1, "high":1.0, "sigma":0.4},
          "weight": {"distribution":"normal_clipped",
                     "mu":1.5, "low":0.5, "high":2.0, "sigma":0.25}}

ngpu.Connect(n1, n2, conn_dict, syn_dict)
ngpu.Connect(n1, n1, conn_dict, syn_dict)
ngpu.Connect(pg, n1, conn_dict, syn_dict)
ngpu.Connect(pg, n2, conn_dict, syn_dict)
ngpu.Connect(n2, n2, conn_dict, syn_dict)

ngpu.Simulate(0.4)

