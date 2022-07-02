for i in $(seq 0 9); do
    cat sim_params_norec.templ | sed "s/__seed__/1234$i/" > sim_params_norec.py
    python3  eval_microcircuit_time.py | tee log$i.txt
done
