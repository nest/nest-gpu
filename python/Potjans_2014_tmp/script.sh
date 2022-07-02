for i in $(seq 0 9); do
    cat sim_params.templ | sed "s/__seed__/1234$i/" > sim_params.py
    python3  run_microcircuit.py
    mv data data$i
done
