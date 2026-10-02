
for syn in cond_alpha cond_beta psc_alpha psc_exp psc_delta; do
    SYN=$(echo $syn | tr 'a-z' 'A-Z' | tr -d '_')

    umf=user_m1_$syn
    um=user_m1
    UM=USERM1
    for fn in $(ls aeif_${syn}*.cu aeif_${syn}*.h); do
	fn1=$(echo $fn | sed "s/aeif_${syn}/$umf/")
	echo "$fn $fn1"
	cat $fn | sed "s/aeif_${syn}/$um/g; s/AEIF${SYN}/$UM/g" > \
		  $fn1
    done

    umf=user_m2_$syn
    um=user_m2
    UM=USERM2
    for fn in $(ls aeif_${syn}*.cu aeif_${syn}*.h); do
	fn1=$(echo $fn | sed "s/aeif_${syn}/$umf/")
	echo "$fn $fn1"
	cat $fn | sed "s/aeif_${syn}/$um/g; s/AEIF${SYN}/$UM/g" > \
		  $fn1
    done
done

for syn in psc_exp psc_exp_g; do
    SYN=$(echo $syn | tr 'a-z' 'A-Z' | tr -d '_')

    umf=user_m1_iaf_$syn
    um=user_m1
    UM=USERM1
    for fn in $(ls iaf_${syn}*.cu iaf_${syn}*.h); do
	fn1=$(echo $fn | sed "s/iaf_${syn}/$umf/")
	echo "$fn $fn1"
	cat $fn | sed "s/iaf_${syn}/$um/g; s/IAF${SYN}/$UM/g" > \
		  $fn1
    done

    umf=user_m2_iaf_$syn
    um=user_m2
    UM=USERM2
    for fn in $(ls iaf_${syn}*.cu iaf_${syn}*.h); do
	fn1=$(echo $fn | sed "s/iaf_${syn}/$umf/")
	echo "$fn $fn1"
	cat $fn | sed "s/iaf_${syn}/$um/g; s/IAF${SYN}/$UM/g" > \
		  $fn1
    done
done

/bin/cp user_m1_cond_beta.cu user_m1.cu
/bin/cp user_m1_cond_beta.h user_m1.h
/bin/cp user_m1_cond_beta_kernel.h user_m1_kernel.h
/bin/cp user_m1_cond_beta_rk5.h user_m1_rk5.h

/bin/cp user_m2_cond_beta.cu user_m2.cu
/bin/cp user_m2_cond_beta.h user_m2.h
/bin/cp user_m2_cond_beta_kernel.h user_m2_kernel.h
/bin/cp user_m2_cond_beta_rk5.h user_m2_rk5.h
