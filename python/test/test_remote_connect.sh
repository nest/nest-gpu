pass_str[0]="TEST PASSED"
pass_str[1]="TEST NOT PASSED"
fn=test_remote_connect.py
mpirun -np 3 python3 $fn | grep CHECK | sort -n > tmp
diff -qs tmp log_remote_connect.txt 2>&1 >> log.txt
res=$?
echo $fn : ${pass_str[$res]}    
