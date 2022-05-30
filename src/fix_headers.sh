for fn in $(ls *.h); do
    a=$(echo "$fn" | tr 'a-z' 'A-Z' | tr -d '_' | tr '.' '_')
    b=$(grep \#ifndef $fn | awk '{print $2}' | head -1)
    c=$(grep -A1 \#ifndef $fn | grep '#define' | awk '{print $2}' | head -1)
    if [ "$b" != "$a" ]; then
	echo "1 $fn $a $b"
	cat $fn | sed "s/$b/$a/" > tmp.txt
	/bin/mv tmp.txt $fn
    elif [ "$c" != "$a" ]; then
	echo "2 $fn $a $c"
	cat $fn | sed "s/$c/$a/" > tmp.txt
	/bin/mv tmp.txt $fn
    fi
    #echo "$a $b $c"
done
