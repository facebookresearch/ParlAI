start=$1
end=$2 

for i in $(seq $start $end); do 
    scancel $i; 
done