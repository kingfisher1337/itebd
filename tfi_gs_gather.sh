#!/bin/bash

cd output_tfi
rm "h_mz_E_D=2_chi=20.dat"

for f in `ls D=2_chi=20_*_itebd.dat`
do
    h=`echo $f | cut -f 3 -d "_" | cut -f 2 -d "="`
    mz=`tail -n 1 $f | cut -f 3 -d " "`
    E=`tail -n 1 $f | cut -f 5 -d " "`
    echo $h $mz $E >> "h_mz_E_D=2_chi=20.dat"
done

cd ..
exit 0

