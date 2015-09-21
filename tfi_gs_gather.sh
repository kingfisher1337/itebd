#!/bin/bash

cd output_tfi
rm "h_mz_E_D=2_chi=20.dat"
rm "h_mz_E_D=2_chi=20_pm.dat"
rm "h_mz_E_D=3_chi=30.dat"
rm "h_mz_E_D=3_chi=30_pm.dat"

for f in `ls D=2_chi=20_*_trotter2_itebd.dat`
do
    h=`echo $f | cut -f 3 -d "_" | cut -f 2 -d "="`
    mz=`tail -n 1 $f | cut -f 3 -d " "`
    E=`tail -n 1 $f | cut -f 5 -d " "`
    tau=`echo $f | cut -f 4 -d "_" | cut -f 2 -d "="`
    echo $h $mz $E $tau >> "h_mz_E_D=2_chi=20.dat"
done

for f in `ls D=2_chi=20_*_trotter2_pm_itebd.dat`
do
    h=`echo $f | cut -f 3 -d "_" | cut -f 2 -d "="`
    mz=`tail -n 1 $f | cut -f 3 -d " "`
    E=`tail -n 1 $f | cut -f 5 -d " "`
    tau=`echo $f | cut -f 4 -d "_" | cut -f 2 -d "="`
    echo $h $mz $E $tau >> "h_mz_E_D=2_chi=20_pm.dat"
done

for f in `ls D=3_chi=30_*_trotter2_itebd.dat`
do
    h=`echo $f | cut -f 3 -d "_" | cut -f 2 -d "="`
    mz=`tail -n 1 $f | cut -f 3 -d " "`
    E=`tail -n 1 $f | cut -f 5 -d " "`
    tau=`echo $f | cut -f 4 -d "_" | cut -f 2 -d "="`
    echo $h $mz $E $tau >> "h_mz_E_D=3_chi=30.dat"
done

for f in `ls D=3_chi=30_*_trotter2_pm_itebd.dat`
do
    h=`echo $f | cut -f 3 -d "_" | cut -f 2 -d "="`
    mz=`tail -n 1 $f | cut -f 3 -d " "`
    E=`tail -n 1 $f | cut -f 5 -d " "`
    tau=`echo $f | cut -f 4 -d "_" | cut -f 2 -d "="`
    echo $h $mz $E $tau >> "h_mz_E_D=3_chi=30_pm.dat"
done

cd ..
exit 0

