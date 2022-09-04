#!/bin/sh



# Lines before execution
MPIRUN='ibrun'
PW='/home1/08702/wkim94/codes/qe-7.0-elpa-hdf5-fft-openmp/bin/pw.x'
PWFLAGS='-nk 7'

$MPIRUN $PW $PWFLAGS -in pw.scf.in &> pw.scf.out
$MPIRUN $PW $PWFLAGS -in pw.bands.in &> pw.bands.out
rm WSe2.wfc*


# Lines after execution
