# SIESTA2BGW
Generate wavefuntions in planewave basis from SIESTA calculations.

## Usage
1. Run SIESTA and generate '.ion' and '.bands.WFSX' files.
2. Convert the WFSX file to AScii format ('.bands.WFSX.txt') using readwfx.
3. Run siesta2bgw.py 

## OUTPUT FILE FORMAT
Current version generate output files in numpy txt format.
Future version will surpport the format of
BerkeleyGW
QuantumESPRESSO
