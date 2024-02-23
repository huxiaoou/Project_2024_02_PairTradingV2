#!/bin/bash

bgn_date_diff="20160104"
bgn_date_fact=$bgn_date_diff
bgn_date_regp="20160201"
bgn_date_mclrn="20160701"
bgn_date_simu=$bgn_date_regp
stp_date="20240219"

# diff return
python main.py diff --mode o --bgn $bgn_date_diff --stp $stp_date

# factor exposure
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor lag
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor sum
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor ewm
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor vol
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor tnr
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor basisa
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor ctp
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor cvp
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor csp
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor rsbr
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor rslr
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor skew
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor mtms
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor tsa
python main.py exposure --mode o --bgn $bgn_date_fact --stp $stp_date --factor tsld

# regroup
python main.py regroups --mode o --bgn $bgn_date_regp --stp $stp_date

# quick simulations and evaluations
python main.py simu-quick --bgn $bgn_date_simu --stp $stp_date --mode o
python main.py eval-quick --bgn $bgn_date_simu --stp $stp_date

# machine learning
python main.py mclrn --bgn $bgn_date_mclrn --stp $stp_date --mode o
python main.py simu-mclrn --bgn $bgn_date_simu --stp $stp_date --mode o
python main.py eval-mclrn --bgn $bgn_date_simu --stp $stp_date

# portfolio
python main.py simu-portfolios --bgn $bgn_date_simu --stp $stp_date --mode o
python main.py eval-portfolios --bgn $bgn_date_simu --stp $stp_date
