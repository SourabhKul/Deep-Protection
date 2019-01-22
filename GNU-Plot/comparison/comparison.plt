set terminal postscript eps color enhanced "Helvetica" 24
set size 1.0, 0.8
set output 'comparison.eps'
set  autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label
# set errorbars fullwidth
set style fill solid 1.00 border -1
set key outside top center maxrows 1
set key font ",20"
# Labels
set xlabel "Attack" font "Helvetica, 30"
set ylabel "Accuracy (%)" font "Helvetica, 30"

# Boxes
set boxwidth 0.4 absolute
set bmargin  3.5
set border 3 # 1 + 2 + 8 - left + bottom

# tics
set xtics nomirror font "Helvetica, 25"
set ytics nomirror font "Helvetica, 25"
set xtics ("Deep Protection" 0, "Prior work" 1) 
# set ytics("0" 0, "100K" 100000, "500K" 500000, "1000K" 1000000)

# GS - Gradient Sign
# AGN - Additive Gaussian Noise
# BUN - Blended Uniform Noise
# AD - ADefAttack
# GB - Gaussian Blur
# NF - NewtonFool
# DFA - DeepFoolAttack
# CWA - CarliniWagnerL2Attack
# SPN - SaltAndPepperNoiseAttack

# ranges
set yrange [0:100]

# plot
plot './comparison.txt' using ($0)-0.2:(($1)) axes x1y1 title 'Clean Images' with boxes lc rgb 'red' lw 2 lt -1 fillstyle pattern 1, \
    '' using ($0)+0.2:(($2)) axes x1y1 title 'Adversarial Images' with boxes lc rgb 'royalblue' lw 2 lt -1 fillstyle pattern 4