set terminal postscript eps color enhanced "Helvetica" 24
set size 1.0, 0.8
set output 'accuracy.eps'
set  autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label
# set errorbars fullwidth
set style fill solid 1.00 border -1
set key to right

# Labels
set xlabel "Attack" font "Helvetica, 20"
set ylabel "Accuracy (%)" font "Helvetica, 20"

# Boxes
set boxwidth 0.5 absolute
set bmargin  3.5
set border 3 # 1 + 2 + 8 - left + bottom

# tics
set xtics nomirror font "Helvetica, 10"
set ytics nomirror font "Helvetica, 14"
set xtics ("GradientSign" 0, "AdditiveGaussianNoise" 1, "BlendedUniformNoise" 2, "GaussianBlur" 3, "NewtonFool" 4) 
# set ytics("0" 0, "100K" 100000, "500K" 500000, "1000K" 1000000)

# ranges
set yrange [0:1.0]

# plot
plot './accuracy.txt' using ($0):($1) axes x1y1 title '' with boxes lc rgb 'royalblue'