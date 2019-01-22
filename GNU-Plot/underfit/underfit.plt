set terminal postscript eps color enhanced "Helvetica" 24
set size 1.0, 0.8
set output 'underfit.eps'
set  autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label

# set errorbars fullwidth
set style fill solid 1.00 border -1
set key outside top center maxrows 1
set key font ",25"

# Labels
set xlabel "Attack" font "Helvetica, 30"
set ylabel "Error (%)" font "Helvetica, 30"

# Boxes
set boxwidth 0.5 absolute
set bmargin  3.5
set border 3 # 1 + 2 + 8 - left + bottom

# Histogram
set style data histograms
set style histogram rowstacked

# tics
set xtics nomirror font "Helvetica, 20"
set ytics nomirror font "Helvetica, 20"
set xtics ("GS" 0, "AGN" 1, "BUN" 2, "AD" 3, "NF" 4, "DFA" 5, "CWA" 6, "SPN" 7) 

# range
set yrange [0:100]

plot 'results.dat' using ($2) ti "Overfit" lc rgb 'royalblue' lw 2 lt -1 fillstyle pattern 1, '' using ($3*0.73) ti "Underfit" lc rgb 'red' lw 2 lt -1 fillstyle pattern 4