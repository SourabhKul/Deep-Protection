PDF = accuracy_fixed.pdf
EPS = accuracy_fixed.eps
PLT = accuracy_fixed.plt

pdf:	$(PLT) 
	gnuplot accuracy_fixed.plt
	epstopdf accuracy_fixed.eps

eps:	$(PLT) 
	gnuplot accuracy_fixed.plt

clean:
	rm $(PDF) $(EPS)