PDF = comparison.pdf
EPS = comparison.eps
PLT = comparison.plt

pdf:	$(PLT) 
	gnuplot comparison.plt
	epstopdf comparison.eps

eps:	$(PLT) 
	gnuplot comparison.plt

clean:
	rm $(PDF) $(EPS)