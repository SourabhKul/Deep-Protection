PDF = underfit.pdf
EPS = underfit.eps
PLT = underfit.plt

pdf:	$(PLT) 
	gnuplot underfit.plt
	epstopdf underfit.eps

eps:	$(PLT) 
	gnuplot underfit.plt

clean:
	rm $(PDF) $(EPS)