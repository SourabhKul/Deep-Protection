PDF = accuracy.pdf
EPS = accuracy.eps
PLT = accuracy.plt

pdf:	$(PLT) 
	gnuplot accuracy.plt
	epstopdf accuracy.eps

eps:	$(PLT) 
	gnuplot accuracy.plt

clean:
	rm $(PDF) $(EPS)