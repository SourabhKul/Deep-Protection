PDF = accuracy_dynamic.pdf
EPS = accuracy_dynamic.eps
PLT = accuracy_dynamic.plt

pdf:	$(PLT) 
	gnuplot accuracy_dynamic.plt
	epstopdf accuracy_dynamic.eps

eps:	$(PLT) 
	gnuplot accuracy_dynamic.plt

clean:
	rm $(PDF) $(EPS)