ALL_FIGURE_NAMES=$(shell cat thesis.figlist)
ALL_FIGURES=$(ALL_FIGURE_NAMES:%=%.pdf)

allimages: $(ALL_FIGURES)
	@echo All images exist now. Use make -B to re-generate them.

FORCEREMAKE:

include $(ALL_FIGURE_NAMES:%=%.dep)

%.dep:
	mkdir -p "$(dir $@)"
	touch "$@" # will be filled later.

./_tikzcache/thesis-figure0.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "./_tikzcache/thesis-figure0" "\def\tikzexternalrealjob{thesis}\input{thesis}"

./_tikzcache/thesis-figure0.pdf: ./_tikzcache/thesis-figure0.md5
./_tikzcache/thesis-figure1.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "./_tikzcache/thesis-figure1" "\def\tikzexternalrealjob{thesis}\input{thesis}"

./_tikzcache/thesis-figure1.pdf: ./_tikzcache/thesis-figure1.md5
./_tikzcache/thesis-figure2.pdf: 
	pdflatex -shell-escape -halt-on-error -interaction=batchmode -jobname "./_tikzcache/thesis-figure2" "\def\tikzexternalrealjob{thesis}\input{thesis}"

./_tikzcache/thesis-figure2.pdf: ./_tikzcache/thesis-figure2.md5
