# Build the OPD-estimation note (run from this directory: `make`).
#
# `make figs` regenerates every figure (Figs/*.pdf) and the numeric macros
# (values.tex) from the raw FM1 data by running the analysis script; `make pdf`
# builds the document; `make` does both.
MAIN := main_OPDEstimation
PDF := $(MAIN).pdf
TEX := $(MAIN).tex

# Prefer the project virtualenv interpreter if present, else fall back to python3.
PYTHON := $(shell [ -x ../../../.venv/bin/python ] && echo ../../../.venv/bin/python || echo python3)

SCRIPT := make_figures.py
VALUES := values.tex
FIGS := $(wildcard Figs/*.pdf)

PDFLATEX := pdflatex
PDFLATEX_FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all pdf figs clean

all: figs pdf

# Regenerate figures and numeric macros from the data.
figs: $(VALUES)

$(VALUES): $(SCRIPT)
	$(PYTHON) $(SCRIPT)

pdf: $(PDF)

$(PDF): $(TEX) $(VALUES) $(FIGS)
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(TEX)
	$(PDFLATEX) $(PDFLATEX_FLAGS) $(TEX)

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).toc \
		$(MAIN).synctex.gz $(MAIN).synctex \
		$(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).nav $(MAIN).snm \
		$(MAIN).bbl $(MAIN).blg $(MAIN).bcf $(MAIN).run.xml \
		$(MAIN).lof $(MAIN).lot $(MAIN).idx $(MAIN).ilg $(MAIN).ind

# Also remove generated figures and macros.
distclean: clean
	rm -f $(PDF) $(VALUES) Figs/*.pdf
