SRC = slides.tex
PDF = slides.pdf

# Primary engine: tectonic (self-contained, auto-fetches packages).
# Falls back to xelatex if tectonic is not installed.
all: $(PDF)

$(PDF): $(SRC)
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic -X compile $(SRC); \
	else \
		xelatex -interaction=nonstopmode $(SRC); \
		xelatex -interaction=nonstopmode $(SRC); \
	fi

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

distclean: clean
	rm -f $(PDF)

.PHONY: all clean distclean
