# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

default: dirhtml

vpath %.ipynb _static/notebooks
ALL_TUTORIALS = $(patsubst _static/notebooks/%.ipynb,tutorials/%.rst,$(wildcard _static/notebooks/*.ipynb))
TUTORIAL_FILES = $(patsubst _static/notebooks/%.ipynb,tutorials/%_files,$(wildcard _static/notebooks/*.ipynb))

tutorials/%.rst: %.ipynb tutorials/tutorial_rst.tpl
	jupyter nbconvert --TemplateExporter.filters={\"fix_links\"\:\ \"link_fixer.fix_links\"} --template tutorials/tutorial_rst --to rst $< --output-dir tutorials

.PHONY: tutorials
tutorials: $(ALL_TUTORIALS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/* $(ALL_TUTORIALS) $(TUTORIAL_FILES)

.PHONY: html
html: tutorials
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

.PHONY: dirhtml
dirhtml: tutorials
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

.PHONY: singlehtml
singlehtml: tutorials
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."
