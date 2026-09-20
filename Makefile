# Render the report to PDF. Needs Google Chrome or Chromium on PATH.
CHROME := $(shell command -v google-chrome || command -v chromium || command -v chromium-browser)

results/El-Nino-and-the-Ski-Season.pdf: results/report.html
	@test -n "$(CHROME)" || { echo "no chrome/chromium on PATH"; exit 1; }
	"$(CHROME)" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
	  --virtual-time-budget=25000 --print-to-pdf="$(CURDIR)/$@" \
	  "file://$(CURDIR)/results/report.html"

.PHONY: pdf
pdf: results/El-Nino-and-the-Ski-Season.pdf
