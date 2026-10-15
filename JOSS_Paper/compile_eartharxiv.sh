#!/usr/bin/env bash
# Copied from https://github.com/johnburley3000/paddocktimeseries/blob/main/manuscript/compile_eartharxiv.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTFILE="Shelterbelts_EarthArXiv_preprint.pdf"
TEMP_MD="$(mktemp "$SCRIPT_DIR/.eartharxiv-paper.XXXXXX")"
HEADER_TEX="$(mktemp "$SCRIPT_DIR/.eartharxiv-header.XXXXXX")"

cleanup() {
  rm -f "$TEMP_MD" "$HEADER_TEX"
}
trap cleanup EXIT

# Convert the JOSS heading to the EarthArXiv-required Abstract heading without
# changing the shared paper.md source.
# add page-break after abstract
awk '
BEGIN { changed=0; pagebreak=0 }

!changed && $0 == "# Summary" {
  $0="# Abstract"
  changed=1
}

changed && !pagebreak && $0 == "# Statement of Need" {
  print "\\newpage"
  print ""
  pagebreak=1
}

{ print }
' "$SCRIPT_DIR/paper.md" > "$TEMP_MD"

# EarthArXiv permits this statement in the header of every page instead of a
# separate cover sheet. Redefining the plain style includes the title page.
printf '%s\n' \
  '\usepackage{fancyhdr}' \
  '\setlength{\headheight}{14pt}' \
  '\pagestyle{fancy}' \
  '\fancyhf{}' \
  '\fancyhead[C]{\scriptsize Non-peer-reviewed preprint submitted to EarthArXiv}' \
  '\fancyfoot[C]{\thepage}' \
  '\fancypagestyle{plain}{\fancyhf{}\fancyhead[C]{\scriptsize Non-peer-reviewed preprint submitted to EarthArXiv}\fancyfoot[C]{\thepage}}' \
  > "$HEADER_TEX"

TEMP_MD_NAME="$(basename "$TEMP_MD")"
HEADER_TEX_NAME="$(basename "$HEADER_TEX")"

docker run --rm \
  --platform linux/amd64 \
  --volume "$SCRIPT_DIR:/data" \
  --workdir /data \
  --user "$(id -u):$(id -g)" \
  pandoc/latex:latest \
  "$TEMP_MD_NAME" \
  --from=markdown \
  --citeproc \
  -M link-citations=true \
  --pdf-engine=xelatex \
  --include-in-header="$HEADER_TEX_NAME" \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V papersize=letter \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue \
  -V citecolor=blue \
  -V float-placement=H \
  -V author='Christopher Bradley$^{1,*}$, John T. Burley$^{1}$, Nicolas Pucino$^{2}$, Justin Borevitz$^{1}$\\[0.5em]$^1$ Australian National University, Canberra, Australia\\$^2$ Climate Friendly, Sydney, Australia\\[0.5em]\small Submitter ORCID: \href{https://orcid.org/0009-0008-2291-8433}{https://orcid.org/0009-0008-2291-8433}\\$^*$ Corresponding author: christopher.bradley@anu.edu.au' \
  -o "$OUTFILE"

SIZE_BYTES="$(wc -c < "$SCRIPT_DIR/$OUTFILE" | tr -d ' ')"
if (( SIZE_BYTES > 40000000 )); then
  echo "Error: $OUTFILE exceeds EarthArXiv's 40 MB file-size limit." >&2
  exit 1
fi

echo "Created: $SCRIPT_DIR/$OUTFILE"
