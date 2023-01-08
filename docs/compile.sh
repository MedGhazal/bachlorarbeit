# usr/bin/zsh

xelatex main.tex
biber main
xelatex main.tex
open main.pdf
