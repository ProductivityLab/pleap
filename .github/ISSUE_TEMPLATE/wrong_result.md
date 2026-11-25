---
name: A result looks wrong
about: PLEAP disagrees with another program, or with what you expected
labels: ["result"]
---

**What PLEAP reported, and what you expected**

<!-- Scores, indices or coefficients. If another program gives something
     different, say which program and version. A disagreement with established
     software is treated as a bug in PLEAP until shown otherwise. -->

**Settings**

- Model: CCR / BCC
- Orientation: input / output
- For bootstrap: replications B, seed, cores
- For Malmquist: model and orientation (these are selectable and the
  returns-to-scale assumption changes the indices substantially)

**Data**

<!-- A small data set that reproduces it - ten or twenty DMUs is usually
     enough. If the data cannot be shared: DEA scores are invariant to the
     units of each column, so multiplying every input and output by a constant
     and dropping the names changes nothing but the numbers on the page. -->

**One check worth doing first**

Multiply one input or output column by 1000 and re-run. If the scores change,
that is a numerical problem and worth saying so here — it narrows things down
a great deal.

**Version and install**

- PLEAP version (shown in the window title):
- Installed from: Windows installer / pip / source
- Operating system:
