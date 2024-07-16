# ReconMem

This repository contains the data collected for Study 3 and the analyses scripts in the following manuscript:

Lin, Q., Li, Z., Lafferty, J. & Yildirim, I. (2024). Images with harder-to-reconstruct visual representations leave stronger memory traces. _Nature Human Behaviour_

The data used for Studies 1 and 2 are from a publicly available dataset from the following paper:

Isola, P., Xiao, J., Parikh, D., Torralba, A., & Oliva, A. (2013). What makes a photograph memorable?. IEEE transactions on pattern analysis and machine intelligence, 36(7), 1469-1482.

The ./SPC_public submodule includes scripts for extracting distinctiveness and reconstruction error for the images. See the ./SPC_public/README for detailed explanation. 

# Setup
First, clone this repo

```
git clone https://github.com/CNCLgithub/ReconMem
```

Then install the required python packages (make sure that you are in the ReconMem directory)

```
pip install requirements.txt
```

# Conducting the analysis

We provided a jupyter notebook (./analysis_scripts/Behavioral analysis.ipynb) recreating the results in Figure 5.  The ANOVA analysis reported in Supplementary Table 1 was performed in R. It can be reproduced by executing the markdown file (./analysis_scripts/ANOVA.rmd) in RStudio.

