# ReconMem

This repository contains the data collected for Study 2 and the analyses scripts in the following manuscript:

Lin, Q., Li, Z., Lafferty, J. & Yildirim, I. (submittedd). Images that are harder to reconstruct are more memorable. 

The data used for Study 1 are from a publicly available dataset from the following paper:

Isola, P., Xiao, J., Parikh, D., Torralba, A., Oliva, A. What makes a photograph memorable? IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013.

# Setup
First, clone this repo

```
git clone https://github.com/CNCLgithub/ReconMem
```

Then install the required python packages (make sure that you are in the ReconMem directory)

```
pip install requirements.txt
```

We provided a jupyter notebook (./analysis_scripts/Behavioral analysis.ipynb) recreating the results in Figure 4.  The ANOVA analysis reported in Table 1 was performed in R. It can be reproduced by executing the markdown file analysis_scripts/ANOVA.rmd in RStudio.

