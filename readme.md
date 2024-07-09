# Project Title

## Description

NoHyP simulator is a computational application for solving SWE, it has features related to Non-Hydrostatic Pressure

## Installation

Provide code and explanations on how to install your project.

```bash
git clone https://github.com/JoseSB-nature/NoHyP-Simulator.git
```

You will need to get your python environment to satisfy the current requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


# Usage
You need to create the following folders:

- _img_: used to save simulation images
- _cases_: to save some cases configuration, it allows you to replicate the conditions used for an specific test

The ```main.py``` example show an specific run case structure, in general, to use the canal _Class_ you can proceed as follows:

```python
from bib.canal import Canal

# Create a new Canal object
river = Canal()
```

# tests and benchmarks

The Simulator provides you with a simple way of runing test cases, only run ```pytest```. The tests are also required on a commit on the main branch
