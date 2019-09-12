Web and Social Information Extraction Project:
==============================================
This repository contains the source code of the *Web and Social Information Extraction*'s project. The problem statement of the project can be found in this [file](docs/project_proposal.pdf). 

-----------------------------------------------------------

**NOTE:** All the scripts in this repository must be called from inside the folder `src`!
This is due to the use of relative paths.

Task 1:
-------
To estimate the topics for each year between 2000 to 2018 run the command:

`python task1.py`


Task 2:
-------
To track the estimated topics through the time run:

`python task2.py <topic-folder>`

with `<topic-folder>` the directory containing the estimated topics (that is, the output of `python task1.py`).