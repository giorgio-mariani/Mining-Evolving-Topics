
Mining Evolving Topics
=======================
Web and Social Information Extraction Project
----------------------------------------------
This repository contains the source code of the *Web and Social Information Extraction* course's project: **Mining Evolving Topics***. The problem statement for the project can be found in this [file](docs/project_proposal.pdf). The report for the project can be found [here](docs/report.pdf)

-----------------------------------------------------------
Requirements
------------
The requirements necessary in order to run the code can be installed through pip using the command:

`pip install -r  requirements.txt`

If conda is used instead, it is possible to create a suitable environment using:

`conda env create -f environment.yml -n websocial_project`


Task 1:
-------
To estimate the topics for each year between 2000 to 2018 run the command:

`python task1.py -s 2000 -e 2018 <output dir>`


Task 2:
-------
To track the estimated topics through the time run:

`python task2.py <input dir>`

with `<input dir>` equal to the values of  `<output dir>` for task-1.
