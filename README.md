# CodeForEarth


To set up the package locally: 

- Cloning the project: 
git clone https://github.com/EmmaScharfmann/CodeForEarth.git


- Importing the package:
cd CodeForEarth
pip install -e .


- Installing the necessary packages:
```bash
conda install -c conda-forge esmpy
pip install -r requirements.txt
```

- Formatting:
Run: 
```black . ``` to format the code automatically.


### Conventions:

#### 1. Code documentation:
- The public methods have full docstring: one line description + input and output descriptions.
- The private methods can only have a one line descriptions.
- Input and output types should always be specified 
- The docstrings should be sufficient to understand the code. In most of the cases, comments are not necessary and are redundant with the docstrings. Comments can be used to explain non-obvious choices.
- TODOs can be added to indicate what must be changed in the future.   
#### 2. Code conventions:
- code repetition should be avoided (as much as possible)
- functions and variable names use snake_case 
- function names should start with a verb (handle_formatting, build_model, get_data)
- when an empty object (list, set, dict,...) is initialized, it is good practice to specify its type (example: `new_list: list[int]: []`)
- function and variable names should be as clear as possible
- input and output type hints should be as precise as possible. Using custom classes is good practice to have precise types.
#### 3. Formatting 
- Run `black .` before creating a PR to format the code. 