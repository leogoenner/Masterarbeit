# Training of Convolutional Neural Networks using Tucker Decompositions

This repository contains the code and for my master's thesis.

The execution of the file main.py starts the training of a CNN.
All parameters can be specified in parameters.py.
Conv_NN.py, Standard_Conv_NN.py and Tucker_Conv_NN.py are class definition files.
algorithm_1 contains the algorithms 7 and 8 described in subsection 2.2.3.
Different kinds of functions are outsourced to utils.py to simplify the other files.
create_plot_ranks.py and create_plot_errors.py are used to create the plots discussed in section 3.2.


The packages required for the code are:
- os
- tensorflow
- datetime
- time
- pandas
- pickle
- shutil
- functools
- funcy
- numpy
- scipy
- random
- warnings
- einops
- matplotlib
- tueplots