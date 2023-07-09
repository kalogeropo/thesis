from encpsulate_test import run_experiment
from infre.models import PGSB, ConGSB

if __name__ == '__main__':

    # Define the desired parameters and model constructors
    model_constructors = [PGSB, ConGSB]  # Add more constructors if needed

    # Run experiments for each combination of parameters and model constructors
    # condition = ["sim", {"edge": ['comp', 'cw', 'pw']}]
    run_experiment(model_constructors, condition={"edge": "comp"})