import argparse
import json
from os.path import isfile

import pandas as pd
from pandas import DataFrame as DF
import numpy as np
import matplotlib.pyplot as plt
from rich import print

def main():

    if isfile("lr_params.json"):
        with open("lr_params.json", "r") as params_file:
            params = json.load(params_file)
    else:
        print("[yellow]Warning: No 'lr_params.json' file found, setting params to 0.")
        params = {"theta_0": 0, "theta_1": 0}

    try:
        km = float(input("Enter a float value: "))
    except ValueError:
        print("[red]Invalid input. Please enter a valid float.")
        exit(0)
    
    print(f"estimated price = theta_0 + theta_1 * km")
    print(f"estimated price = {params['theta_0']} + {params['theta_1']} * {km}")
    print(f"estimated price = {params['theta_0'] + params['theta_1'] * km}")
    print(f"estimated price = {params['theta_0'] + params['theta_1'] * km:.2f} (rounded)")
    
    
if __name__ == "__main__":
    main()