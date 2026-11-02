import pandas as pd
import sys

def load_parameters(filepath):
    try:
        df = pd.read_csv(filepath)
        params = df.iloc[0].to_dict()
        params['L'] = int(params['L'])
        params['steps'] = int(params['steps'])
        return params
        
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        sys.exit(1)