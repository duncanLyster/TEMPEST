#!/usr/bin/env python3
"""
Performance profiling script for TEMPEST: runs parameter sweeps, records runtime and insolation noise,
and generates CSV of results and diagnostic plots.
"""
import os
import sys
import time
import yaml
import subprocess
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths and constants
CONFIG_TEMPLATE = os.path.join('data', 'config', 'example_config.yaml')
OUTPUT_CSV = 'performance_results.csv'
INSOLATION_DIR = 'insolation_data'
TEMPERATURE_DIR = 'temperature_data'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(INSOLATION_DIR, exist_ok=True)
os.makedirs(TEMPERATURE_DIR, exist_ok=True)

# Test definitions: (parameter_name, list_of_values)
TESTS = [
    ('kernel_subfacets_count', [10, 30, 100, 300]),
    ('kernel_directional_bins', [10, 30, 100]),
    ('vf_rays', [100, 1000, 5000]),
    ('intra_facet_scatters', [0, 1, 2, 4]),
    ('chunk_size', [100, 500, 1000, 5000])
]

# Function to load and save YAML configs

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)

# Function to compute noise metric from insolation CSV for facet 0
def compute_noise(ins_csv):
    try:
        df = pd.read_csv(ins_csv)
        return float(np.std(df.iloc[:, 1].values))
    except Exception:
        return None

# Main experiment runner
def run_experiments():
    # Load full example config as base
    template = load_config(CONFIG_TEMPLATE)
    results = []

    for param, values in TESTS:
        for val in values:
            print(f"\n=== Running test: {param} = {val} ===")
            # Prepare config for this run
            cfg = template.copy()
            cfg[param] = val
            # Silence console and disable interactive plots
            cfg['silent_mode'] = True
            cfg['remote'] = True
            temp_cfg_file = f'temp_config_{param}_{val}.yaml'
            save_config(cfg, temp_cfg_file)

            start_time = time.time()
            status = 'OK'
            try:
                subprocess.run(['python', 'tempest.py', '--config', temp_cfg_file], check=True)
            except Exception:
                status = 'ERROR'
                print(f"Run failed for {param}={val}")
                traceback.print_exc()
            runtime = time.time() - start_time

            # compute noise for facet 0 and 1
            f0 = os.path.join(INSOLATION_DIR, 'facet_0.csv')
            f1 = os.path.join(INSOLATION_DIR, 'facet_1.csv')
            noise0 = compute_noise(f0)
            noise1 = compute_noise(f1)
            # archive CSVs with param and value suffix
            for idx, src_dir in [(0, INSOLATION_DIR), (1, INSOLATION_DIR)]:
                src = os.path.join(src_dir, f'facet_{idx}.csv')
                dst = os.path.join(src_dir, f'facet_{idx}_{param}_{val}.csv')
                if os.path.exists(src):
                    os.replace(src, dst)
            for idx, src_dir in [(0, TEMPERATURE_DIR), (1, TEMPERATURE_DIR)]:
                src = os.path.join(src_dir, f'facet_{idx}.csv')
                dst = os.path.join(src_dir, f'facet_{idx}_{param}_{val}.csv')
                if os.path.exists(src):
                    os.replace(src, dst)
            results.append({
                'parameter': param,
                'value': val,
                'runtime_s': runtime,
                'noise0': noise0,
                'noise1': noise1,
                'status': status
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved performance results to {OUTPUT_CSV}")

    # Generate diagnostic plots
    for param in df['parameter'].unique():
        sub = df[df['parameter'] == param]
        # runtime plot
        plt.figure()
        plt.plot(sub['value'], sub['runtime_s'], marker='o')
        plt.title(f'Runtime vs {param}')
        plt.xlabel(param)
        plt.ylabel('Runtime (s)')
        plt.savefig(os.path.join(PLOTS_DIR, f'runtime_vs_{param}.png'))
        plt.close()

        # noise plots for facet 0 and facet 1
        if 'noise0' in sub and sub['noise0'].notnull().any():
            plt.figure()
            plt.plot(sub['value'], sub['noise0'], marker='o')
            plt.title(f'Insolation Noise (facet0) vs {param}')
            plt.xlabel(param)
            plt.ylabel('Noise0 (std of insolation)')
            plt.savefig(os.path.join(PLOTS_DIR, f'noise0_vs_{param}.png'))
            plt.close()
        if 'noise1' in sub and sub['noise1'].notnull().any():
            plt.figure()
            plt.plot(sub['value'], sub['noise1'], marker='o')
            plt.title(f'Insolation Noise (facet1) vs {param}')
            plt.xlabel(param)
            plt.ylabel('Noise1 (std of insolation)')
            plt.savefig(os.path.join(PLOTS_DIR, f'noise1_vs_{param}.png'))
            plt.close()

    print(f"Generated diagnostic plots in {PLOTS_DIR}")


if __name__ == '__main__':
    run_experiments() 