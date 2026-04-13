import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_three_tier_performance():
    path = 'data/output/stress_test_all_new.csv'
    
    python_val = 224939.2912991128
    serial_val = 19881532.274669345
    parallel_val = 78579168.2317951

    if not os.path.exists(path):
        print(f"Error: {path} not found. Run batch_all.py first!")
        return
    
    df = pd.read_csv(path)

    T, J = 2.27, 1.0
    df['complexity'] = (df['L']**2) * np.log(T/J) * df['total_steps']

    df['Python Throughput'] = df['total_steps'] / df['python_time']
    df['Serial Cython Throughput'] = df['total_steps'] / df['serial_time']
    
    df['Parallel Optimized Throughput'] = (df['total_steps'] * df['parallel_boost']) / df['serial_time']

    plot_df = df.melt(
        id_vars=['complexity', 'L'], 
        value_vars=['Python Throughput', 'Serial Cython Throughput', 'Parallel Optimized Throughput'],
        var_name='Engine', 
        value_name='Throughput'
    )

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    scatter = sns.scatterplot(
        data=plot_df, 
        x='complexity', 
        y='Throughput', 
        hue='Engine',
        size='L',
        sizes=(20, 200),
        palette=['#95a5a6', '#3498db', '#e74c3c'], # Grey, Blue, Red
        alpha=0.6, 
        edgecolor='w'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Complexity Metric: $L^2 \cdot \ln(T/J) \cdot \text{steps}$', fontsize=13)
    ax.set_ylabel('Throughput (Spin-Flips / Second)', fontsize=13)
    ax.set_title('Architectural Performance Scaling: Python vs. Serial Cython vs. Parallel 1D', fontsize=16, pad=20)
    ax.axhline(y=python_val, color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=serial_val, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=parallel_val, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8)
    
    plt.legend(title="Execution Engine / Lattice Size", bbox_to_anchor=(1.02, 1), loc='upper left')

    os.makedirs('docs', exist_ok=True)
    save_path = 'docs/three_tier_performance_mean.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Professional scaling plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_three_tier_performance()