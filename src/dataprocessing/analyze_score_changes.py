import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_score_changes():
    """
    Analyze score changes in test_pairs.csv:
    - Count cases where at least one score improves
    - Calculate percentages
    - Create visualizations
    """
    # Read data
    df = pd.read_csv('data/test_pairs.csv')
    
    # Calculate score changes
    df['ADAS11_change'] = df['ADAS11_future'] - df['ADAS11_now']
    df['ADAS13_change'] = df['ADAS13_future'] - df['ADAS13_now']
    df['MMSCORE_change'] = df['MMSCORE_future'] - df['MMSCORE_now']
    
    # Count cases
    total_cases = len(df)
    
    # Count improvements (ADAS decrease or MMSCORE increase)
    adas11_improved = len(df[df['ADAS11_change'] < 0])
    adas13_improved = len(df[df['ADAS13_change'] < 0])
    mmse_improved = len(df[df['MMSCORE_change'] > 0])
    
    # Count cases that improved in at least one score
    improved_in_any = len(df[
        (df['ADAS11_change'] < 0) | 
        (df['ADAS13_change'] < 0) | 
        (df['MMSCORE_change'] > 0)
    ])
    
    # Print results
    print("\nScore Improvement Analysis:")
    print("-" * 50)
    print(f"Total cases: {total_cases}")
    print("\nIndividual Improvements:")
    print(f"ADAS11 improved: {adas11_improved} ({adas11_improved/total_cases*100:.1f}%)")
    print(f"ADAS13 improved: {adas13_improved} ({adas13_improved/total_cases*100:.1f}%)")
    print(f"MMSCORE improved: {mmse_improved} ({mmse_improved/total_cases*100:.1f}%)")
    print("\nOverall Improvement:")
    print(f"Improved in at least one score: {improved_in_any} ({improved_in_any/total_cases*100:.1f}%)")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Score change distributions
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='ADAS11_change', bins=30)
    plt.axvline(x=0, color='r', linestyle='--', label='No change')
    plt.title('ADAS11 Score Changes')
    plt.xlabel('Change (Future - Now)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='ADAS13_change', bins=30)
    plt.axvline(x=0, color='r', linestyle='--', label='No change')
    plt.title('ADAS13 Score Changes')
    plt.xlabel('Change (Future - Now)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='MMSCORE_change', bins=30)
    plt.axvline(x=0, color='r', linestyle='--', label='No change')
    plt.title('MMSCORE Changes')
    plt.xlabel('Change (Future - Now)')
    plt.ylabel('Count')
    plt.legend()
    
    # 2. Scatter plots
    plt.subplot(2, 2, 4)
    plt.scatter(df['ADAS11_now'], df['ADAS11_future'], alpha=0.5)
    plt.plot([0, 70], [0, 70], 'r--', label='No change')  # Diagonal line
    plt.title('ADAS11 Now vs Future')
    plt.xlabel('ADAS11 Now')
    plt.ylabel('ADAS11 Future')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data/score_changes_analysis.png')
    plt.close()
    
    # Additional analysis: Time between tests
    print("\nTime Analysis:")
    print("-" * 50)
    print(f"Average time between tests: {df['time_lapsed'].mean():.1f} days")
    print(f"Median time between tests: {df['time_lapsed'].median():.1f} days")
    print(f"Min time between tests: {df['time_lapsed'].min():.1f} days")
    print(f"Max time between tests: {df['time_lapsed'].max():.1f} days")
    
    # Analyze improvement by time ranges
    print("\nImprovement Analysis by Time Ranges:")
    print("-" * 50)
    time_ranges = [(180, 360), (360, 540)]
    
    for start, end in time_ranges:
        mask = (df['time_lapsed'] >= start) & (df['time_lapsed'] < end)
        subset = df[mask]
        if len(subset) > 0:
            print(f"\nTime range: {start}-{end} days ({len(subset)} cases):")
            print(f"ADAS11 improved: {len(subset[subset['ADAS11_change'] < 0])} ({len(subset[subset['ADAS11_change'] < 0])/len(subset)*100:.1f}%)")
            print(f"ADAS13 improved: {len(subset[subset['ADAS13_change'] < 0])} ({len(subset[subset['ADAS13_change'] < 0])/len(subset)*100:.1f}%)")
            print(f"MMSCORE improved: {len(subset[subset['MMSCORE_change'] > 0])} ({len(subset[subset['MMSCORE_change'] > 0])/len(subset)*100:.1f}%)")

if __name__ == "__main__":
    analyze_score_changes() 