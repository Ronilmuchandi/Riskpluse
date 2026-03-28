import numpy as np
from scipy import stats
import json
from datetime import datetime

def detect_drift(reference_data, current_data, ks_threshold=0.1):
    drift_report = {}
    for i in range(reference_data.shape[1]):
        ks_stat, p_value = stats.ks_2samp(reference_data[:, i], current_data[:, i])
        drift_report[f'feature_{i}'] = {
            'ks_statistic': round(float(ks_stat), 4),
            'p_value': round(float(p_value), 4),
            'drift_detected': bool(ks_stat > ks_threshold)
        }
    drifted = sum(1 for f in drift_report.values() if f['drift_detected'])
    drift_report['summary'] = {
        'total_features': reference_data.shape[1],
        'drifted_features': drifted,
        'drift_percentage': round(drifted / reference_data.shape[1] * 100, 2),
        'timestamp': datetime.now().isoformat()
    }
    return drift_report

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/raw/creditcard.csv')
    df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    df['Time'] = (df['Time'] - df['Time'].mean()) / df['Time'].std()
    X = df.drop(columns=['Class']).values

    split = int(len(X) * 0.8)
    X_reference = X[:split]
    X_current = X[split:]

    X_drifted = X_current.copy()
    X_drifted[:, :10] += np.random.normal(0, 2, (X_drifted.shape[0], 10))

    print("Running drift detection on clean data...")
    clean_report = detect_drift(X_reference, X_current)
    print(f"Drifted features: {clean_report['summary']['drifted_features']}/{clean_report['summary']['total_features']}")

    print("\nRunning drift detection on drifted data...")
    drift_report = detect_drift(X_reference, X_drifted)
    print(f"Drifted features: {drift_report['summary']['drifted_features']}/{drift_report['summary']['total_features']}")

    with open('data/processed/drift_report.json', 'w') as f:
        json.dump(drift_report, f, indent=2)
    print("\nDrift report saved to data/processed/drift_report.json")