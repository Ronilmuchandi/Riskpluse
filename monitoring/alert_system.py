import json
from datetime import datetime

def check_alerts(drift_report, drift_threshold=30.0, accuracy_threshold=0.90):
    alerts = []
    drift_pct = drift_report['summary']['drift_percentage']
    if drift_pct > drift_threshold:
        alerts.append({
            'type': 'DRIFT_ALERT',
            'severity': 'HIGH' if drift_pct > 50 else 'MEDIUM',
            'message': f'{drift_pct}% of features have drifted — model retraining recommended',
            'timestamp': datetime.now().isoformat()
        })
    return alerts

if __name__ == '__main__':
    with open('data/processed/drift_report.json', 'r') as f:
        drift_report = json.load(f)

    alerts = check_alerts(drift_report)
    if alerts:
        for alert in alerts:
            print(f"[{alert['severity']}] {alert['type']}: {alert['message']}")
    else:
        print("No alerts — model is healthy")

    with open('data/processed/alerts.json', 'w') as f:
        json.dump(alerts, f, indent=2)
    print("Alerts saved to data/processed/alerts.json")