import pandas as pd
import numpy as np
from scapy.all import rdpcap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ipaddress

def ip_to_int(ip):
    """Convert IP address to an integer for machine learning models."""
    return int(ipaddress.IPv4Address(ip))

def capture_network_traffic(pcap_file="traffic.pcapng", output_file="traffic_data.csv"):
    print("Processing captured network traffic...")
    if not os.path.exists(pcap_file):
        print(f"Error: {pcap_file} not found! Please capture network traffic using Wireshark or tcpdump.")
        return
    
    packets = rdpcap(pcap_file)
    data = []
    for pkt in packets:
        if pkt.haslayer("IP"):
            data.append({
                'Source_IP': pkt["IP"].src,
                'Destination_IP': pkt["IP"].dst,
                'Packet_Size': len(pkt),
                'Label': 'Unknown'  # Labeling is manual or ML-based
            })
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Traffic data saved to {output_file}")

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_model(df):
    # Convert IP addresses to integer values
    df['Source_IP'] = df['Source_IP'].apply(ip_to_int)
    df['Destination_IP'] = df['Destination_IP'].apply(ip_to_int)

    # Prepare features and labels
    X = df.drop(['Label'], axis=1)
    y = df['Label'].apply(lambda x: 1 if x == 'Malicious' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

def visualize_data(df):
    sns.pairplot(df, hue='Label')
    plt.show()

def main():
    capture_network_traffic()
    df = load_data("traffic_data.csv")
    if not df.empty:
        visualize_data(df)
        model = train_model(df)

if __name__ == "__main__":
    main()