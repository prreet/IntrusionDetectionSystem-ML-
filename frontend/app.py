from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load models
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('models/idsMODEL.pkl', 'rb') as f:
    model = pickle.load(f)

rename_map = {
    'Dst Port': ' Destination Port',
    'Flow Duration': ' Flow Duration',
    'Tot Fwd Pkts': ' Total Fwd Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': ' Total Length of Bwd Packets',
    'Fwd Pkt Len Max': ' Fwd Packet Length Max',
    'Fwd Pkt Len Mean': ' Fwd Packet Length Mean',
    'Fwd Pkt Len Std': ' Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': ' Bwd Packet Length Min',
    'Bwd Pkt Len Mean': ' Bwd Packet Length Mean',
    'Bwd Pkt Len Std': ' Bwd Packet Length Std',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': ' Flow Packets/s',
    'Flow IAT Mean': ' Flow IAT Mean',
    'Flow IAT Std': ' Flow IAT Std',
    'Flow IAT Max': ' Flow IAT Max',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Fwd IAT Mean': ' Fwd IAT Mean',
    'Fwd IAT Std': ' Fwd IAT Std',
    'Fwd IAT Max': ' Fwd IAT Max',
    'Fwd Header Len': ' Fwd Header Length',
    'Bwd Header Len': ' Bwd Header Length',
    'Bwd Pkts/s': ' Bwd Packets/s',
    'Pkt Len Max': ' Max Packet Length',
    'Pkt Len Mean': ' Packet Length Mean',
    'Pkt Len Std': ' Packet Length Std',
    'Pkt Len Var': ' Packet Length Variance',
    'PSH Flag Cnt': ' PSH Flag Count',
    'ACK Flag Cnt': ' ACK Flag Count',
    'Pkt Size Avg': ' Average Packet Size',
    'Fwd Seg Size Avg': ' Avg Fwd Segment Size',
    'Bwd Seg Size Avg': ' Avg Bwd Segment Size',
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Fwd Byts': ' Subflow Fwd Bytes',
    'Subflow Bwd Byts': ' Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',
    'Init Bwd Win Byts': ' Init_Win_bytes_backward',
    'Fwd Act Data Pkts': ' act_data_pkt_fwd',
    'Fwd Seg Size Min': ' min_seg_size_forward',
    'Idle Mean': 'Idle Mean',
    'Idle Min': ' Idle Min'
}
required_columns = list(rename_map.values())

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        if file:
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='ISO-8859-1')
                
            df.rename(columns=rename_map, inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            df = df[required_columns]
            X_scaled = scaler.transform(df)
            preds = model.predict(X_scaled)
            predictions = encoder.inverse_transform(preds)
    
    return render_template('index.html', predictions=predictions.tolist() if predictions is not None else None)

if __name__ == '__main__':
    app.run(debug=True)
