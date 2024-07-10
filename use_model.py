import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Feature descriptions
feature_descriptions = {
    "state": "Indicates the state and its dependent protocol (nominal, e.g. ACC, CLO, etc.)",
    "rate": "Transaction rate (float)",
    "sttl": "Source to destination time to live value (integer)",
    "dload": "Destination bits per second (float)",
    "swin": "Source TCP window advertisement value (integer)",
    "stcpb": "Source TCP base sequence number (integer)",
    "dtcpb": "Destination TCP base sequence number (integer)",
    "dwin": "Destination TCP window advertisement value (integer)",
    "dmean": "Mean of the flow packet size transmitted by the destination (integer)",
    "ct_state_ttl": "Number of connections with the same state according to specific TTL range (integer)",
    "ct_src_dport_ltm": "Number of connections of the same source address and destination port in 100 connections (integer)",
    "ct_dst_sport_ltm": "Number of connections of the same destination address and source port in 100 connections (integer)",
    "ct_dst_src_ltm": "Number of connections of the same source and destination address in 100 connections (integer)",
}

# Selected features will be used for prediction
selected_features = [
    'state', 'rate', 'sttl', 'dload', 'swin', 'stcpb', 'dtcpb', 'dwin', 
    'dmean', 'ct_state_ttl', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
    'ct_dst_src_ltm'
]

# Function to get user input for each selected feature
def get_user_input():
    user_input = {}
    for feature in selected_features:
        if feature in feature_descriptions:
            value = input(f"Enter value for {feature} ({feature_descriptions[feature]}): ")
            user_input[feature] = value
    return user_input

# Main function
def main():
    # Loading the saved SVM model
    model_path = 'Model/svm_model_kernel.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    
    user_input = get_user_input()

    # Converting the user input to the correct data format 
    input_data = []
    for feature in selected_features:
        if feature in user_input:
            try:
                value = float(user_input[feature]) if '.' in user_input[feature] else int(user_input[feature])
            except ValueError:
                value = user_input[feature]  
            input_data.append(value)
        else:
            input_data.append(0)  

    # Converting the input data to numpy array
    input_data = np.array(input_data).reshape(1, -1)

    # Normalizing the input data
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    # Making prediction using the SVM model
    prediction = model.predict(input_data)

    # Result
    if prediction[0] == 0:
        print("Prediction: No network intrusion")
    else:
        print("Prediction: Network intrusion")

if __name__ == "__main__":
    main()
