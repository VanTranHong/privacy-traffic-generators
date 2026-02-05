import pandas as pd

import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
## XGBoost
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Initialize CatBoost


# We use the fields port number, protocol,
# bytes/flow, packets/flow, and flow duration


def prepare_data(folder):
    df_lst = []
    for fn in glob.glob(folder):
        df = pd.read_csv(fn)
        df = df.iloc[:3]
        drop_columns = ["src_ip"] + [col for col in df.columns if "prt" in col ]
        # print("Dropping columns:", drop_columns)
        df = df.drop(columns=drop_columns, errors='ignore')
        df_flat = df.stack().to_frame().T
        # or with better column names:
        df_flat = pd.DataFrame([df.values.flatten()], 
                            columns=[f'{col}_{i}' for i in range(len(df)) for col in df.columns])

        df_flat['label'] = [fn.split("/")[-2].split("_")[-1]]  # extract label from folder name
        df_lst.append(df_flat)
    df_all = pd.concat(df_lst, ignore_index=True)
    return df_all
    # print(df_all['label'].value_counts())
    # print("Data shape:", df_all.shape)
    # print("Columns:", df_all.columns)

def prepare_data_real(folder):
    df_label = pd.read_csv(folder+"labels.csv")
    label_lst = []
    df_lst = []
    for i,row in df_label.iterrows():
        fname = row["File"]
        label = row["Label"]
        label_lst.append(label)
        df = pd.read_csv(folder+fname.replace(".pcap", ".nprint"))
        df = df.iloc[:3]
        drop_columns = ["src_ip"] + [col for col in df.columns if "prt" in col ]
        # print("Dropping columns:", drop_columns)
        df = df.drop(columns=drop_columns, errors='ignore')
        df_flat = df.stack().to_frame().T
        # or with better column names:
        df_flat = pd.DataFrame([df.values.flatten()], 
                            columns=[f'{col}_{i}' for i in range(len(df)) for col in df.columns])
        df_lst.append(df_flat)
    df_all = pd.concat(df_lst, ignore_index=True)
    df_all['label'] = label_lst
    return df_all    


def prepare_real_stats(folder):
    """generate statistics from each flow to be used in classification"""
    df = pd.read_csv(folder+"labels.csv")
    print("Preparing real data stats from folder:", folder)
    label_lst = []
    df_lst = []
    for i,row in df.iterrows():
        fname = row["File"]
        label = row["Label"]
        
        try:
            df_flow = pd.read_csv(folder+fname.replace(".pcap", "_flows.csv"))
        
            
            columns = ["Total Packets","Total Bytes","Fwd Packets","Bwd Packets","Fwd Bytes","Bwd Bytes", "Flow Duration","Avg Packet Size", "Std Packet Size","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Packet Rate","Byte Rate"]
            df_flat = df_flow[columns].iloc[:1]
            df_lst.append(df_flat)
            label_lst.append(label)
        except Exception as e:
            print(f"Error processing file {fname}: {e}")
            continue
    df_all = pd.concat(df_lst, ignore_index=True)
    df_all['label'] = label_lst
    return df_all

   

def prepare_synthetic_stats(folder):
    """generate statistics from each flow to be used in classification"""
    df_lst = []
    for fn in glob.glob(folder):
        df = pd.read_csv(fn)
        columns = ["Total Packets","Total Bytes","Fwd Packets","Bwd Packets","Fwd Bytes","Bwd Bytes", "Flow Duration","Avg Packet Size", "Std Packet Size","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Packet Rate","Byte Rate"]
        df_flat = df[columns].iloc[:1]
        df_flat['label'] = [fn.split("/")[-2].split("_")[-1]]  # extract label from folder name
        df_lst.append(df_flat)
    df_all = pd.concat(df_lst, ignore_index=True)
    return df_all
    
# dataset = "servicerecognition"
dataset = "IoT" 
# caption = "_completeanonymized"
# caption = "_identitymasked"
# caption = "_subnet"
# caption = "_epsilon_01_new"
# caption = "_epsilon_1_new"
caption = ""
# synthetic = f"/net/scratch/vantran/code/netssm/inference/{dataset}_singleflow_truncated_1epochs{caption}/*/singleprompt_*_flows.csv"
synthetic = f"/net/scratch/vantran/code/netssm/inference/{dataset}_singleflow_truncated_1epochs{caption}/*/singleprompt_*.nprint"

train_real = f"/net/scratch/vantran/code/netssm_old/data_MIA_new/single_{dataset}/new_data/train/"
test_real = f"/net/scratch/vantran/code/netssm_old/data_MIA_new/single_{dataset}/new_data/test/"
# test_real_df = prepare_real_stats(test_real)

# synthetic_df = prepare_synthetic_stats(synthetic)
# train_real_df = prepare_real_stats(train_real)


test_real_df = prepare_data_real(test_real)
synthetic_df = prepare_data(synthetic)
train_real_df = prepare_data_real(train_real)

print("Train real labels:")
print(train_real_df['label'].value_counts())
print("Test real labels:")
print(test_real_df['label'].value_counts())
print("Synthetic labels:")
print(synthetic_df['label'].value_counts())

labels_not_in = [label for label in synthetic_df['label'].unique() if label not in test_real_df['label'].unique()]
if len(labels_not_in) > 0:
    print("Labels in synthetic but not in real test data:", labels_not_in)
    synthetic_df = synthetic_df[~synthetic_df['label'].isin(labels_not_in)]
    print("After removal, synthetic labels:")
    print(synthetic_df['label'].value_counts())


# labels_all = set(train_real_df['label'].unique()).union(set(test_real_df['label'].unique()))
labels_all = set(test_real_df['label'].unique()).union(set(synthetic_df['label'].unique()))
label_to_idx = {label: idx for idx, label in enumerate(labels_all)}
train_real_df['label'] = train_real_df['label'].map(label_to_idx)
test_real_df['label'] = test_real_df['label'].map(label_to_idx)
synthetic_df['label'] = synthetic_df['label'].map(label_to_idx)

# ## train_real
X_train_real = train_real_df.drop(columns=['label'])
# print("Real training data shape:", train_real_df.shape)
y_train_real = train_real_df['label']
# test_real
X_test_real = test_real_df.drop(columns=['label'])
# print("Real testing data shape:", test_real_df.shape)
y_test_real = test_real_df['label']
# ## train_synthetic
X_train_synthetic = synthetic_df.drop(columns=['label'])
# print("Synthetic data shape:", synthetic_df.shape)
y_train_synthetic = synthetic_df['label'] 



## train XGB on syntehtic, test on real
xgb_syn = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_syn.fit(X_train_real, y_train_real)
y_pred_real = xgb_syn.predict(X_test_real)
print("XGB trained on real, tested on real:")
# print(classification_report(y_test_real, y_pred_syn, zero_division=0, digits=4))
report = classification_report(y_test_real, y_pred_real, output_dict=True)
accuracy = report['accuracy']
print(dataset)
print(caption)
print(f"Accuracy: {accuracy:.4f}")


xgb_syn = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_syn.fit(X_train_synthetic, y_train_synthetic)
y_pred_syn = xgb_syn.predict(X_test_real)
print("XGB trained on synthetic, tested on real:")
# print(classification_report(y_test_real, y_pred_syn, zero_division=0, digits=4))
report = classification_report(y_test_real, y_pred_syn, output_dict=True)
accuracy = report['accuracy']
print(dataset)
print(caption)
print(f"Accuracy: {accuracy:.4f}")






# model = CatBoostClassifier(
#     iterations=1000,           # Number of boosting iterations
#     learning_rate=0.1,         # Learning rate
#     depth=6,                   # Tree depth
#     loss_function='MultiClass', # For multi-class classification
#     verbose=100,               # Print progress every 100 iterations
#     random_seed=42
# )
# print(dataset)
# model.fit(X_train_synthetic, y_train_synthetic)
# y_pred_syn = model.predict(X_test_real)
# print("CatBoost trained on synthetic, tested on real:")
# report = classification_report(y_test_real, y_pred_syn, output_dict=True)
# accuracy = report['accuracy']
# print(f"Accuracy: {accuracy:.4f}")

# model_real = CatBoostClassifier(
#     iterations=1000,           # Number of boosting iterations
#     learning_rate=0.1,         # Learning rate
#     depth=6,                   # Tree depth
#     loss_function='MultiClass', # For multi-class classification
#     verbose=100,               # Print progress every 100 iterations
#     random_seed=42
# )
# model_real.fit(X_train_real, y_train_real)
# y_pred_real = model_real.predict(X_test_real)
# print("CatBoost trained on real, tested on real:")
# report = classification_report(y_test_real, y_pred_real, output_dict=True)
# accuracy = report['accuracy']
# print(f"Accuracy: {accuracy:.4f}")




