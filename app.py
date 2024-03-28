import base64
import io
import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.spatial import distance
import matplotlib.pyplot as plt

app = Flask(__name__)

# Ensure that Matplotlib works in non-interactive mode
plt.switch_backend('agg')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/train')
def train():
    # STEP 1: Load Input dataset
    dataset = pd.read_csv('D:\Dummy\data\KDD Cup 1999 imbalanced Data with 10000 records.csv')

    # STEP 2: Convert categorical to numerical data using Label Encoding technique
    label_encoder = preprocessing.LabelEncoder()
    cols = ['protocol_type', 'service', 'flag', 'class']
    dataset[cols] = dataset[cols].apply(label_encoder.fit_transform)

    X = dataset.iloc[:, :-1]   # data without class label
    Y = dataset.iloc[:, 41].values     # class label values only

    # STEP 3: Transform the dataset using Synthetic Minority Oversampling Technique (SMOTE)
    oversample = SMOTE(k_neighbors=1)
    X1, Y1 = oversample.fit_resample(X, Y)

    # STEP 4: Split dataset into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)

    # STEP 5: Principal Component Analysis (PCA) for feature selection and dimensionality reduction
    pca = PCA(n_components=1)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # STEP 6: Predict Intrusion with Attack Type using Principal Component Analysis with Random Forest
    PCARF = RandomForestClassifier()
    PCARF.fit(X_train, Y_train)
    Y_pred_pca_rf = PCARF.predict(X_test)
    pca_rf_accuracy = (1 - distance.cosine(Y_test, Y_pred_pca_rf)) * 100

    # STEP 7: Predict Intrusion with Attack Type using Support Vector Machine
    svm = SVC()
    svm.fit(X_train, Y_train)
    Y_pred_svm = svm.predict(X_test)
    svm_accuracy = (1 - distance.cosine(Y_test, Y_pred_svm)) * 100

    # STEP 8: Generate CSV file with actual and predicted class labels
    actual_predicted_data = {'Actual': Y_test, 'PCA-RF Predicted': Y_pred_pca_rf}
    df = pd.DataFrame(actual_predicted_data)
    df.to_csv('actual_predicted_labels.csv', index=False)
   

    # Dump the trained PCA model into a pickle file
    with open('pca_model.pkl', 'wb') as file:
        pickle.dump(pca, file)

    # Dump the trained PCA-RF model into a pickle file
    with open('pca_rf_model.pkl', 'wb') as file:
        pickle.dump(PCARF, file)

    # Generate Accuracy Comparison Plot
    plt.figure()
    x1 = ['SVM', 'PCA-RF']
    y1 = [svm_accuracy, pca_rf_accuracy]
    plt.plot(x1, y1, label="Accuracy")
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy (in %)')
    plt.title('Accuracy Comparison')
    plt.legend()

    # Convert plot to base64 encoding
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Clear the plot to prevent it from being reused in subsequent requests
    plt.clf()

    return render_template('train.html',
                           pca_rf_accuracy=pca_rf_accuracy,
                           svm_accuracy=svm_accuracy,
                           plot_url=plot_url)

@app.route('/test', methods=['GET', 'POST'])
def test():
    # Load the CSV file with actual and predicted class labels
    dataset = pd.read_csv(r'D:\Anomaly Detection\actual_predicted_labels.csv')
    
    # Convert the dataset to HTML format
    data_html = dataset.to_html(index=False)

    return render_template('test.html', data_html=data_html)


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')
@app.route('/view_data')
def view_data():
    # Read the CSV file
    dataset = pd.read_csv('D:\Dummy\data\KDD Cup 1999 imbalanced Data with 10000 records.csv')
    
    # Convert the dataset to HTML format
    data_html = dataset.to_html(index=False)

    return render_template('view_data.html', data_html=data_html)


if __name__ == '__main__':
    app.run(debug=True)
