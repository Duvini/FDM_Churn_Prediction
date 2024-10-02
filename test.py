import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime

# Load your trained model (Decision Tree Classifier saved in a pickle file)
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)  # Corrected initialization

# Define the route to handle form submissions (POST method)
@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    customer_id = request.form['CustomerID']
    purchase_date = pd.to_datetime(request.form['PurchaseDate'])
    product_category = request.form['ProductCategory']
    total_purchase_amount = float(request.form['TotalPurchaseAmount'])
    payment_method = request.form['PaymentMethod']
    customer_age = int(request.form['CustomerAge'])
    returns = int(request.form['Returns'])
    gender = request.form['Gender']
    
    # Preprocess the data
    current_date = pd.to_datetime('today')
    
    # Calculate features based on inputs
    recency = (current_date - purchase_date).days
    frequency = 1  # Assume 1 transaction for simplicity, or calculate from past data if available
    monetary = total_purchase_amount
    
    average_purchase_value = monetary / frequency
    churn_risk_score = (-recency + frequency + monetary) / 3
    
    # Assuming holidays are in November/December and promotions in April/May
    is_holiday_season = 1 if purchase_date.month in [11, 12] else 0
    is_promotional_month = 1 if purchase_date.month in [4, 5] else 0
    
    # Set dummy engagement level based on arbitrary conditions (You can adjust these)
    customer_engagement_level = 2 if recency <= 30 and frequency > 5 and monetary > 500 else \
                                1 if recency <= 90 and frequency > 2 else 0

    # Average return rate (for simplicity, assume returns divided by transactions)
    average_return_rate = returns / frequency if frequency > 0 else 0
    
    # Create a DataFrame to hold the features
    input_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary],
        'Average Purchase Value': [average_purchase_value],
        'Churn Risk Score': [churn_risk_score],
        'Customer Engagement Level': [customer_engagement_level],
        'Is Holiday Season': [is_holiday_season],
        'Is Promotional Month': [is_promotional_month],
        'Average Return Rate': [average_return_rate]
    })

    # Make predictions
    prediction = model.predict(input_data)
    
    # Return the result
    return jsonify({
        'CustomerID': customer_id,
        'Prediction': 'Churn' if prediction[0] == 1 else 'Not Churn',
        'Features': input_data.to_dict()
    })

# Home route for testing (GET method for rendering the form)
@app.route('/')
def home():
    return render_template('index.html')  # Create a simple HTML form for input

if __name__ == '__main__':
    app.run(debug=True)