import pickle
from flask import Flask, request, render_template
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the decision tree model from the .pkl file
with open('dt_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load historical data
historical_data = pd.read_csv('clean_data-3.csv')  # Load your dataset
# After loading historical data
historical_data['Customer ID'] = historical_data['Customer ID'].astype(int)

historical_data['Purchase Date'] = pd.to_datetime(historical_data['Purchase Date'])  # Ensure date is in datetime format
print(historical_data['Customer ID'].unique())


# Print the column names for debugging
print("Columns in historical data:", historical_data.columns.tolist())
#dd
# Define the home route to render the landing page
@app.route('/')
def home():
    return render_template('home.html')

# Define the form route to render the HTML form
@app.route('/form')
def form():
    return render_template('index.html')

# Define a route to handle predictions and redirect to the result page
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the HTML form
    form_data = request.form

    # Extract fields from form
    customer_id = int(form_data['customer_id'])
    purchase_date = pd.to_datetime(form_data['purchase_date'])
    total_purchase_amount = float(form_data['total_purchase_amount'])
    customer_age = int(form_data['customer_age'])
    returns = int(form_data['returns'])
    product_category = form_data['product_category']
    payment_method = form_data['payment_method']
    gender = form_data['gender']

    # Print the input for debugging
    print("Input Data:", customer_id, purchase_date, total_purchase_amount)

    print(f"Searching for Customer ID: {customer_id}")

    # Assume you have a max date (current date) as a datetime object
    current_date = pd.to_datetime('today')  # Get today's date

    # Calculate Recency
    recency = (current_date - purchase_date).days

    # Get the Frequency value
    customer_data = historical_data[historical_data['Customer ID'] == customer_id]
    
    if not customer_data.empty:
        frequency = customer_data['Frequency'].values[0]
        
        # Calculate Average Return Rate
        total_returns = customer_data['Returns'].sum()
        total_transactions = frequency  # Using frequency as the total number of transactions
        average_return_rate = total_returns / total_transactions if total_transactions > 0 else 0
    else:
        print(f"No data found for Customer ID: {customer_id}. Setting frequency to 0.")
        frequency = 0  # Default value if no data found
        average_return_rate = 0  # Default value if no customer data

    # Assuming total_purchase_amount is used for Monetary
    monetary = total_purchase_amount

    # Calculate Average Purchase Value
    average_purchase_value = monetary / (frequency if frequency > 0 else 1)

    # Calculate Churn Risk Score
    churn_risk_score = (recency * -1 + frequency + monetary) / 3

    # Customer Engagement Level
    if recency <= 30 and frequency > 5 and monetary > 500:
        customer_engagement_level = 2
    elif recency <= 90 and frequency > 2:
        customer_engagement_level = 1
    else:
        customer_engagement_level = 0

    # Determine if it's a holiday season or promotional month
    is_holiday_season = 1 if purchase_date.month in [11, 12] else 0
    is_promotional_month = 1 if purchase_date.month in [4, 5] else 0

    # Create a DataFrame for input to the model
    input_data = pd.DataFrame([{
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'Average Purchase Value': average_purchase_value,
        'Churn Risk Score': churn_risk_score,
        'Customer Engagement Level': customer_engagement_level,
        'Is Holiday Season': is_holiday_season,
        'Is Promotional Month': is_promotional_month,
        'Average Return Rate': average_return_rate
    }])

    # Make prediction using the decision tree model
    prediction = model.predict(input_data)[0]
    result_text = 'Likely to Churn' if prediction == 1 else 'Not Likely to Churn'

    # Pass the result to the result.html template
    return render_template('result.html', prediction=result_text)


if __name__ == '__main__':
    app.run(debug=True)
