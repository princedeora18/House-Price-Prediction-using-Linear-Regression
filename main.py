import numpy as np
import pandas as pd
import os
import locale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Setting up locale for formatting Indian numbers (if available)
try:
    locale.setlocale(locale.LC_ALL, 'en_IN')  # Setting locale to Indian format
except:
    # Fallback if 'en_IN' locale isn't available
    locale.setlocale(locale.LC_ALL, '')

# USD to INR conversion rate (adjust as necessary)
USD_TO_INR = 83.0

def format_inr(amount_usd):
    """Convert USD to INR and format the amount with commas."""
    amount_inr = amount_usd * USD_TO_INR
    
    try:
        # Try formatting using locale's formatting
        return f"₹{locale.format_string('%,.2f', amount_inr, grouping=True)}"
    except:
        # Fallback for manual formatting
        amount_str = f"{amount_inr:.2f}"
        integer_part, decimal_part = amount_str.split('.') if '.' in amount_str else (amount_str, '00')
        
        # Implementing Indian number system: 1,23,456.78
        result = ""
        if len(integer_part) <= 3:
            result = integer_part
        else:
            result = integer_part[-3:]
            integer_part = integer_part[:-3]
            
            while integer_part:
                result = integer_part[-2:] + "," + result if len(integer_part) >= 2 else integer_part + "," + result
                integer_part = integer_part[:-2]
                
        return f"₹{result}.{decimal_part}"

def main():
    # Load data from the Ames Housing dataset
    print("Loading data from 'train.csv'...")
    df = pd.read_csv('train.csv')
    
    # Display basic summary and sample of the dataset
    print("\nData Summary:")
    print(df.describe())
    print("\nSample data:")
    print(df.head())
    
    # Checking for necessary columns
    if 'GrLivArea' in df.columns and 'BedroomAbvGr' in df.columns and 'FullBath' in df.columns:
        print("\nUsing the following columns from the Ames Housing dataset:")
        print("- 'GrLivArea' for square footage")
        print("- 'BedroomAbvGr' for number of bedrooms")
        print("- 'FullBath' for number of bathrooms")
        print("- 'SalePrice' for target variable (price)")
        
        # Selecting features and target
        X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
        y = df['SalePrice']
    else:
        # In case the dataset structure is slightly different, try to match appropriate columns
        print("\nAttempting to identify relevant columns...")

        # Detecting best match for square footage
        area_columns = [col for col in df.columns if 'area' in col.lower() or 'sqft' in col.lower() or 'sf' in col.lower()]
        sq_ft_col = area_columns[0] if area_columns else 'LotArea'  # Default if not found
        
        # Detecting best match for bedrooms and bathrooms
        bedroom_columns = [col for col in df.columns if 'bed' in col.lower()]
        bathroom_columns = [col for col in df.columns if 'bath' in col.lower()]
        
        if not bedroom_columns or not bathroom_columns:
            raise ValueError("Couldn't identify the necessary columns for bedrooms or bathrooms.")
        
        print(f"Using the following columns:")
        print(f"- {sq_ft_col} for square footage")
        print(f"- {bedroom_columns[0]} for bedrooms")
        print(f"- {bathroom_columns[0]} for bathrooms")
        print(f"- 'SalePrice' for target variable (price)")
        
        X = df[[sq_ft_col, bedroom_columns[0], bathroom_columns[0]]]
        y = df['SalePrice']
    
    # Handle missing values
    print("\nHandling missing values...")
    X.fillna(X.mean(), inplace=True)
    y.dropna(inplace=True)

    # Split data into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print model coefficients
    print("\nModel Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {format_inr(coef)}")
    print(f"Intercept: {format_inr(model.intercept_)}")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Plot Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted House Prices (in INR)')
    plt.savefig('price_prediction.png')

    # Default prediction (1,200 sqft, 3 bedrooms, 2 bathrooms)
    default_house = pd.DataFrame({
        X.columns[0]: [1200],
        X.columns[1]: [3],
        X.columns[2]: [2]
    })
    default_price = model.predict(default_house)[0]

    print("\nDefault House Price Prediction:")
    print(f"Square Footage: 1,200")
    print(f"Bedrooms: 3")
    print(f"Bathrooms: 2")
    print(f"Predicted Price: {format_inr(default_price)}")

    # Allow user to input custom values
    print("\n--- Custom House Price Prediction ---")
    try:
        # Get user input (or use default values)
        sq_ft = int(input(f"Enter Square Footage [default=1200]: ") or 1200)
        bedrooms = int(input(f"Enter number of Bedrooms [default=3]: ") or 3)
        bathrooms = int(input(f"Enter number of Bathrooms [default=2]: ") or 2)

        # Create DataFrame for prediction
        custom_house = pd.DataFrame({
            X.columns[0]: [sq_ft],
            X.columns[1]: [bedrooms],
            X.columns[2]: [bathrooms]
        })

        # Make prediction for custom house
        custom_price = model.predict(custom_house)[0]
        print("\nCustom House Price Prediction:")
        print(f"Square Footage: {sq_ft}")
        print(f"Bedrooms: {bedrooms}")
        print(f"Bathrooms: {bathrooms}")
        print(f"Predicted Price: {format_inr(custom_price)}")

    except ValueError:
        print("Invalid input. Using default values for prediction.")
        print(f"Default Predicted Price: {format_inr(default_price)}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()