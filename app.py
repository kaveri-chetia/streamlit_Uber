import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Uber Fare Prediction", layout="wide")

# Function to load data
def load_data():
    try:
        df = pd.read_csv("./data/cleaned_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

df = load_data()

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the trained model
with open('gradientboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a dictionary of popular locations with their corresponding latitude and longitude
locations = {
    "Times Square, New York, NY": (40.7580, -73.9855),
    "Central Park, New York, NY": (40.7851, -73.9683),
    "Empire State Building, New York, NY": (40.748817, -73.985428),
    "Brooklyn Bridge, New York, NY": (40.706086, -73.996864),
    "Statue of Liberty, New York, NY": (40.689247, -74.044502),
    "Metropolitan Museum of Art, New York, NY": (40.779437, -73.963244),
    "Grand Central Terminal, New York, NY": (40.752726, -73.977229),
    "One World Trade Center, New York, NY": (40.712743, -74.013379),
    "Rockefeller Center, New York, NY": (40.7587, -73.9787),
    "Wall Street, New York, NY": (40.7064, -74.0094),
    "Broadway Theatre District, New York, NY": (40.7590, -73.9845),
    "Bryant Park, New York, NY": (40.753596, -73.983233),
    "Chinatown, New York, NY": (40.715751, -73.997031),
    "Flatiron Building, New York, NY": (40.741061, -73.989699),
    "SoHo, New York, NY": (40.7233, -74.0020),
    "Washington Square Park, New York, NY": (40.730823, -73.997332),
    "Fifth Avenue, New York, NY": (40.775036, -73.964926),
    "Madison Square Garden, New York, NY": (40.750504, -73.993438),
    "Union Square, New York, NY": (40.735863, -73.991084),
    "St. Patrick's Cathedral, New York, NY": (40.758465, -73.975993),
    "The High Line, New York, NY": (40.7480, -74.0048),
    "Columbus Circle, New York, NY": (40.7681, -73.9819),
    "Carnegie Hall, New York, NY": (40.7651, -73.9799),
    "Battery Park, New York, NY": (40.7033, -74.0170),
    "Chelsea Market, New York, NY": (40.7423, -74.0060),
    "New York Public Library, New York, NY": (40.7532, -73.9822),
    "Tribeca, New York, NY": (40.7163, -74.0086),
    "West Village, New York, NY": (40.7358, -74.0036),
    "Harlem, New York, NY": (40.8116, -73.9465),
    "Greenwich Village, New York, NY": (40.7336, -74.0027),
    "Little Italy, New York, NY": (40.719141, -73.997327),
    "Coney Island, Brooklyn, NY": (40.574926, -73.985941),
    "Williamsburg, Brooklyn, NY": (40.7081, -73.9571),
    "Prospect Park, Brooklyn, NY": (40.6602, -73.9690),
    "Brooklyn Heights Promenade, Brooklyn, NY": (40.6955, -73.9969),
    "DUMBO, Brooklyn, NY": (40.7033, -73.9893),
    "Flushing Meadows-Corona Park, Queens, NY": (40.7498, -73.8447),
    "Astoria, Queens, NY": (40.7644, -73.9235),
    "Jackson Heights, Queens, NY": (40.7557, -73.8831),
    "Long Island City, Queens, NY": (40.7447, -73.9485),
    "Yankee Stadium, Bronx, NY": (40.8296, -73.9262),
    "Bronx Zoo, Bronx, NY": (40.8506, -73.8760),
    "Arthur Avenue, Bronx, NY": (40.8548, -73.8870),
    "Pelham Bay Park, Bronx, NY": (40.8674, -73.8057),
    "Staten Island Ferry Terminal, Staten Island, NY": (40.6437, -74.0735),
    "St. George Theatre, Staten Island, NY": (40.6442, -74.0761),
    "Snug Harbor Cultural Center, Staten Island, NY": (40.6437, -74.1001),
    "Richmond Town, Staten Island, NY": (40.5700, -74.1453),
    "South Beach, Staten Island, NY": (40.5830, -74.0722),
    "Fort Wadsworth, Staten Island, NY": (40.6062, -74.0579)
}

page = st.sidebar.radio("Go to", ["Home", "Predict Fare", "Data Insights"])

if page == "Home":
    # Display the title and introductory text
    st.title("Welcome to Uber Fare Prediction App!")
    st.write("Navigate through the app using the sidebar to predict fares or explore data insights.")

    # Display an image
    st.image("Uber fare.jpg", width=500)

    # Provide a brief description of the app
    st.markdown("""
    The **Uber Fare Prediction App** is a user-friendly web application designed to predict the cost of an Uber ride based on various factors such as the pickup and dropoff locations,
    the number of passengers, the distance of the trip, and the time of the ride. Leveraging advanced machine learning algorithms, the app provides accurate fare estimates, helping users get an idea of the potential cost before booking their ride.
    """)
    

elif page == "Predict Fare":

    if not df.empty:

        st.image("images.png", width=500)

        st.sidebar.header("Please enter your details")

        # User input for pickup and dropoff locations from dropdowns
        pickup_location = st.sidebar.selectbox('Pickup Location', list(locations.keys()), index=0)
        dropoff_location = st.sidebar.selectbox('Dropoff Location', list(locations.keys()), index=1)

        # Get latitude and longitude for the selected locations
        pickup_latitude, pickup_longitude = locations[pickup_location]
        dropoff_latitude, dropoff_longitude = locations[dropoff_location]

        # Get other inputs
        passenger_count = st.sidebar.number_input('Passenger Count', min_value=1, max_value=10, value=1)
        year = st.sidebar.number_input('Year', min_value=2000, max_value=2100, value=2024)
        trip_distance_km = st.sidebar.number_input('Trip Distance (km)', value=0.0)

        # Dropdowns for Hour, Day, and Month
        hour = st.sidebar.selectbox('Hour', list(range(0, 24)))
        day = st.sidebar.selectbox('Day of the Week', list(range(1, 8)))  # Assuming 1=Monday, 7=Sunday
        month = st.sidebar.selectbox('Month', list(range(1, 13)))

        # Convert hour, day, and month to sine and cosine
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # Prepare the feature inputs for prediction
        feature_inputs = [
            pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, 
            passenger_count, year, trip_distance_km,
            month_sin, month_cos, hour_sin, hour_cos, day_sin, day_cos
        ]

        # Convert user input to a numpy array
        input_data = np.array([feature_inputs])

        # Scale the user input using the same scaler
        input_data_scaled = scaler.transform(input_data)

        # Add a button to predict the fare
        if st.button("Predict Fare"):
            # Make predictions
            prediction = model.predict(input_data_scaled)

            # Display the prediction
            st.header("Prediction")
            st.write(f'The predicted fare is: ${prediction[0]:.2f}')

            

elif page == "Data Insights":
    # Data Insights Page
    st.header("Data Insights")

    # Main content
    col1, col2 = st.columns([2,2])

    with col1:
        # Total passenger count
        st.subheader("Passenger count details")
        passenger_count = df['passenger_count'].value_counts()
        fig1 = px.bar(passenger_count, x=passenger_count.index, y=passenger_count.values,
                    labels={'x': 'Passenger'})
        
        st.plotly_chart(fig1)

    with col2:
        # Peak hour of the day
        st.subheader("Peak Hour")
        hourly_counts = df['Hour'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette="viridis")
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Rides')
        plt.title('Number of Rides per Hour (Peak Hours)')
        st.pyplot(plt.gcf())  # Use st.pyplot to display matplotlib plots

    # Additional insights
    st.subheader("Key Insights")
    col3, col4, col5 = st.columns(3)

    with col3:
        total_revenue = df['fare_amount'].sum()
        st.metric("Total Revenue", f"${total_revenue:.2f}")

    with col4:
        trip_distance = df['trip_distance_km'].mean()
        st.metric("Average Trip Distance", f"{trip_distance:.2f} km")

    with col5:
        avg_price = df['fare_amount'].mean()
        st.metric("Average Fare", f"${avg_price:.2f}")

    # Create the scatter mapbox plot
    st.subheader("Pick up location")
    fig = px.scatter_mapbox(
        df,
        lat="pickup_latitude",
        lon="pickup_longitude",
        zoom=3,
        height=400
    )

    # Set the mapbox style (you can choose from various map styles)
    fig.update_layout(mapbox_style="open-street-map")

    # Show the plot
    st.plotly_chart(fig)    

    # Raw data
    st.subheader("Raw Data")
    st.dataframe(df)


    st.subheader("Monthly statistics")
    monthly_counts = df['Month'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette="viridis")
    plt.xlabel('Months')
    plt.ylabel('Number of Rides')
    plt.title('Distribution of rides through out the months')
    st.pyplot(plt.gcf())


    

def plot_with_colorful_background(df):
    # Create a figure with a custom background color
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('lightyellow')  # Background color for the entire figure

    # Plot Average Fare by Hour of the Day using Seaborn
    average_fare_by_hour = df.groupby('Hour')['fare_amount'].mean().reset_index()
    sns.lineplot(data=average_fare_by_hour, x='Hour', y='fare_amount', marker='o', color='blue', linewidth=2.5, ax=ax)

    # Customizing the plot
    ax.set_title('Average Fare Amount by Hour of the Day', fontsize=16, color='darkblue')
    ax.set_xlabel('Hour of the Day', fontsize=14, color='darkblue')
    ax.set_ylabel('Average Fare Amount ($)', fontsize=14, color='darkblue')
    ax.set_xticks(range(0, 24))
    ax.grid(True, color='lightgray')
    ax.set_facecolor('lightcyan')  # Background color for the plotting area
    st.pyplot(fig)

    # Create a figure with a custom background color for the second plot
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('lightpink')  # Background color for the entire figure

    # Plot Average Fare by Month of the Year using Seaborn
    average_fare_by_month = df.groupby('Month')['fare_amount'].mean().reset_index()
    sns.lineplot(data=average_fare_by_month, x='Month', y='fare_amount', marker='o', color='green', linewidth=2.5, ax=ax)

    # Customizing the plot
    ax.set_title('Average Fare Amount by Month of the Year', fontsize=16, color='darkgreen')
    ax.set_xlabel('Month', fontsize=14, color='darkgreen')
    ax.set_ylabel('Average Fare Amount ($)', fontsize=14, color='darkgreen')
    ax.set_xticks(range(1, 13))
    ax.grid(True, color='lightgray')
    ax.set_facecolor('lightyellow')  # Background color for the plotting area
    st.pyplot(fig)

st.subheader("Time Analysis")
plot_with_colorful_background(df)