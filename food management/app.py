import streamlit as st
import sqlite3
import pandas as pd

# Database connection
conn = sqlite3.connect("food.db")

st.set_page_config(page_title="Food Wastage Management", layout="wide")

# ----------------------------
# Tabs for Queries & CRUD
# ----------------------------
tab1, tab2 = st.tabs([" SQL Queries & Insights", "✏️ Manage Records (CRUD)"])

# ----------------------------
# Tab 1: Queries & Insights
# ----------------------------
with tab1:
    st.title(" Food Wastage Management Dashboard")

    st.markdown("""
This dashboard answers key questions on **food donations, claims, providers, and receivers**  
using SQL queries. Select a question from the dropdown to view results.
""")

    # Dropdown for queries
    query_options = {
        "1️⃣ How many food providers and receivers are there in each city?": """
            SELECT p.City,
                   COUNT(DISTINCT p.Provider_ID) AS Total_Providers,
                   COUNT(DISTINCT r.Receiver_ID) AS Total_Receivers
            FROM Providers p
            LEFT JOIN Receivers r ON p.City = r.City
            GROUP BY p.City;
        """,

        "2️⃣ Which type of food provider contributes the most food?": """
            SELECT Type, COUNT(*) AS Total_Contributions
            FROM Providers
            GROUP BY Type
            ORDER BY Total_Contributions DESC;
        """,

        "3️⃣ What is the contact information of food providers in a specific city?": """
            SELECT Name, City, Contact
            FROM Providers;
        """,

        "4️⃣ Which receivers have claimed the most food?": """
            SELECT r.Name, COUNT(c.Claim_ID) AS Total_Claims
            FROM Receivers r
            JOIN Claims c ON r.Receiver_ID = c.Receiver_ID
            GROUP BY r.Name
            ORDER BY Total_Claims DESC;
        """,

        "5️⃣ What is the total quantity of food available from all providers?": """
            SELECT SUM(Quantity) AS Total_Quantity
            FROM Food_Listings;
        """,

        "6️⃣ Which city has the highest number of food listings?": """
            SELECT Location, COUNT(*) AS Total_Listings
            FROM Food_Listings
            GROUP BY Location
            ORDER BY Total_Listings DESC;
        """,

        "7️⃣ What are the most commonly available food types?": """
            SELECT Food_Type, COUNT(*) AS Count_Type
            FROM Food_Listings
            GROUP BY Food_Type
            ORDER BY Count_Type DESC;
        """,

        "8️⃣ How many food claims have been made for each food item?": """
            SELECT f.Food_Name, COUNT(c.Claim_ID) AS Total_Claims
            FROM Food_Listings f
            JOIN Claims c ON f.Food_ID = c.Food_ID
            GROUP BY f.Food_Name
            ORDER BY Total_Claims DESC;
        """,

        "9️⃣ Which provider has had the highest number of successful food claims?": """
            SELECT p.Name, COUNT(c.Claim_ID) AS Successful_Claims
            FROM Providers p
            JOIN Food_Listings f ON p.Provider_ID = f.Provider_ID
            JOIN Claims c ON f.Food_ID = c.Food_ID
            WHERE c.Status = 'Completed'
            GROUP BY p.Name
            ORDER BY Successful_Claims DESC;
        """,

        "🔟 What percentage of food claims are completed vs. pending vs. canceled?": """
            SELECT Status,
                   COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Claims) AS Percentage
            FROM Claims
            GROUP BY Status;
        """,

        "1️⃣1️⃣ What is the average quantity of food claimed per receiver?": """
            SELECT r.Name, AVG(f.Quantity) AS Avg_Quantity_Claimed
            FROM Receivers r
            JOIN Claims c ON r.Receiver_ID = c.Receiver_ID
            JOIN Food_Listings f ON c.Food_ID = f.Food_ID
            GROUP BY r.Name;
        """,

        "1️⃣2️⃣ Which meal type (breakfast, lunch, dinner, snacks) is claimed the most?": """
            SELECT Meal_Type, COUNT(c.Claim_ID) AS Total_Claims
            FROM Food_Listings f
            JOIN Claims c ON f.Food_ID = c.Food_ID
            GROUP BY Meal_Type
            ORDER BY Total_Claims DESC;
        """,

        "1️⃣3️⃣ What is the total quantity of food donated by each provider?": """
            SELECT p.Name, SUM(f.Quantity) AS Total_Donated
            FROM Providers p
            JOIN Food_Listings f ON p.Provider_ID = f.Provider_ID
            GROUP BY p.Name
            ORDER BY Total_Donated DESC;
        """,

            "1️⃣4️⃣ Which food items are currently available (not yet claimed)?": """
        SELECT Food_ID, Food_Name, Quantity, Expiry_Date, Location, Food_Type, Meal_Type
        FROM Food_Listings
        WHERE Food_ID NOT IN (
            SELECT Food_ID FROM Claims WHERE Status = 'Completed'
        );
       """

    }

    # Select query
    selected_query = st.selectbox(" Select a question to analyze:", list(query_options.keys()))

    # Execute and display
    if selected_query:
        query = query_options[selected_query]
        df = pd.read_sql(query, conn)
        st.dataframe(df)

         # Auto-generate chart if data is numeric
        try:
            # If there are exactly 2 columns (category + numeric)
            if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
                st.bar_chart(df.set_index(df.columns[0]))
            # If there are >2 columns but at least one numeric column
            elif any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns[1:]):
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                st.bar_chart(df.set_index(df.columns[0])[numeric_cols])

        except Exception as e:
            st.warning(f" Could not generate chart: {e}")

# ----------------------------
# Tab 2: CRUD Operations
# ----------------------------
with tab2:
    st.title(" Manage Records (CRUD)")

    st.markdown("Here you can **Add, Update, or Delete records** from the database.")

    # Add New Food Listing
    with st.expander(" Add New Food Listing"):
        food_name = st.text_input("Food Name")
        qty = st.number_input("Quantity", min_value=1)
        provider_id = st.number_input("Provider ID", min_value=1)
        expiry = st.date_input("Expiry Date")
        location = st.text_input("Location")
        food_type = st.selectbox("Food Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
        meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snacks"])

        if st.button("Add Listing"):
            conn.execute(
                "INSERT INTO Food_Listings (Food_Name, Quantity, Expiry_Date, Provider_ID, Location, Food_Type, Meal_Type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (food_name, qty, expiry, provider_id, location, food_type, meal_type)
            )
            conn.commit()
            st.success("Food listing added successfully!")

    # Update Food Listing
    with st.expander(" Update Food Listing"):
        food_id = st.number_input("Food ID to Update", min_value=1)
        new_qty = st.number_input("New Quantity", min_value=1)
        if st.button("Update Quantity"):
            conn.execute("UPDATE Food_Listings SET Quantity=? WHERE Food_ID=?", (new_qty, food_id))
            conn.commit()
            st.success(" Food listing updated!")

    # Delete Food Listing
    with st.expander(" Delete Food Listing"):
        delete_id = st.number_input("Food ID to Delete", min_value=1)
        if st.button("Delete Listing"):
            conn.execute("DELETE FROM Food_Listings WHERE Food_ID=?", (delete_id,))
            conn.commit()
            st.success(" Food listing deleted!")

