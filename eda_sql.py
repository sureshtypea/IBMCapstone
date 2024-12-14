import sqlite3
import pandas as pd

# Load the dataset
file_path = 'Spacex.csv'  # Replace with the correct file path if needed
spacex_data = pd.read_csv(file_path)

# Create a SQLite database and connection
connection = sqlite3.connect("spacex_lab.db")
cursor = connection.cursor()

# Load the data into the database
spacex_data.to_sql("SPACEXTBL", connection, if_exists='replace', index=False, method="multi")

# Remove blank rows by recreating the table with valid entries only
cursor.execute("DROP TABLE IF EXISTS SPACEXTABLE;")
cursor.execute("""
    CREATE TABLE SPACEXTABLE AS 
    SELECT * 
    FROM SPACEXTBL 
    WHERE Date IS NOT NULL;
""")
connection.commit()

# Query definitions for all tasks
queries = {
    "Task 1": "SELECT DISTINCT Launch_Site FROM SPACEXTABLE;",
    "Task 2": "SELECT * FROM SPACEXTABLE WHERE Launch_Site LIKE 'CCA%' LIMIT 5;",
    "Task 3": "SELECT SUM(PAYLOAD_MASS__KG_) AS Total_Payload_Mass FROM SPACEXTABLE WHERE Customer = 'NASA (CRS)';",
    "Task 4": "SELECT AVG(PAYLOAD_MASS__KG_) AS Average_Payload_Mass FROM SPACEXTABLE WHERE Booster_Version = 'F9 v1.1';",
    "Task 5": "SELECT MIN(Date) AS First_Successful_Landing_Date FROM SPACEXTABLE WHERE Landing_Outcome = 'Success (ground pad)';",
    "Task 6": """
        SELECT Booster_Version FROM SPACEXTABLE 
        WHERE Landing_Outcome = 'Success (drone ship)' 
        AND PAYLOAD_MASS__KG_ > 4000 
        AND PAYLOAD_MASS__KG_ < 6000;
    """,
    "Task 7": """
        SELECT Mission_Outcome, COUNT(*) AS Total_Count 
        FROM SPACEXTABLE 
        GROUP BY Mission_Outcome;
    """,
    "Task 8": """
        SELECT Booster_Version 
        FROM SPACEXTABLE 
        WHERE PAYLOAD_MASS__KG_ = (
            SELECT MAX(PAYLOAD_MASS__KG_) 
            FROM SPACEXTABLE
        );
    """,
    "Task 9": """
        SELECT 
            CASE 
                WHEN SUBSTR(Date, 6, 2) = '01' THEN 'January'
                WHEN SUBSTR(Date, 6, 2) = '02' THEN 'February'
                WHEN SUBSTR(Date, 6, 2) = '03' THEN 'March'
                WHEN SUBSTR(Date, 6, 2) = '04' THEN 'April'
                WHEN SUBSTR(Date, 6, 2) = '05' THEN 'May'
                WHEN SUBSTR(Date, 6, 2) = '06' THEN 'June'
                WHEN SUBSTR(Date, 6, 2) = '07' THEN 'July'
                WHEN SUBSTR(Date, 6, 2) = '08' THEN 'August'
                WHEN SUBSTR(Date, 6, 2) = '09' THEN 'September'
                WHEN SUBSTR(Date, 6, 2) = '10' THEN 'October'
                WHEN SUBSTR(Date, 6, 2) = '11' THEN 'November'
                WHEN SUBSTR(Date, 6, 2) = '12' THEN 'December'
            END AS Month,
            Landing_Outcome,
            Booster_Version,
            Launch_Site
        FROM SPACEXTABLE
        WHERE Landing_Outcome = 'Failure (drone ship)'
        AND SUBSTR(Date, 1, 4) = '2015';
    """,
    "Task 10": """
        SELECT Landing_Outcome, COUNT(*) AS Outcome_Count
        FROM SPACEXTABLE
        WHERE Date BETWEEN '2010-06-04' AND '2017-03-20'
        GROUP BY Landing_Outcome
        ORDER BY Outcome_Count DESC;
    """
}

# Execute and collect results for each task
results = {}
for task, query in queries.items():
    results[task] = pd.read_sql_query(query, connection)

# Display all results
for task, result in results.items():
    print(f"\n{task} Results:")
    print(result)

# Close the connection
connection.close()
