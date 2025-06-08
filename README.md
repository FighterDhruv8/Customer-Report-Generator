# Overview
This project reads data from a postgresql database and generates a Natural Language report based on it.

# Setup

- Ensure your system has python installed.
- Get a Google Gemini API key, and store it in your environment variables with the name `Gemini_API_Key`.
- Download or clone this repo to your system.
- Make a postgresql database, and enter your db name, your username, and password into the [config.json](config.json) file.
- Transfer the .csv file into the postgresql db via the following steps:

CREATE TABLE DDL:

    CREATE TABLE customer_data (
    customer_id TEXT,
    product TEXT,
    quantity INTEGER CONSTRAINT pos_quantity CHECK (quantity>=0),
    unit_price_usd MONEY,
    total_price_usd MONEY,
    purchase_date DATE,
    customer_name TEXT,
    industry TEXT,
    annual_revenue_usd MONEY,
    number_of_employees INTEGER,
    customer_priority TEXT,
    customer_rating TEXT,
    account_type TEXT,
    "location" TEXT,
    current_products TEXT,
    product_usage_percent SMALLINT,
    cross_sell_synergy TEXT,
    last_activity_date DATE,
    opportunity_stage TEXT
    );

INSERT DATA CMD COMMAND:

    psql -U postgres -d mydb -h localhost -c "\copy customer_data FROM 'C:/Users/user/Downloads/customer_data.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '\"')"

Once these steps are completed, run `pip install -r requirements.txt` to install all the dependencies and then run the [main.py](main.py) file. The output should say app running.

Access the API endpoint via a GET request of format `/recommendation?customer_id=<id>`, or paste the URL `http://localhost:8000/recommendation?customer_id=<id>` into a web browser (don't forget to swap in the customer ID!).

The output should be the Natural Language Report with the timestamp and the customer ID, and it should also have generated a text file with the report in your local system.
