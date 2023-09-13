# %%
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import calendar
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
# Initialize SparkSession
spark = SparkSession.builder.appName("CSVToPostgreSQL").getOrCreate()

# Read the CSV data into a DataFrame
df1 = spark.read.csv("/home/rojesh/Documents/spark-training/final/Fuel_Station_Information.csv", header=True, inferSchema=True)
df2 = spark.read.csv("/home/rojesh/Documents/spark-training/final/Hourly_Gasoline_Prices.csv", header=True, inferSchema=True)

joined_df = df1.join(df2, "Id", "inner")

# Data Cleaning: Remove rows with null values in any column
cleaned_df = joined_df.dropna()

# Data Cleaning: Remove rows where the "Type" column has the value "autostradle"
cleaned_df = cleaned_df.filter(F.col("Type") != "autostradle")

parquet_path = "../trying/parquet"
cleaned_df.write.parquet(parquet_path, mode="overwrite")




# %%

# Define the JDBC connection properties
jdbc_url = "jdbc:postgresql://localhost:5432/postgres"
properties = {
    "user": "postgres",
    "password": "Potanginamo123",
    "driver": "org.postgresql.Driver"
}

table_name = "newtable"

cleaned_df.write.jdbc(url=jdbc_url, table=table_name, mode="append", properties=properties)

# %%


# Define the JDBC connection properties
jdbc_url = "jdbc:postgresql://localhost:5432/postgres"
properties = {
    "user": "postgres",
    "password": "Potanginamo123",
    "driver": "org.postgresql.Driver"
}

table_name = "newtable"

#filteringtable
filter_condition = "1=1 LIMIT 5000"

df3 = spark.read \
    .format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", f"(SELECT * FROM {table_name} WHERE {filter_condition}) AS filtered_table") \
    .options(**properties) \
    .load()


row_count = df3.count()
print(f"Number of rows read: {row_count}") 


df3.show()

# %% [markdown]
# ### Calculate the average seasonal prices for a dataset containing date and price information, while also assigning each date a season label based on the month

# %%
df = df3.withColumn('month', F.month('date'))

#defining a UDF to map months to seasons
def get_season(month):
    seasons = {
        1: 'Winter',
        2: 'Winter',
        3: 'Spring',
        4: 'Spring',
        5: 'Spring',
        6: 'Summer',
        7: 'Summer',
        8: 'Summer',
        9: 'Autumn',
        10: 'Autumn',
        11: 'Autumn',
        12: 'Winter'
    }
    return seasons.get(month, 'Unknown')

get_season_udf = F.udf(get_season, StringType())

df = df.withColumn('Season', get_season_udf(F.col('month')))

window_spec = Window.partitionBy('Season')

seasonal_avg_prices = df.withColumn('Average Seasonal Price',
    F.avg(F.col('Price')).over(window_spec)
)

seasonal_avg_prices = seasonal_avg_prices.dropDuplicates(['Season'])

seasonal_avg_prices.select("Season" , "Average Seasonal Price").show()



# %%
df = df3.withColumn('month', F.month('date')) \
        .withColumn('Season', F.when(F.col('month').between(3, 5), 'Spring')
                               .when(F.col('month').between(6, 8), 'Summer')
                               .when(F.col('month').between(9, 11), 'Autumn')
                               .otherwise('Winter'))

window_spec = Window.partitionBy('Season')

seasonal_avg_prices = df.withColumn('Average Seasonal Price',
                                    F.avg('Price').over(window_spec))

seasonal_avg_prices = seasonal_avg_prices.dropDuplicates(['Season'])

seasonal_avg_prices.select("Season", "Average Seasonal Price").show()


# %%
df3.printSchema()

# %% [markdown]
# ###  Find the distance between the locations with the minimum and maximum prices in a dataset

# %%
min_price_row = df3.orderBy(F.col("Price")).first()
max_price_row = df3.orderBy(F.col("Price").desc()).first()

print(min_price_row)
print(max_price_row)

#checking if min_price_row and max_price_row are not None
if min_price_row is not None and max_price_row is not None:
    min_latitude = float(min_price_row["Latitude"])
    min_longitude = float(min_price_row["Longitudine"])
    max_latitude = float(max_price_row["Latitude"])
    max_longitude = float(max_price_row["Longitudine"])

    min_latitude_rad = F.radians(F.lit(min_latitude)).cast("double")
    min_longitude_rad = F.radians(F.lit(min_longitude)).cast("double")
    max_latitude_rad = F.radians(F.lit(max_latitude)).cast("double")
    max_longitude_rad = F.radians(F.lit(max_longitude)).cast("double")

    distance_km = F.acos(
    F.sin(min_latitude_rad) * F.sin(max_latitude_rad) +
    F.cos(min_latitude_rad) * F.cos(max_latitude_rad) *
    F.cos(max_longitude_rad - min_longitude_rad)
).cast("double") * 6371.0

    df_with_distance = df3.withColumn("Distance_km", distance_km)
    df_with_distance.select("Distance_km").distinct().show()
else:
    print("No data found to calculate minimum and maximum prices.")

# %% [markdown]
# ### Calculate the average prices for each day of the month and each month, then presents the results in a pivot table

# %%
df = df3.withColumn("day_of_month", F.dayofmonth("Date"))
df = df.withColumn("month", F.month("Date"))


day_pivot_table = df.groupBy("day_of_month").pivot("month").agg(F.avg("Price"))

# Define a list of month names
month_names = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


for i in range(1, 13):
    month_name = month_names[i - 1]
    day_pivot_table = day_pivot_table.withColumnRenamed(str(i), month_name)


day_pivot_table.show()

# %%

# Define the window specification for rolling hourly averages
window_spec = Window.partitionBy('date').orderBy('date').rowsBetween(-4, 0)

# Calculate rolling hourly averages
df = df.withColumn('RollingHourlyAvgPrice', F.avg('Price').over(window_spec))

df.show()

# %%
csv_file_path = "../trying/worldcities.csv"
df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Filter rows where the 'country' column is 'Italy'
italy_df = df.filter(F.col('country') == 'Italy')

# Show the filtered DataFrame
italy_df.show(truncate=False)

# %%
italy_df = italy_df.withColumnRenamed('city', 'City')

italy_df.show(truncate=False)


# %%
df3 = df3.withColumn('City', F.lower(F.col('City')))
italy_df = italy_df.withColumn('City', F.lower(F.col('City')))

joined_df2 = df3.join(italy_df, ['City'], 'inner')


joined_df2 = joined_df2.drop('capital')

                    
joined_df2.distinct().show()



# %%
jdbc_url = "jdbc:postgresql://localhost:5432/postgres"
properties = {
    "user": "postgres",
    "password": "Potanginamo123",
    "driver": "org.postgresql.Driver"
}

table_name = "table3"

columns_to_insert = ["City", "Fuel_station_manager", "Petrol_company", "Type", "Station_name", "Latitude", "Longitudine", "isSelf", "Price", "Date", "city_ascii", "lat", "lng", "country", "iso2", "iso3", "admin_name", "population"]

joined_df2.select(*columns_to_insert).write.jdbc(url=jdbc_url, table=table_name, mode="append", properties=properties)


# %%



