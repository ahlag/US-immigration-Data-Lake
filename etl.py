import pandas as pd
import os, re
import configparser
from datetime import timedelta, datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when, lower, isnull, year, month, dayofmonth, hour, weekofyear, dayofweek, date_format, to_date
from pyspark.sql.types import StructField, StructType, IntegerType, DoubleType, LongType, DateType, StringType

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = "/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/jvm/java-8-openjdk-amd64/bin"
os.environ["SPARK_HOME"] = "/opt/spark-2.4.3-bin-hadoop2.7"
os.environ["HADOOP_HOME"] = "/opt/spark-2.4.3-bin-hadoop2.7"

# The AWS key id and password are configured in a configuration file "dl.cfg"
config = configparser.ConfigParser()
config.read('dl.cfg')

# Reads and saves the AWS access key information and saves them in a environment variable
os.environ['AWS_ACCESS_KEY_ID']=config['KEYS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['KEYS']['AWS_SECRET_ACCESS_KEY']

def create_spark_session():
    """
    Description: Start spark session
    """
    spark = SparkSession.builder \
                .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
                .getOrCreate()

    return spark

def cast_type(df, cols):
    """
    Description: Convert the types of the columns according to the configuration supplied in the cols dictionary in the format {"column_name": type}
    
    Parameters:
            df   : Spark dataframe to be processed. Represents the entry point to programming Spark with the Dataset and DataFrame API.
            cols : Dictionary in the format of {"column_name": type} indicating what columns and types they should be converted to
    """
    
    for k,v in cols.items():
        if k in df.columns:
            df = df.withColumn(k, df[k].cast(v))
            
    return df

def convert_sas_date(df, cols):
    """
    Description: Convert dates in the SAS datatype to a date in a string format YYYY-MM-DD
    
    Parameters:
        df   : Spark dataframe to be processed. Represents the entry point to programming Spark with the Dataset and DataFrame API.
        cols : List of columns in the SAS date format to be convert
    """
    for c in [c for c in cols if c in df.columns]:
        df = df.withColumn(c, convert_sas_udf(df[c]))
    return df

@udf(StringType())
def convert_sas_udf(sas_date, date_format="%Y-%m-%d"):
    """
    Description: UDF for converting dates in the SAS datatype to a date in a string format YYYY-MM-DD
    
    Parameters:
        sas_date    : SAS date datatype.
        date_format : Date format e.g. %Y-%m-%d : List of columns in the SAS date format to be convert
    """ 
    if sas_date is None:
        return sas_date
    else:
        return (timedelta(days=sas_date) + datetime(1960, 1, 1)).strftime(date_format)

def process_demographics_data(spark, input_data, output_data):
    """
    Description: This function loads demographics dataset locally and processes it by extracting the and then again loaded back to S3
                    
    Parameters:
            spark       : Spark Session
            input_data  : location of demographics dataset
            output_data : S3 bucket were dimensional tables in parquet format will be stored
    """
    
    # If input_data is not set
    if len(input_data) == 0:
        input_data = "us-cities-demographics.csv"
        
    # Read US Cities Demo dataset file
    demographics=spark.read.csv(input_data, sep=';', header=True)
    
    # Convert numeric columns to the proper types: Integer and Double
    int_cols = ['Count', 'Male Population', 'Female Population', 'Total Population', 'Number of Veterans', 'Foreign-born']
    float_cols = ['Median Age', 'Average Household Size']
    demographics = cast_type(demographics, dict(zip(int_cols, len(int_cols)*[IntegerType()])))
    demographics = cast_type(demographics, dict(zip(float_cols, len(float_cols)*[DoubleType()])))
    
    first_agg = {
                    "Median Age": "first", 
                    "Male Population": "first", 
                    "Female Population": "first", 
                    "Total Population": "first", 
                    "Number of Veterans": "first", 
                    "Foreign-born": "first", 
                    "Average Household Size": "first"
                }

    # Aggregate by city
    agg_df = demographics.groupby(["City", "State", "State Code"]).agg(first_agg)

    # Pivot Table to transform values of the column Race to different columns
    pivot_df = demographics.groupBy(["City", "State", "State Code"]).pivot("Race").sum("Count")
    
    # Rename column names removing the spaces to avoid problems when saving to disk (we got errors when trying to save column names with spaces)
    demographics = agg_df.join(other=pivot_df, on=["City", "State", "State Code"], how="inner")\
                        .withColumnRenamed('State Code', 'StateCode')\
                        .withColumnRenamed('first(Total Population)', 'TotalPopulation')\
                        .withColumnRenamed('first(Female Population)', 'FemalePopulation')\
                        .withColumnRenamed('first(Male Population)', 'MalePopulation')\
                        .withColumnRenamed('first(Median Age)', 'MedianAge')\
                        .withColumnRenamed('first(Number of Veterans)', 'NumberVeterans')\
                        .withColumnRenamed('first(Foreign-born)', 'ForeignBorn')\
                        .withColumnRenamed('first(Average Household Size)', 'AverageHouseholdSize')\
                        .withColumnRenamed('Hispanic or Latino', 'HispanicOrLatino')\
                        .withColumnRenamed('Black or African-American', 'BlackOrAfricanAmerican')\
                        .withColumnRenamed('American Indian and Alaska Native', 'AmericanIndianAndAlaskaNative')

    numeric_cols = [
                        'TotalPopulation', 
                        'FemalePopulation', 
                        'MedianAge', 
                        'NumberVeterans', 
                        'ForeignBorn', 
                        'MalePopulation', 
                        'AverageHouseholdSize', 
                        'AmericanIndianAndAlaskaNative', 
                        'Asian', 
                        'BlackOrAfricanAmerican', 
                        'HispanicOrLatino', 
                        'White'
                    ]

    # Fill the null values with 0
    demographics = demographics.fillna(0, numeric_cols)

    # Write and transform demographics dataset into a parquet file
#     demographics.write.mode('overwrite').parquet(output_data + "/datalake/us_cities_demographics.parquet")
    

def process_immigration_data(spark, input_data, output_data):
    """
    Description: This function loads immigration dataset locally and processes it by extracting the and then again loaded back to S3
                    
    Parameters:
            spark       : Spark Session
            input_data  : location of demographics dataset
            output_data : S3 bucket were dimensional tables in parquet format will be stored
    """


    # Read i94 immigration dataset
    immigration=spark.read.parquet("sas_data")

    int_cols = [
                    'cicid', 
                    'i94yr', 
                    'i94mon', 
                    'i94cit', 
                    'i94res', 
                    'arrdate', 
                    'i94mode',
                    'i94bir', 
                    'i94visa', 
                    'count', 
                    'biryear', 
                    'dtadfile', 
                    'depdate'
                ]

    date_cols       = ['arrdate', 'depdate']
    high_null       = ["visapost", "occup", "entdepu", "insnum"]
    not_useful_cols = ["count", "entdepa", "entdepd", "matflag", "dtaddto", "biryear", "admnum"]

    # Convert columns read as string/double to integer
    immigration = cast_type(immigration, dict(zip(int_cols, len(int_cols)*[IntegerType()])))

     # Convert SAS date to a meaningful string date in the format of YYYY-MM-DD
    immigration = convert_sas_date(immigration, date_cols)
    
#     # Drop high null columns and not useful columns
#     immigration = immigration.drop(*high_null)
#     immigration = immigration.drop(*not_useful_cols)
    
    immigration.show(10)

# date_diff_udf = udf(date_diff)

# # Create a new columns to store the length of the visitor stay in the US
# immigration = immigration.withColumn('stay', date_diff_udf(immigration.arrdate, immigration.depdate))
# immigration = cast_type(immigration, {'stay': IntegerType()})

# immigration.write.mode("overwrite").parquet('s3a://ychang-output/datalake/immigration.parquet')

@udf(DateType())
def date_diff(date1, date2, date_format="%Y-%m-%d"):
    '''
    Description: This fucntion calculates the difference in days between two dates
    
    Parameters:
            date1: First date
            date2: Second date
            date_format: Date format. default: %Y-%m-%d
    '''
    
    if date1 is None or date2 is None:
        return None
    
    a = datetime.strptime(date1, date_format)
    b = datetime.strptime(date2, date_format)
    delta = b - a
    return delta.days



# # Start processing the I9I94_SAS_Labels_Description.SAS to create master i94 code dimensions:
# # Create i94mode list
# i94mode_data =[[1,'Air'],[2,'Sea'],[3,'Land'],[9,'Not reported']]

# # Convert to spark dataframe
# i94mode=spark.createDataFrame(i94mode_data)

# # Create i94mode parquet file
# i94mode.write.mode("overwrite").parquet('s3a://ychang-output/datalake/i94mode.parquet')

# countries = spark.read.format('csv').options(header='true', inferSchema='true').load("../../data2/GlobalLandTemperaturesByCity.csv")

# # Aggregates the dataset by Country and rename the name of new columns
# countries = countries.groupby(["Country"]).agg({"AverageTemperature": "avg", "Latitude": "first", "Longitude": "first"})\
# .withColumnRenamed('avg(AverageTemperature)', 'Temperature')\
# .withColumnRenamed('first(Latitude)', 'Latitude')\
# .withColumnRenamed('first(Longitude)', 'Longitude')

# def change_field_value_condition(df, change_list):
#     '''
#     Helper function used to rename column values based on condition.
    
#     Args:
#         df (:obj:`SparkDataFrame`): Spark dataframe to be processed.
#         change_list (:obj: `list`): List of tuples in the format (field, old value, new value)
#     '''
#     for field, old, new in change_list:
#         df = df.withColumn(field, when(df[field] == old, new).otherwise(df[field]))
#     return df

# # Rename specific country names to match the I94CIT_I94RES lookup table when joining them
# change_countries = [("Country", "Congo (Democratic Republic Of The)", "Congo"), ("Country", "Côte D'Ivoire", "Ivory Coast")]
# countries = change_field_value_condition(countries, change_countries)
# countries = countries.withColumn('CountryLower', lower(countries.Country))

# # Rename specific country names to match the demographics dataset when joining them
# change_res = [("I94CTRY", "BOSNIA-HERZEGOVINA", "BOSNIA AND HERZEGOVINA"), 
#                   ("I94CTRY", "INVALID: CANADA", "CANADA"),
#                   ("I94CTRY", "CHINA, PRC", "CHINA"),
#                   ("I94CTRY", "GUINEA-BISSAU", "GUINEA BISSAU"),
#                   ("I94CTRY", "INVALID: PUERTO RICO", "PUERTO RICO"),
#                   ("I94CTRY", "INVALID: UNITED STATES", "UNITED STATES")]

# # Loads the lookup table I94CIT_I94RES
# res = spark.read.format('csv').options(header='true', inferSchema='true').load("I94CIT_I94RES.csv")

# res = cast_type(res, {"Code": IntegerType()})
# res = change_field_value_condition(res, change_res)
# res = res.withColumn('resCountry_Lower', lower(res.I94CTRY))

# capitalize_udf = udf(lambda x: x if x is None else x.title())

# # Join the two datasets to create the country dimmension table
# res = res.join(countries, res.resCountry_Lower == countries.CountryLower, how="left")
# res = res.withColumn("Country", when(isnull(res["Country"]), capitalize_udf(res.I94CTRY)).otherwise(res["Country"]))   
# res = res.drop("I94CTRY", "CountryLower")
# res = res.drop("resCountry_Lower")

# # Create i94mode parquet file
# res.write.mode("overwrite").parquet('s3a://ychang-output/datalake/country.parquet')

# # Read i94 immigration dataset to create Date Frame
# i94_spark=spark.read.parquet("sas_data")

# i94_spark=i94_spark.select(col("i94res").cast(IntegerType()),col("i94port"),
#                            col("arrdate").cast(IntegerType()), \
#                            col("i94mode").cast(IntegerType()),col("depdate").cast(IntegerType()),
#                            col("i94bir").cast(IntegerType()),col("i94visa").cast(IntegerType()), 
#                            col("count").cast(IntegerType()), \
#                               "gender",col("admnum").cast(LongType()))

# # We will drop duplicate rows and save it as final dataset for i94
# i94_spark=i94_spark.dropDuplicates()

# import datetime as dt
# # Convert SAS arrival date to datetime format
# get_date = udf(lambda x: (dt.datetime(1960, 1, 1).date() + dt.timedelta(x)).isoformat() if x else None)
# i94non_immigrant_port_entry = i94_spark.withColumn("arrival_date", get_date(i94_spark.arrdate))

# from pyspark.sql import functions as F
# i94date= i94non_immigrant_port_entry.withColumn('Darrival_date',F.to_date(i94non_immigrant_port_entry.arrival_date))
# i94date = i94date.withColumn('arrival_month',month(i94date.Darrival_date))
# i94date = i94date.withColumn('arrival_year',year(i94date.Darrival_date))
# i94date = i94date.withColumn('arrival_day',dayofmonth(i94date.Darrival_date))
# i94date = i94date.withColumn('day_of_week',dayofweek(i94date.Darrival_date))
# i94date = i94date.withColumn('arrival_weekofyear',weekofyear(i94date.Darrival_date))

# i94date=i94date.select(col('arrdate').alias('arrival_sasdate'),col('Darrival_date').alias('arrival_iso_date'), 'arrival_month','day_of_week','arrival_year','arrival_day','arrival_weekofyear').dropDuplicates()

# # Create temporary sql table
# i94date.createOrReplaceTempView("i94date_table")

# # Add seasons to i94 date dimension table
# i94date_season=spark.sql('''
#                             SELECT
#                                 arrival_sasdate,
#                                 arrival_iso_date,
#                                 arrival_month,
#                                 day_of_week,
#                                 arrival_year,
#                                 arrival_day,
#                                 arrival_weekofyear,
#                                 CASE WHEN arrival_month IN (12, 1, 2) THEN 'winter' 
#                                      WHEN arrival_month IN (3, 4, 5) THEN 'spring' 
#                                      WHEN arrival_month IN (6, 7, 8) THEN 'summer' 
#                                      ELSE 'autumn' END AS date_season
#                             FROM i94date_table
#                          ''')

# # Perform quality checks here
# if i94date_season.count() > 0:
#     print('Passed reading data file.')
# else:
#     raise ValueError('Seems to be nothing in file!')
    
# if i94_spark.count() > 0:
#     print('Passed reading data file.')
# else:
#     raise ValueError('Seems to be nothing in file!')
    
# if i94_spark.count() == i94non_immigrant_port_entry.count():
#     print('Transformation went perfect.')
# else:
#     raise ValueError('Inconsistant data between both dataframes!')

# # Save i94date dimension to parquet file partitioned by year and month:
# i94date_season.write.mode("overwrite").partitionBy("arrival_year", "arrival_month").parquet('s3a://ychang-output/datalake/i94date.parquet')


def main():
    """
    Extract immigration, demographics, country, mode of transport and arrival data from S3, Transform it into dimensional tables format, and Load it back to S3 in Parquet format
    """
    spark = create_spark_session()
    spark = create_spark_session()
    
    ## Input and output paths
    output_data = "s3a://ychang-output/"
    input_data  = ""
    
#     print(date_diff('2020-01-01', '2020-02-01'))
    process_demographics_data(spark, input_data, output_data)
    process_immigration_data(spark, input_data, output_data)

if __name__ == "__main__":
    main()