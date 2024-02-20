#######################################################################################
# This file has the problem statement and code solution for each problem. They are
# stored in Python multi line strings. The function DoProblem prints the problem
# statement, prints the code and then uses the exec() function to execute the code.
# This worked fine for problems 1-8. For problem 9, the calling of functions defined 
# in the exec() did not work. So, it is slightly different.
#######################################################################################
def DoProblem(ProblemDescription, ProblemCode):
    print(ProblemDescription)
    print("Code:")
    print(ProblemCode)
    print("Execution:")
    print("")
    exec(ProblemCode)

#######################################################################################
#
# Example running this file:
# C:\Users\Micha\github>py ".\GWU\SEAS 6414\SEAS6414_HW4_Wacey.py" >".\GWU\SEAS 6414\SEAS6414_HW4_Wacey.txt"
#
#######################################################################################
P1Text = """
#######################################################################################
# Problem 1
#######################################################################################

Problem:

Dataset: homework4 file1.csv

Data Description: The dataset contains records of merchant transactions, each
with a unique merchant identifier, time of transaction, and amount in cents.

Objective: Analyze merchant transaction data to understand business growth and
health. Preprocess the dataset for future merchant transactions and generate specific
features for each merchant.

Task: Generate the following features for each unique merchant:
- trans amount avg: Average transaction amount for each merchant.
- trans amount volume: Total transaction amount for each merchant.
- trans frequency: Total count of transactions for each merchant.
- trans recency: Recency of the last transaction (in days from 1/1/2035).
- avg time btwn trans: Average time between transactions (in hours).
- avg trans growth rate: Average growth rate in transaction amounts.

Data Dimension: The dataset is N by 3, where N is the number of records.

Final Deliverables:
- Shape of the new dataset.
- The top five rows of the new dataset using new dataset.head().
- Descriptive statistics of the new dataset.
"""
P1Code = """
import pandas as pd
import numpy as np
import datetime as dt
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
HW4F1 = pd.read_csv('./gwu/SEAS 6414/homework4_file1.csv')
# Make the time column a Pandas time rather than a string
HW4F1['time'] = [pd.Timestamp(ts) for ts in HW4F1.time]
HW4F1.sort_values(by=['merchant', 'time'], inplace=True)
HW4F1New = HW4F1.groupby("merchant").agg(
    min_amount=("amount_usd_in_cents", "min")
    , max_amount=("amount_usd_in_cents", "max")
    , trans_amount_avg=("amount_usd_in_cents", "mean")
    , trans_amount_volume=("amount_usd_in_cents", "sum")
    , trans_frequency=("amount_usd_in_cents", "count")
    , most_recent_date=("time", "max")
    , avg_time_btwn_trans=("time", lambda group: group.sort_values().diff().mean().seconds/(60*60))
    , avg_trans_growth_rate=("amount_usd_in_cents", lambda group: group.sort_values().pct_change().mean())
)
# I tried to do this as a Lambda in the agg, but it would not recognize the dt library
# So, in the agg, I find the max and here I calculate the delta
HW4F1New['trans_recency'] = (HW4F1New['most_recent_date'] - dt.datetime(2035, 1, 1)).dt.days
# getting rid of the no longer needed maximum value
HW4F1Final = HW4F1New.drop(columns=['most_recent_date'])

print(f"The shape of the original data frame is: {HW4F1.shape}")
print(f"The shape of the new data frame is: {HW4F1Final.shape}")
print("")
print("The top five rows are:")
print(HW4F1Final.head(5))
print("")
print("Descriptive statistics:")
print(HW4F1Final.describe())
"""
DoProblem(P1Text, P1Code)

P2Text = """
#######################################################################################
# Problem 2
#######################################################################################

Problem:

You are provided with two datasets: sales data.csv and product info.csv.
- sales data.csv contains transaction records with columns: 'TransactionID',
  'ProductID', 'Date', 'Quantity', and 'Price'.
- product info.csv contains product details with columns: 'ProductID', 'ProductName', 'Category'.
 
Your task involves multiple steps of data manipulation using Pandas and NumPy to
extract insights from these datasets.

Tasks:
1. Data Loading and Merging:
- Load both datasets using Pandas.
- Merge them into a single DataFrame on 'ProductID'.
2. Data Cleaning:
- Check for and handle any missing values in the merged dataset.
- Convert the 'Date' column to a DateTime object.
3. Data Analysis using Slicing and Indexing:
- Create a new column 'TotalSale', calculated as 'Quantity' * 'Price'.
- Using slicing, create a subset DataFrame containing only transactions from
  the last quarter of the year (October, November, December).
- Using Boolean indexing, find all transactions for a specific 'Category' (e.g.,
  'Electronics').
- Extract all transactions where the 'TotalSale' is above the 75th percentile
  of the 'TotalSale' column using NumPy functions.
4. Advanced Indexing:
- Using loc and iloc, perform the following:
  - Select all rows for 'ProductID' 101 and columns 'ProductName' and
    'TotalSale'.
  - Select every 10th row from the merged dataset and only the columns
    'Date' and 'Category'.
5. Grouping and Aggregation:
- Group the data by 'Category' and calculate the total and average 'TotalSale'
  for each category.
6. Time-Series Analysis:
- Resample the data on a monthly basis and calculate the total 'Quantity'
  sold per month.
Final Deliverables:
- Provide the code for each step.
- Include comments explaining your approach.
- Display the first 5 rows of the DataFrame after each major step.
"""
P2Code = """
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
print("Task 1 - Data Loading and Merging")
SalesData = pd.read_csv("./gwu/SEAS 6414/sales_data.csv")
Product = pd.read_csv("./gwu/SEAS 6414/product_info.csv")
# I checked the row counts and there are no product ids in the Sales Data
# that have product keys that are not in Product data. So, an inner join
# will work for this data.
SalesProductData = pd.merge(SalesData, Product, on="ProductID", how="inner")
print("The merged SalesProductData data frame.")
print(SalesProductData)
print("")
print("Task 2 - Data Cleaning")
# Counting the NAs across the dimensions shows that there is no missing data. I also ran
# dropna and saw that the result had the same shape as the input. So, I am confident that
# there is no missing data. Which worries me. Why would you ask us to address missing data
# if there was none.
print(f"The merged data frame has {SalesProductData.isnull().sum().sum()} missing values.")
print("")
print("The SalesProductData types before converting to a datetime:")
print(SalesProductData.dtypes)
SalesProductData['Date'] = [pd.to_datetime(aDate) for aDate in SalesProductData.Date]
print("")
print("The SalesProductData types after converting to a datetime:")
print(SalesProductData.dtypes)
print("")
print("Task 3 - Data Analysis using Slicing and Indexing")
SalesProductData['TotalSale'] = SalesProductData['Quantity'] * SalesProductData['Price']
SalesProductData4Q = SalesProductData.set_index('Date').sort_values(by=['Date'])['2023-10-01' : '2023-12-31']
print("")
print("Sales records for the fourth quarter:")
print(SalesProductData4Q)
mask = SalesProductData['Category'] == 'Electronics'
SalesProductElectronics = SalesProductData[mask]
print("")
print("Sales records for Electronics:")
print(SalesProductElectronics)
# First create and index of all records that have a TotalSale value greater than the 75th percentile
SalesProductOver75Index = np.where(SalesProductData['TotalSale']>np.percentile(SalesProductData['TotalSale'],75))
# Next select those values.
SalesProductOver75 = SalesProductData.loc[SalesProductOver75Index]
print("")
print("Sales records for total price over the 75th percentile:")
print(SalesProductOver75)
print("")
print("Task 4 - Advanced Indexing")
SalesProductDataPID = SalesProductData.set_index('ProductID')
SalesProductData101 = SalesProductDataPID.loc[101,['ProductName','TotalSale']]
print("")
print("Sales records for product 101 with Product Name and Total Sale:")
print(SalesProductData101)
SalesProductDataEvery10th = SalesProductData.iloc[::10,[2,6]]
print("")
print("Sales records for every 10th row with Date and Category:")
print(SalesProductDataEvery10th)
print("")
print("Task 5 - Grouping and Aggregation")
SalesProductDataCatGrp = SalesProductData.groupby("Category").agg(
    total_sale=("TotalSale", "sum")
    , average_sale=("TotalSale", "mean")
)
print("")
print("Sales records grouped by category with total and average sales by category:")
print(SalesProductDataCatGrp)
print("")
print("Task 6 - Time-Series Analysis")
# Get down to just the columns needed. I tried to combine this with the indexing but
# none of my incantations would work.
SalesProductDataSmall = SalesProductData.loc[:,['Date','Quantity']]
# To resample, we need the date to be the index
SalesProductDataDate = SalesProductDataSmall.set_index('Date')
# Now we can resample down to Month End and calculate the average
SalesProductDataMonth = SalesProductDataDate.resample('ME').mean()
print(SalesProductDataMonth)
"""
DoProblem(P2Text, P2Code)

P3Text = """
#######################################################################################
# Problem 3
#######################################################################################

Problem:

Zillow's marketplace offers a data-driven home valuation platform utilized by a diverse
range of users including home buyers, sellers, renters, homeowners, real estate
agents, mortgage providers, property managers, and landlords. The machine learning
and data science team at Zillow employs various tools for predicting home valuations,
such as Zestimate (Zillow Estimate), Zestimate Forecast, Zillow Home Value Index,
Rent Zestimate, Zillow Rent Index, and the Pricing Tool.

Assignment Overview:
You are provided with a dataset named zillow feature sample.csv, containing
various features relevant to Zillow's marketplace. Accompanying the dataset is a
data dictionary titled zillow data dictionary.xlsx, which details the description
of each column.

Tasks:
1. Develop a Missing Data Strategy:
- Assess the zillow feature sample.csv dataset and devise a comprehensive strategy to handle missing data.
2. Quantitative Analysis of Missing Data:
- Calculate and report the percentage of missing data in each feature of the
  dataset.
- Analyze and infer the potential mechanism of missing data (e.g., Missing
  Completely at Random, Missing at Random, Missing Not at Random).
3. Imputation Strategy:
- Propose and justify an imputation strategy for the missing values in the
  dataset. Your rationale should be data-driven and well-explained.
4. Open-Ended Exploration:
- This question is open-ended, allowing you to explore other relevant aspects
  of the dataset. Conduct additional analyses or apply data processing techniques as appropriate.

Submission Guidelines:
- Document your analysis and findings in a clear and structured format.
- Ensure that your submission is thorough and well-reasoned.
"""
P3Code = """
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
print("")
print("Task 1 - Develop a Missing Data Strategy")
ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample.csv")
print("The data provided:")
print(ZillowFeatureSample)
print("Descriptive statistics for each feature:")
print(ZillowFeatureSample.describe())
print("")
print("This data is used to predict house prices. Since it does not have actual prices, we cannot")
print("use it for training or testing our models. Therefore, we cannot test the impact of any")
print("missing data strategy with just this data at hand. However, we can look at the data and")
print("determine if any missing data approach would be useful. Below is my strategy based on a")
print("review of the data values and data dictionary.")
print("")
print("From the data dictionary:")
print("  - The data dictionary has eight tabs.")
print("    - The first one is for the data file.")
print("    - The remaining seven are code tables for features that are coded.")
print("  - Eight of feature descriptions had the phrase 'if any' in them, or should.")
print("    - Some features probably should include 'if any' in the description")
print("    - For example, 'airconditioningtypeid' is described as 'Type of cooling system")
print("      present in the home (if any)'")
print("    - For example, 'assessmentyear' is described as 'The year of the property tax assessment'.")
print("      Since a house may never have been assessed, this is similar to 'if any'.")
print("    - In both these cases, any unavailable information could be treated as a No or whatever")
print("      is appropriate.")
print("  - Seventeen of the features have the characters ID at the end of the name.")
print("    - Of these seven have tables on other tabs and ten do not.")
print("    - Assignment of an ID means that a process was followed to code the data.")
print("    - Given this process, I would be reluctant to replace the missing data with a value.")
print("  - Some data is dependant on other data.")
print("    - If 'regionidzip' is available, we could use that to fill in City, State, etc.")
print("    - For each feature, we can look into any dependencies that could help derive the values.")
print("    - We will need to be careful with this. We will have to determine the dependencies, then")
print("    - derive the data, then remove the dependant values so that only one of them remains. This")
print("    - ensures that we are only left with independent variables (features).")
print("  - There appear to be a lot of missing values. We will need to carefully consider these")
print("    features. We may need to drop those that are missing too many values.")
print("")
print("Task 2 - Quantitative Analysis of Missing Data")
missing_value_analysis = pd.DataFrame({'count_missing': ZillowFeatureSample.isna().sum()
                                 , 'percent_missing': ZillowFeatureSample.isnull().sum() * 100 / len(ZillowFeatureSample)})
print("")
print("Count and percent missing for each feature, sorted low to high by percent:")
print(missing_value_analysis.sort_values(by=['percent_missing']))
print("")
print("Searching the web, it looks like a lot of people consider between 10 and 20% missing")
print("a cutoff point -> more than 20% missing, do not use the feature. But this is always followed")
print("with - there is no hard cutoff point. Since we have 9.25% missing and then 34.00% missing")
print("my working assumption for now is that this will be the cutoff point. But I will continue")
print("analyzing the data to see if some of the features with 34.00% or greater missing are useful.")
print("")
print("Trying to infer the mechanism of missing data will be tricky for me. There are several")
print("reasons for this:")
print("  - I do not know how any of the data was collected.")
print("  - This is not an area that I have any expertise in.")
print("")
print("With those caveats in mind, here is my estimation for each feature.")
print("  - For the 23 features that have a missing percent under 4%, I deem them as not")
print("    really missing. If a value is needed for them, it can easily be imputed.")
print("  - For the 26 features with a missing percent over 70%, I deem them as to much")
print("    missing. I would be hard pressed to impute these values. There may be special")
print("    cases as the analysis progresses.")
print("  - The remaining nine features need to be addressed.")
print("  - Based on the information provided, I cannot say if they are MCAR, MAR, or MNAR.")
print("    I would need details about how the information was collected and about housing")
print("    data.")
print("")
print("Based on the above, I created the table below for values that could be imputed:")
print("    finishedsquarefeet12   Impute from Calculated square feet")
print("    lotsizesquarefeet      Impute from address")
print("    unitcnt                Do not impute - I expect number of units to be unique")
print("    propertyzoningdesc     Impute from address")
print("    buildingqualitytypeid  Do not impute - an ID")
print("    heatingorsystemtypeid  Do not impute - an ID")
print("    regionidneighborhood   Impute from address")
print("    garagecarcnt           Impute from address")
print("    garagetotalsqft        Impute from address")
print("")
print("Task 3 - Imputation strategy")
print("")
print("Let me start by saying that my gut reaction is that using imputation is a really bad")
print("idea. We have data that we are trying to use to predict something and before we do")
print("we are predicting values that are missing from the data. If we use existing values to")
print("impute the values, we are not adding anything to the data we have. I am actually concerned")
print("that people are making decisions based on this. It seems like an incredibly bad idea.")
print("")
print("If I had to impute values for this data set, I would use averages in most cases. I would")
print("try to find a set of the data from the same general area and similar houses. This is based")
print("on the idea that all 3,000 square foot houses built in the same area in the same time period")
print("will essentially be the same. So, if we can get enough records, we can do that. This data set")
print("may be too small to get enough records. But given that Zillow seems to have data for every")
print("house in the US, it should be possible to get more data.")
print("")
print("Based on this, I would be willing to impute values for the 23 features that are missing under")
print("4% of the values and the four features identified above.")
print("")
print("Task 4 - Open-Ended Exploration")
print("")
print("Does year built correlate with size?")
ZillowFeatureSampleSmall = ZillowFeatureSample.loc[:,['yearbuilt','calculatedfinishedsquarefeet']]
print(ZillowFeatureSampleSmall.corr(numeric_only=True))
print("It appears to have a low correlation.")
print("")
print("Does latitude correlate air conditioning?")
ZillowFeatureSampleSmall = ZillowFeatureSample.loc[:,['latitude','airconditioningtypeid']]
print(ZillowFeatureSampleSmall.corr(numeric_only=True))
print("This seems to be saying that there is an inverse relation. That makes sense. The higher")
print("the latitude, the less need there is for air conditioning. Note that the values for")
print("air conditioning are not really good for this correlation. To really do it right, I would")
print("need to convert the values. But as a first cut, it makes sense.")
print("I could probably do similar things for pools at lower latitudes and fire places at higher")
print("latitudes. I am not sure it would be worthwhile given the amount of missing data.")
"""
DoProblem(P3Text, P3Code)