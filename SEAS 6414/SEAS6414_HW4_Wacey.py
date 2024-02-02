
"""





3. Zillow's marketplace offers a data-driven home valuation platform utilized by a diverse range of users including home buyers, sellers, renters, homeowners, real estate
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
• Assess the zillow feature sample.csv dataset and devise a comprehensive strategy to handle missing data.
2. Quantitative Analysis of Missing Data:
• Calculate and report the percentage of missing data in each feature of the
dataset.
• Analyze and infer the potential mechanism of missing data (e.g., Missing
Completely at Random, Missing at Random, Missing Not at Random).
3. Imputation Strategy:
• Propose and justify an imputation strategy for the missing values in the
dataset. Your rationale should be data-driven and well-explained.
4. Open-Ended Exploration:
• This question is open-ended, allowing you to explore other relevant aspects
of the dataset. Conduct additional analyses or apply data processing techniques as appropriate.
Submission Guidelines:
• Document your analysis and findings in a clear and structured format.
• Ensure that your submission is thorough and well-reasoned.

"""




#######################################################################################
# This file has the problem statement and code solution for each problem. They are
# stored in Python multi line strings. The function DoProblem prints the problem
# statement, prints the code and then uses the exec() function to execute the code.
# This worked fine for problems 1-8. For problem 9, the calling of functions defined 
# in the exec() did not work. So, it is slighly different.
#######################################################################################
def DoProblem(ProblemDescription, ProblemCode):
    print(ProblemDescription)
    print("Code:")
    print(ProblemCode)
    print("Execution:")
    print("")
    exec(ProblemCode)

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
• trans amount avg: Average transaction amount for each merchant.
• trans amount volume: Total transaction amount for each merchant.
• trans frequency: Total count of transactions for each merchant.
• trans recency: Recency of the last transaction (in days from 1/1/2035).
• avg time btwn trans: Average time between transactions (in hours).
• avg trans growth rate: Average growth rate in transaction amounts.

Data Dimension: The dataset is N by 3, where N is the number of records.

Final Deliverables:
• Shape of the new dataset.
• The top five rows of the new dataset using new dataset.head().
• Descriptive statistics of the new dataset.
"""
P1Code = """
import pandas as pd
import numpy as np
import datetime as dt
pd.options.display.float_format = '{:,.2f}'.format
HW4F1 = pd.read_csv(r".\gwu\SEAS 6414\homework4_file1.csv")
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
# I tried to do this as a Lamda in the agg, but it would not recognize the dt library
# So, in the agg, I find the max and here I calculate the delta
HW4F1New['trans_recency'] = (HW4F1New['most_recent_date'] - dt.datetime(2035, 1, 1)).dt.days
# getting rid of the no longer needed maximum value
HW4F1Final = HW4F1New.drop(columns=['most_recent_date'])

print(f"The shape of the original dataframe is: {HW4F1.shape}")
print(f"The shape of the new dataframe is: {HW4F1Final.shape}")
print("The top five rows are:")
print(HW4F1Final.head(5))
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
SalesData = pd.read_csv(r".\gwu\SEAS 6414\sales_data.csv")
Product = pd.read_csv(r".\gwu\SEAS 6414\product_info.csv")
# I checked the row counts and there are no product ids in the Sales Data
# that have product keys that are not in Product data. So, an inner join
# will work for this data.
SalesProductData = pd.merge(SalesData, Product, on="ProductID", how="inner")
print("The merged SalesProductData data frame.")
print(SalesProductData)
print("")
print("Task 2 - Data Cleaning")
# Counting the NAs across the dimensions showsthat there is no missing data. I also ran
# dropna and saw that the result had the same shape as the input. So, I am confident that
# there is no missing data. Which worries me. Why would you ask us to address missing data
# if there was none.
print(f"The merged dataframe has {SalesProductData.isnull().sum().sum()} missing values.")
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
# Next select thos values.
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
print("Sales records grouped by category with toal and average sales by category:")
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

Implement a NumPy program to pad strings with leading zeros to create a uniform
numeric string length.

Task Description:
- Given an array of string elements representing numbers, transform each element
  into a 5-digit numeric string.
- Pad strings with fewer than 5 digits with leading zeros.
- Strings with 5 or more digits should remain unchanged.

Example:
- Original Array: ['2', '11', '234', '1234', '12345']
- Formatted Output: ['00002', '00011', '00234', '01234', '12345']

Implementation Requirement:
- Utilize NumPy's capabilities for efficient string manipulation and array processing.
"""
P3Code = """
import numpy as np
def PadTo5(anArray):
    import numpy as np
    mask = np.char.str_len(anArray) < 6    # Needed because zfill will truncate everything to 5
    anArray[mask] = np.char.zfill(anArray[mask], 5)
    return anArray
X = np.array(['2', '11', '234', '1234', '12345'])
print(f"Original Array: {X}")
Y = PadTo5(X)
print(f"Formatted Output: {Y}")
M = np.array(['2', '11', '234', '1234', '12345', '1234567'])
print(f"Original Array: {M}")
N = PadTo5(M)
print(f"Formatted Output: {N}")
"""
DoProblem(P3Text, P3Code)

