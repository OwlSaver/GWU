
"""

2. You are provided with two datasets: sales data.csv and product info.csv.
• sales data.csv contains transaction records with columns: ’TransactionID’,
’ProductID’, ’Date’, ’Quantity’, and ’Price’.
• product info.csv contains product details with columns: ’ProductID’, ’ProductName’, ’Category’.
Your task involves multiple steps of data manipulation using Pandas and NumPy to
extract insights from these datasets.
Tasks:
1. Data Loading and Merging:
• Load both datasets using Pandas.
• Merge them into a single DataFrame on ’ProductID’.
2. Data Cleaning:
• Check for and handle any missing values in the merged dataset.
• Convert the ’Date’ column to a DateTime object.
3. Data Analysis using Slicing and Indexing:
• Create a new column ’TotalSale’, calculated as ’Quantity’ * ’Price’.
• Using slicing, create a subset DataFrame containing only transactions from
the last quarter of the year (October, November, December).
• Using Boolean indexing, find all transactions for a specific ’Category’ (e.g.,
’Electronics’).
• Extract all transactions where the ’TotalSale’ is above the 75th percentile
of the ’TotalSale’ column using NumPy functions.
4. Advanced Indexing:
• Using loc and iloc, perform the following:
– Select all rows for ’ProductID’ 101 and columns ’ProductName’ and
’TotalSale’.
– Select every 10th row from the merged dataset and only the columns
’Date’ and ’Category’.
5. Grouping and Aggregation:
• Group the data by ’Category’ and calculate the total and average ’TotalSale’
for each category.
6. Time-Series Analysis:
• Resample the data on a monthly basis and calculate the total ’Quantity’
sold per month.
Page 2
Final Deliverables:
• Provide the code for each step.
• Include comments explaining your approach.
• Display the first 5 rows of the DataFrame after each major step.

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
HW4F1 = pd.read_csv(r".\gwu\SEAS 6414\homework4_file1.csv")
HW4F1.sort_values(by=['merchant', 'time'], inplace=True)
print(HW4F1)

HW4F1New = HW4F1.groupby("merchant").agg(
    min_amount=("amount_usd_in_cents", "min")
    , max_amount=("amount_usd_in_cents", "max")
    , trans_amount_avg=("amount_usd_in_cents", "mean")
    , trans_amount_volume=("amount_usd_in_cents", "sum")
    , trans_frequency=("amount_usd_in_cents", "count")
    , most_recent_date=("time", "max")
)
HW4F1New['trans_recency'] = (pd.to_datetime(HW4F1New['most_recent_date']) - dt.datetime(2035, 1, 1)).dt.days



print(HW4F1New)
"""
DoProblem(P1Text, P1Code)

P2Text = """
#######################################################################################
# Problem 2
#######################################################################################

Problem:

Generate a series of normal random variables for different sample sizes and compute
their averages.
Task:
- For each N in {5, 20, 100, 500, 2000, 50000}, generate N normal random
  variables.
- Each set of random variables should have a mean of 10 and a standard deviation
  of 5.
- Compute the average of these random variables for each N.
- Store the averages in a NumPy array.
- Additionally, write the results to a file using NumPy's save function.
Provide a printout of the final array. (Note: You do not need to submit the file
itself.)

Expected Output: A NumPy array containing the average values for each specified
N.
"""
P2Code = r"""
import numpy as np
S = {5, 20, 100, 500, 2000, 50000}
T = np.array([np.average(np.random.normal(10,5,N)) for N in S])
np.save('.\\HW3Output.npy',T)
print(T)
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

