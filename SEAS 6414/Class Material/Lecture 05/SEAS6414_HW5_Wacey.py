#######################################################################################
# This file has the problem statement and code solution for each problem. They are
# stored in Python multi line strings. The function DoProblem prints the problem
# statement, prints the code and then uses the exec() function to execute the code.
# This worked fine for problems 1-8. For problem 9, the calling of functions defined 
# in the exec() did not work. So, it is slightly different.
#######################################################################################
def DoProblem(ProblemNumber, ProblemName, ProblemDescription, ProblemCode):
    RunList = [0,1,2,3,4,5,6,7,8,9,10]
    #RunList = [0,5]
    if ProblemNumber in RunList:
      if ProblemNumber != "" and ProblemName != "":
        print("#######################################################################################")
        print(f"# Problem {ProblemNumber} - {ProblemName}")
        print("#######################################################################################")
        print("")
        print("Problem:")
        print("")
      if ProblemDescription != "":
        print(ProblemDescription)
        print("")
      if ProblemCode != "":
        print("Code:")
        print(ProblemCode)
        print("Execution:")
        print("")
        exec(ProblemCode)
        print("")

#######################################################################################
#
# Example running this file:
# C:\Users\Micha\github>py ".\GWU\SEAS 6414\SEAS6414_HW5_Wacey.py" >".\GWU\SEAS 6414\SEAS6414_HW5_Wacey.txt"
#
#######################################################################################

WNumber = 5
PNumber = 0
PName = ""
PText = f"""
#######################################################################################
# Week {WNumber}
#######################################################################################
"""
PCode = ""
DoProblem(PNumber, PName, PText, PCode)

PNumber = 1
PName = "Data Cleaning and Exploration"
PText = f"""
- Load the zillow feature sample.csv dataset using Pandas and report any missing values
  per column. Create a strategy to handle these missing values, justifying your approach.
- Generate a summary table that shows the mean, median, and standard deviation of
  taxvaluedollarcnt, structuretaxvaluedollarcnt, and landtaxvaluedollarcnt
  for properties built in each decade (1960s, 1970s, etc.).
"""
PCode = """
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)
print("")
ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")
print(f"The Zillow Feature Sample has {ZillowFeatureSample.isnull().sum().sum()} missing values.")
print(f"The Zillow Feature Sample missing values by feature:")
print(ZillowFeatureSample.isnull().sum())
print("")
print("My strategy to address the missing values is based on the number of missing features and")
print("an understanding of the data. The steps I would go through:")
print(" 1)  For each feature, determine if there is a logical value for it, if it is missing. For")
print("     example, if the number of fireplaces is missing, then assume it is zero.")
print(" 2)  Any feature that is missing under 2% of the values and is continuous would be")
print("     replaced with the mean od the non missing values.")
print(" 3)  Any feature that is missing more than 60% of its values, would be removed.")
print(" 4)  Next I would sort the rows by the number of missing features they have. I would look")
print("     for a natural break where there are two many missing features. These rows would be")
print("     removed.")
print(" 5)  I would then compare the mean and standard deviation of each feature in the original")
print("     dataset to the one adjusted for missing values. If any large differences appear, I")
print("     look into why this happened and what can be done about it.")
print("")
ZillowFeatureSample['decadebuilt'] = ["0000" if yb != yb else str(int(yb-(yb%10))) for yb in ZillowFeatureSample.yearbuilt]
ZillowFeatureSampleSummary = ZillowFeatureSample.groupby("decadebuilt").agg(
   row_count=("taxvaluedollarcnt", "count")
   , taxvaluedollarcnt_mean=("taxvaluedollarcnt", "mean")
   , taxvaluedollarcnt_median=("taxvaluedollarcnt", "median")
   , taxvaluedollarcnt_std=("taxvaluedollarcnt", "std")
   , structuretaxvaluedollarcnt_mean=("structuretaxvaluedollarcnt", "mean")
   , structuretaxvaluedollarcnt_median=("structuretaxvaluedollarcnt", "median")
   , structuretaxvaluedollarcnt_std=("structuretaxvaluedollarcnt", "std")
   , landtaxvaluedollarcnt_mean=("landtaxvaluedollarcnt", "mean")
   , landtaxvaluedollarcnt_median=("landtaxvaluedollarcnt", "median")
   , landtaxvaluedollarcnt_std=("landtaxvaluedollarcnt", "std")
)
print("")
print("Summary table:")
print(ZillowFeatureSampleSummary)
"""
DoProblem(PNumber, PName, PText, PCode)

PNumber = 2
PName = "Feature Engineering"
PText = f"""
- Create a new feature Age that represents the age of each property from the yearbuilt
  column, considering the dataset's latest assessmentyear.
- Develop a binary feature HasPool based on the poolcnt column, where 1 indicates
  the presence of a pool and 0 or NaN indicates no pool.
- Calculate and return the descriptive statistics for the age of the properties. Specifically,
  report the median age of the properties based on the yearbuilt and the
  latest assessmentyear.
- Generate and plot a bar chart of the counts of the binary feature ”HasPool” created
  earlier. Set the y-axis to a logarithmic scale to better visualize the distribution of
  properties with and without pools.
"""
PCode = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)
print("")
ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")
ZillowFeatureSample['Age'] = [ZillowFeatureSample["assessmentyear"].max() - yb for yb in ZillowFeatureSample.yearbuilt]
ZillowFeatureSample['HasPool'] =  [1 if pc > 0 else 0 for pc in ZillowFeatureSample.poolcnt]

print("")
print("Summary table:")
print(ZillowFeatureSample[["Age", "yearbuilt", "poolcnt", "HasPool"]])
print("")
print(f"The medan age of the properties based on Year Built and latest assement year is {ZillowFeatureSample["Age"].median()}.")
print("")
x = ZillowFeatureSample['HasPool'].value_counts().plot(kind='bar')
x.set_yscale('log')
plt.title(f"Problem {PNumber} A:Houses with and without pools")
plt.show()
"""
DoProblem(PNumber, PName, PText, PCode)

PNumber = 3
PName = "Correlation Analysis"
PText = f"""
- Using NumPy, calculate the Pearson correlation coefficient between bedroomcnt
  and bathroomcnt. Visualize the correlation matrix of the numerical features of the
  dataset using a heatmap in matplotlib.
"""
PCode = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

print("")
ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")

ZillowFeatureSampleSmall = ZillowFeatureSample.loc[:,['bedroomcnt','bathroomcnt']]
print(ZillowFeatureSampleSmall.corr(numeric_only=True))
plt.imshow(ZillowFeatureSampleSmall.corr(numeric_only=True), cmap='hot', interpolation='nearest')
plt.title(f"Problem {PNumber} A: Correlation of Bedrooms and Bathrooms")
plt.show()
"""
DoProblem(PNumber, PName, PText, PCode)


PNumber = 4
PName = "Geospatial Analysis"
PText = f"""
- Plot a scatter plot of latitude and longitude to visualize the geographical distribution
  of properties. Overlay this plot with a density estimate to highlight property
  clusters.
"""
PCode = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

print("")
ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")
print("I did the scatter and looked for a density plot to overlay. I could not find one. So,")
print("instead I have calculated an area and used it to color the plots to show density. But")
print("the density seems to be consistent. So, it does not really show much.")
# Get just the columns that we need and drop any rows with NaN. The gaussian calculation
# does not work with NaN values.
ZillowFeatureSampleSmall = ZillowFeatureSample.loc[:,['longitude','latitude']].dropna()
z = gaussian_kde(ZillowFeatureSampleSmall["longitude"])(ZillowFeatureSampleSmall["latitude"])
fig, ax = plt.subplots()
ax.scatter(ZillowFeatureSampleSmall["longitude"], ZillowFeatureSampleSmall["latitude"], c=z, s=100)
plt.title(f"Problem {PNumber} A: Scatter Plot")
plt.show()
"""
DoProblem(PNumber, PName, PText, PCode)

PNumber = 5
PName = "Market Value Analysis"
PText = f"""
- Visualize the trend of average taxvaluedollarcnt over the years using a line chart.
  Add a shaded area representing the 95% confidence interval for the average values.
- Create a boxplot to compare the distribution of taxvaluedollarcnt across different
  buildingqualitytypeid.
"""
PCode = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

print("")
ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")
ZillowFeatureSample['assessyear'] = ["0000" if ay != ay else str(int(ay)) for ay in ZillowFeatureSample.assessmentyear]
ZillowFeatureSampleSummary = ZillowFeatureSample.groupby("assessyear").agg(
   taxvaluedollarcnt_mean=("taxvaluedollarcnt", "mean")
   , taxvaluedollarcnt_std=("taxvaluedollarcnt", "std")
).dropna()

ZillowFeatureSampleSummary['plus95'] = ZillowFeatureSampleSummary.apply(lambda row: row.taxvaluedollarcnt_mean + (2 * row.taxvaluedollarcnt_std), axis=1)
ZillowFeatureSampleSummary['minus95'] = ZillowFeatureSampleSummary.apply(lambda row: row.taxvaluedollarcnt_mean - (2 * row.taxvaluedollarcnt_std), axis=1)
plt.plot(ZillowFeatureSampleSummary.index, ZillowFeatureSampleSummary["taxvaluedollarcnt_mean"], label="Average")
plt.fill_between(ZillowFeatureSampleSummary.index, ZillowFeatureSampleSummary["plus95"], ZillowFeatureSampleSummary["minus95"], alpha=0.2, label="Confidence Interval")
plt.title(f"Problem {PNumber} A: Average with Confidence Interval")
plt.show()

ax = ZillowFeatureSample.boxplot(column="taxvaluedollarcnt", by="buildingqualitytypeid")
plt.title(f"Problem {PNumber} B: Box Plot")
plt.show()
"""
DoProblem(PNumber, PName, PText, PCode)

PNumber = 6
PName = "Tax Analysis"
PText = f"""
- Analyze the relationship between taxamount and taxvaluedollarcnt using a scatter plot
  and fit a linear regression line to it. Calculate the R-squared value for this
  fit.
"""
PCode = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

print("")
ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")
ZillowFeatureSampleSmall = ZillowFeatureSample.loc[:,['taxamount','taxvaluedollarcnt']].dropna()
X = np.array(ZillowFeatureSampleSmall["taxvaluedollarcnt"]).reshape(-1, 1)
y = np.array(ZillowFeatureSampleSmall["taxamount"])
# fit the model
reg = LinearRegression().fit(X, y)
ZillowFeatureSampleSmall['predicted'] = reg.predict(X)
rsquare = reg.score(X, y)

fig, ax = plt.subplots()
ax.text(1, 150000, f"$R^2$ = {rsquare}", bbox=dict(facecolor='red', alpha=0.5))
ZillowFeatureSampleSmall.plot.scatter(x = 'taxvaluedollarcnt', y = 'taxamount', ax = ax)
ZillowFeatureSampleSmall.plot.line(x = 'taxvaluedollarcnt', y = 'predicted', color = 'red', ax = ax)
plt.title(f"Problem {PNumber} A: Linear Regression")
plt.show()
"""
DoProblem(PNumber, PName, PText, PCode)

PNumber = 7
PName = "Comparative Analysis"
PText = f"""
- For properties with numberofstories more than 1, compare the average
  calculatedfinishedsquarefeet against those with only 1 story using a bar chart.
- Compare the taxvaluedollarcnt for properties with and without a fireplace (fireplaceflag)
  using a violin plot.
"""
PCode = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

print("")
ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")
ZillowFeatureSample['storygroup'] = ["Multi" if nos > 1 else "One" for nos in ZillowFeatureSample.numberofstories]
ZillowFeatureSample['fpgroup'] = ["None" if fpf != fpf else "Has" for fpf in ZillowFeatureSample.fireplaceflag]
ZillowFeatureSampleSG = ZillowFeatureSample.groupby("storygroup").agg(
   calculatedfinishedsquarefeet_mean=("calculatedfinishedsquarefeet", "mean")
).dropna()

x = ZillowFeatureSampleSG.plot.bar()
plt.title(f"Problem {PNumber} A: Houses average square feet for one or multi stories")
plt.show()

sb.violinplot(x = 'fpgroup', y = "taxvaluedollarcnt", data = ZillowFeatureSample, inner="stick")
plt.title(f"Problem {PNumber} B: Fireplace Comparison")
plt.show()
"""
DoProblem(PNumber, PName, PText, PCode)

PNumber = 8
PName = "Time-Series Forecasting (Advanced)"
PText = f"""
- Group the data by yearbuilt and calculate the annual mean of landtaxvaluedollarcnt.
  Using this time series data, create a forecast plot for the next 10 years with a rolling
  mean and standard deviation.
"""
PCode = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

print("")
ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")
# Drop outliers before 1900
ZillowFeatureSampleYB = ZillowFeatureSample.loc[ZillowFeatureSample['yearbuilt'] > 1900]
ZillowFeatureSampleYB = ZillowFeatureSampleYB.groupby("yearbuilt").agg(
   landtaxvaluedollarcnt_mean=("landtaxvaluedollarcnt", "mean")
).dropna()

print("I searched the web and found several articles about ARIMA / SARIMA as a way to train")
print("Models for prediction. I was not sure that this was the path to take. So, instead")
print("just used regression on a ten year window and predicted the next year from that.")

ZillowFeatureSampleSmall = ZillowFeatureSample.loc[:,['landtaxvaluedollarcnt','yearbuilt']].dropna()
X = np.array(ZillowFeatureSampleSmall["yearbuilt"]).reshape(-1, 1)
y = np.array(ZillowFeatureSampleSmall["landtaxvaluedollarcnt"])
# fit the model
reg = LinearRegression().fit(X, y)
ZillowFeatureSampleSmall['predicted'] = reg.predict(X)
future_years = np.array(range(2015, 2024))
predicted_prices = np.array(reg.predict(future_years.reshape(-1, 1)))
predicted_data = pd.DataFrame({'yearbuilt': future_years, 'landtaxvaluedollarcnt':predicted_prices}).set_index('yearbuilt')
plt.plot(ZillowFeatureSampleYB, color = "black", label = "History")
plt.plot(predicted_data, color = "red", label = "Prediction")
plt.ylabel('House Price')
plt.xlabel('Year Built')
plt.legend()
plt.title(f"Problem {PNumber} A: Prediction")
plt.show()

print("OK, that did not work well. So, back to the drawing board. Below is another attempt that I think does")
print("what you asked.")

# Define window size for the rolling window
window_size = 10
# Calculate rolling mean
ZillowFeatureSampleYB["rolling_mean"] = ZillowFeatureSampleYB["landtaxvaluedollarcnt_mean"].rolling(window=window_size).mean()
# Calculate rolling standard deviation
ZillowFeatureSampleYB["rolling_std"] = ZillowFeatureSampleYB["landtaxvaluedollarcnt_mean"].rolling(window=window_size).std()
# Extend index for 10 years
future_years = np.array(range(2015, 2024))
# Extend the existing data with NaNs for future dates
ZillowFeatureSampleYB_extended = pd.concat([ZillowFeatureSampleYB,pd.DataFrame(index=future_years)])
# Fill NaN values with the last rolling mean
ZillowFeatureSampleYB_extended["rolling_mean"] = ZillowFeatureSampleYB_extended["rolling_mean"].fillna(method="ffill")
# Calculate the upper and lower bounds based on rolling mean and standard deviation
ZillowFeatureSampleYB_extended["upper_bound"] = ZillowFeatureSampleYB_extended["rolling_mean"] + 2 * ZillowFeatureSampleYB_extended["rolling_std"]
ZillowFeatureSampleYB_extended["lower_bound"] = ZillowFeatureSampleYB_extended["rolling_mean"] - 2 * ZillowFeatureSampleYB_extended["rolling_std"]
# Plot observed data, rolling mean, and bounds
plt.figure(figsize=(12, 6))
plt.plot(ZillowFeatureSampleYB.index, ZillowFeatureSampleYB["landtaxvaluedollarcnt_mean"], label="Observed")
plt.plot(ZillowFeatureSampleYB_extended.index, ZillowFeatureSampleYB_extended["rolling_mean"], label="Rolling Mean")
plt.fill_between(ZillowFeatureSampleYB_extended.index, ZillowFeatureSampleYB_extended["upper_bound"], ZillowFeatureSampleYB_extended["lower_bound"], alpha=0.2, label="Confidence Interval")
# Add labels and title
plt.xlabel("Year")
plt.ylabel("Land Tax Value")
plt.title(f"Problem {PNumber} B: Forecast with Rolling Mean and Standard Deviation")
# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
# Show the plot
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
"""
DoProblem(PNumber, PName, PText, PCode)

PNumber = 9
PName = "Amenities Impact Analysis"
PText = f"""
- Determine how the presence of a hot tub or spa (hashottuborspa) and air conditioning (airconditioningtypeid)
  impacts the taxvaluedollarcnt. Use a grouped bar chart to represent the average taxvaluedollarcnt for
  properties with and without these amenities.
- Investigate if there is a significant difference in the calculatedfinishedsquarefeet for properties with a
  basement (basementsqft) versus those without. Perform a hypothesis test and visualize the results using a
  histogram overlaid with the probability density function.
"""
PCode = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")
ZillowFeatureSample['HasHTS'] = ["No Hot Tub" if hhts != hhts else "Hot Tub" for hhts in ZillowFeatureSample.hashottuborspa]
ZillowFeatureSample['HasAC'] = ["No AC" if act != act or act == 5.0 else "AC" for act in ZillowFeatureSample.airconditioningtypeid]

ZillowFeatureSampleAM = ZillowFeatureSample.groupby(["HasHTS", "HasAC"]).agg(
   taxvaluedollarcnt_mean=("taxvaluedollarcnt", "mean")
).dropna()

x = ZillowFeatureSampleAM.plot.bar()
#x.ticklabel_format(style='plain')
plt.xlabel("Amenities")
plt.ylabel("Average Tax Value ($)")
plt.title(f"Problem {PNumber} A: Average Tax Value based on Amenities")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

ZillowFeatureSampleSF = ZillowFeatureSample.loc[:,['calculatedfinishedsquarefeet','basementsqft']]
ZillowFeatureSampleSF['HasBasement'] = ["No Basement" if bsf != bsf or bsf <= 0 else "Basement" for bsf in ZillowFeatureSampleSF.basementsqft]
ZillowFeatureSampleHasB = ZillowFeatureSampleSF.loc[ZillowFeatureSampleSF['HasBasement'] == "Basement"]
ZillowFeatureSampleHasB = ZillowFeatureSampleHasB.drop(columns=['basementsqft','HasBasement'])
ZillowFeatureSampleNoB = ZillowFeatureSampleSF.loc[ZillowFeatureSampleSF['HasBasement'] != "Basement"]
ZillowFeatureSampleNoB = ZillowFeatureSampleNoB.drop(columns=['basementsqft','HasBasement'])

# Define the null hypothesis
H0 = "Properties with a basement will have more square feet than those without."

# Define the alternative hypothesis
H1 = "Properties with a basement will have the same or fewer square feet than those without."

# Calculate the test statistic
t_stat, p_value = stats.ttest_ind(ZillowFeatureSampleHasB, ZillowFeatureSampleNoB,nan_policy='omit')

# Print the results
print("Test statistic:", t_stat)
print("p-value:", p_value)

# Conclusion
if p_value != p_value:
  print("t Test failed.")
elif p_value < 0.05:
  print(f"Reject the null hypothesis of {H0}.")
else:
  print(f"Failed to reject the null hypothesis of {H0}.")

ZillowFeatureSampleSFA = ZillowFeatureSampleSF.drop(columns=['HasBasement'])
sns.displot(ZillowFeatureSampleSFA, kde=True)
plt.title(f"Problem {PNumber} B: Histogram overlaid with the probability density function")
plt.show()
"""
DoProblem(PNumber, PName, PText, PCode)

PNumber = 10
PName = "Neighborhood and Regional Analysis"
PText = f"""
- Group the properties by regionidneighborhood and plot a horizontal bar chart
  showing the top 10 neighborhoods with the highest average taxvaluedollarcnt.
- Using regionidzip, create a pie chart to display the proportion of total taxamount
  contributed by the top 5 zip codes. Include a separate 'other' slice for the remaining
  zip codes.
"""
PCode = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 4000)

ZillowFeatureSample = pd.read_csv("./gwu/SEAS 6414/zillow_feature_sample(1).csv")

ZillowFeatureSampleAM = ZillowFeatureSample.groupby(["regionidneighborhood"]).agg(
   taxvaluedollarcnt_mean=("taxvaluedollarcnt", "mean")
)
ZillowFeatureSampleAM = ZillowFeatureSampleAM.sort_values('taxvaluedollarcnt_mean', ascending=False).head(10)
ZillowFeatureSampleAM.plot.barh()
plt.ylabel('Neighborhood')
plt.xlabel('Average Tax Value ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.title(f"Problem {PNumber} A: Top 10 Neigborhoods")
plt.show()

ZillowFeatureSampleZIP = ZillowFeatureSample.groupby(["regionidzip"]).agg(
   taxamount_sum=("taxamount", "sum")
)
ZillowFeatureSampleZipSort = ZillowFeatureSampleZIP.sort_values('taxamount_sum', ascending=False)
ZillowFeatureSampleZipTop5 = ZillowFeatureSampleZipSort.head(5).copy()
# We have to recalulate the mean, since we cannot take the mean of the mean - originally I did mean, sum is simpler
# but this works for mean witht he proper changes and so I am leaving it in the more complicated form.
ZillowFeatureSampleZipFull = ZillowFeatureSample.join(ZillowFeatureSampleZipTop5,on='regionidzip',how='outer',)
ZillowFeatureSampleZipFull['zipgroup'] = ZillowFeatureSampleZipFull.apply(lambda row: 'other' if row.taxamount_sum != row.taxamount_sum else row.regionidzip, axis=1)
ZillowFeatureSampleZipSum = ZillowFeatureSampleZipFull.groupby(["zipgroup"]).agg(
   taxamount_sum=("taxamount", "sum")
)
ZillowFeatureSampleZipSum.plot.pie(y='taxamount_sum',legend=None)
plt.tight_layout()
plt.ylabel('Total Tax')
plt.title(f"Problem {PNumber} B: Top 5 zip codes")
plt.show()

"""
DoProblem(PNumber, PName, PText, PCode)
