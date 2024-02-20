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

Use NumPy to identify unique elements in an array and count their occurrences.
Problem Statement:
  Given the NumPy array:
  x = [3, 1, 4, 2, 4, 3, 6, 1, 2, 5, 5, 6, 2, 3]

  Write a Python function using NumPy to accomplish the following tasks:
  1. Extract an array of unique elements from array x.
  2. Create an array representing the count of each unique element in x.

  Expected Output:
  For the provided array x, your function should return:
  - Unique elements array: [1, 2, 3, 4, 5, 6]
  - Counts array: [2, 3, 3, 2, 2, 2]
"""
P1Code = """
import numpy as np
def UniqueElements(anArray):
  import numpy as np
  ue, ca = np.unique(anArray,return_counts = True)
  print(f"Unique elements array: {ue}")
  print(f"Counts array: {ca}")
x = np.array([3, 1, 4, 2, 4, 3, 6, 1, 2, 5, 5, 6, 2, 3])
UniqueElements(x)
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

P4Text = """
#######################################################################################
# Problem 4
#######################################################################################

Problem:

Implement a Python function using NumPy to convert Cartesian coordinates to polar
coordinates.

Task Details:
- Generate a random 10x2 matrix using NumPy, where each row represents a
  Cartesian coordinate (x, y).
- Develop a function to convert these Cartesian coordinates into polar coordinates
  (r, theta).
- The polar coordinates should be calculated as follows:
  - r = sqroot(x**2 + y**2)  (radial distance)
  - theta = arctan(y/x) (angle in radians)
- The function should return a new 10x2 matrix with polar coordinates.

Example: For a point (x, y) in the Cartesian coordinate system, the corresponding
polar coordinates (r, theta) should be computed and stored in the resulting matrix.
"""

P4Code = """
import numpy as np
def Cart2Polar(cartArray):
  import numpy as np
  polarArray = cartArray.copy()
  for i in range(cartArray.shape[0]):
    polarArray[i,0]=np.sqrt((cartArray[i,0]**2) + (cartArray[i,1]**2))
    polarArray[i,1]=np.arctan(cartArray[i,1] / cartArray[i,0])
  return polarArray
cartArray = np.random.uniform(-100,100,(10,2))
polarArray = Cart2Polar(cartArray)
np.set_printoptions(suppress=True,precision=4)
print("Cartesian")
print(cartArray)
print("Polar")
print(polarArray)
"""
DoProblem(P4Text, P4Code)

P5Text = """
#######################################################################################
# Problem 5
#######################################################################################

Problem:

Manually compute the covariance matrix of two given datasets without using the
built-in 'numpy.cov' function.

Task Description:
- Given two 1D NumPy arrays x and y, representing two different datasets.
- Write a Python function using NumPy to calculate the covariance matrix of x
  and y.
- The function should manually compute the covariance values, without utilizing
  the 'numpy.cov' function.
- Validate your function by comparing its output with manually computed covariance values.

Covariance Formula:
- The covariance between two variables x and y can be computed as:
        
        Cov(x,y)=E[(x-Ex)(y-Ey)]=E[xy]-(Ex)(Ey).

Expected Output:
- A 2x2 covariance matrix representing the covariance between x and y.
"""
P5Code = """
import numpy as np
def cov_value(x,y):
  mean_x = sum(x) / float(len(x))
  mean_y = sum(y) / float(len(y))
  sub_x = [i - mean_x for i in x]
  sub_y = [i - mean_y for i in y]
  sum_value = sum([sub_y[i]*sub_x[i] for i in range(len(x))])
  denom = float(len(x)-1)
  cov = sum_value/denom
  return cov
def covariance(x, y):
  c = np.array([[cov_value(x,x), cov_value(y,x)], [cov_value(x,y), cov_value(y,y)]])
  return c
np.set_printoptions(suppress=True,precision=2)
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 1, 1, 1, 1])
print(f"The x array is {x} and the y array is {y}.")
print("Manually calculated covariance:")
print(covariance(x, y))
print("NumPy calculated covariance:")
print(np.cov(x, y))
print("")
x = np.random.uniform(-100,100,10)
y = np.random.uniform(-100,100,10)
print(f"The x array is {x} and the y array is {y}.")
print("Manually calculated covariance:")
print(covariance(x, y))
print("NumPy calculated covariance:")
print(np.cov(x, y))
"""
print(P5Text)
print("Code:")
print(P5Code)
print("Execution:")
print("")
exec(P5Code)

P6Text = """
#######################################################################################
# Problem 6
#######################################################################################

Problem:

Create a 2D matrix from a given 1D array using specific window length and strides.

Problem Statement:
Consider the following 1D NumPy array named 'arr'. Your task is to write a Python
program using NumPy to transform 'arr' into a 2D matrix. The matrix should be
constructed by applying a sliding window approach with a specified window length
and stride.
Task Details:
1. Given a 1D NumPy array 'arr'.
2. Create a 2D matrix where each row is generated by sliding a window of length
   4 over 'arr'.
3. The stride for the sliding window should be 2 elements.
4. Example: If 'arr' is [0, 1, 2, 3, 4, 5, 6, 7, 8, . . .], the resulting matrix should be:

              0  1  2  3
              2  3  4  5
              4  5  6  7
              .. .. .. ..
- Provide the Python code for generating the 2D matrix from 'arr'.
"""
P6Code = """
import numpy as np
window = 4
stride = 2
arr = np.arange(20)
maxsteps = int((arr.shape[0] - window) / stride)
newarr = np.zeros((maxsteps,window), dtype=int)
for i in range(maxsteps):
  start = i * stride
  end = start + window
  newarr[i] = arr[start:end]
print("Input vector is:")
print(arr)
print("")
print(f"Using the input vector with a window of {window} and stride of {stride} results in:")
print(newarr)
"""
DoProblem(P6Text, P6Code)

P7Text = """
#######################################################################################
# Problem 7
#######################################################################################

Problem:

Develop a NumPy program to compute one-hot encodings for a given array.

Problem Statement:
One-hot encoding is a process by which categorical variables are converted into a
binary (0 or 1) matrix. Your task is to write a Python function using NumPy to
create one-hot encodings for each unique value in a given array.

Task Details:
1. Given the 1D NumPy array: array([2, 3, 2, 4, 1, 2]).
2. Your function should compute the one-hot encoding for this array.
3. Each unique value in the array should correspond to a column in the resulting
   binary matrix.

Example:
- Input Array: array([2, 3, 2, 4, 1, 2])
- One-Hot Encoding Output:

        0 1 0 0
        0 0 1 0
        0 1 0 0
        0 0 0 1
        1 0 0 0
        0 1 0 0

Submission:
- Provide the Python code for your one-hot encoding function. The ONLY library
  you should import to solve this problem is Numpy.
"""
P7Code = """
import numpy as np
def OneHotEncoding(aVector):
  cols = int(np.max(aVector) + 0.5)
  rows = aVector.shape[0]
  OHE = np.zeros((rows,cols), dtype=int)
  for i in range(rows):
    OHE[i,aVector[i] - 1] = 1
  return OHE
vec = np.array([2, 3, 2, 4, 1, 2])
OHE = OneHotEncoding(vec)
print(f"The input vector is: {vec}.")
print("The one-hot encoding is:")
print(OHE)
"""
DoProblem(P7Text, P7Code)