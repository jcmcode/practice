import pandas as pd

### Part 1: Series
#Series = A pandas 1-dimensional labeled array that can hold any data type
#Like a single column in a spreadsheet


data = [100, 102, 104]

#Series is the constructor
series = pd.Series(data)

#print(series)
#output:
#0    100
#1    102
#2    104
#dtype: int64 Metdata about series(datatype)

#custom labels for the index (passed in a keyword argument)
series2 = pd.Series(data, index = ["a", "b", "c"])

#print(series2)
#ouput:
#dtype: int64
#a    100
#b    102
#c    104
#dtype: int64

#print(series.loc["a"]) #returns value where label is "a"

series.loc["c"] = 200 # sets value at index "c" to 200

#print(series.iloc[0]) #returns value by integer position (this would print 100)

data2 = [100, 102, 104, 200, 202]

series3 = pd.Series(data2, index = ["a", "b", "c", "d", "e"])
#print(series3[series3 >= 200]) #prints values greater or equal to 200

#Dictionary of Key Value Pairs
calories = {"Day 1": 1750, "Day 2": 2100, "Day 3": 1700}

series4 = pd.Series(calories)
#can use loc to access individual values
print(series4)
#output:
#Day 1    1750
#Day 2    2100
#Day 3    1700
#dtype: int64

#can incriment values by series.loc["Day 3"] += 500
