from statistics import *
from pandas.core.algorithms import factorize
from scipy.stats import gmean
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from ycimpute.imputer import knnimput, EM
import pandas as pd
import numpy as np
import researchpy as rp
pd.options.mode.chained_assignment = None


class DataScienceKit():
    def __init__(self) -> None:
        pass

    
    class MeasurementUnits():
        def __init__(self):
            self.data_operations = DataScienceKit.DataOperations()
            self.exploratory_data_analysis = DataScienceKit.ExploratoryDataAnalysis()


        # Arithmetic Mean
        # numeric_col: Numeric DataFrame Column
        def arithmetic_mean(self, dataset, numerical_column_name):
            if(self.exploratory_data_analysis.is_numerical(dataset, numerical_column_name)):
                if(self.data_operations.there_any_NaN_values_column(dataset, numerical_column_name) != True):
                    return mean(dataset[numerical_column_name])
                else:
                    return ValueError(f'NaN Value Error: There is missing (NaN) data in {numerical_column_name} ')
            else:
                return ValueError(f'Variable Type Error: {numerical_column_name} is not a Numerical Variable ')
        

        # Geometric Mean
        # numerical_col: Numeric DataFrame Column
        def geometric_mean(self, dataset, numerical_column_name):
            if(self.exploratory_data_analysis.is_numerical(dataset, numerical_column_name)):
                if(self.data_operations.there_any_NaN_values_column(dataset, numerical_column_name) != True):
                    return gmean(dataset[numerical_column_name])
                else:
                    return ValueError(f'NaN Value Error: There is missing (NaN) data in {numerical_column_name} ')
            else:
                return ValueError(f'Variable Type Error: {numerical_column_name} is not a Numerical Variable ')
        

        # Harmonic Mean
        # numeric_col: Numeric DataFrame Column
        def harmonic_mean(self, dataset, numerical_column_name):
            if(self.exploratory_data_analysis.is_numerical(dataset, numerical_column_name)):
                if(self.data_operations.there_any_NaN_values_column(dataset, numerical_column_name) != True):
                    return harmonic_mean(dataset[numerical_column_name])
                else:
                    return ValueError(f'NaN Value Error: There is missing (NaN) data in {numerical_column_name} ')
            else:
                return ValueError(f'Variable Type Error: {numerical_column_name} is not a Numerical Variable ')
        

        # Median
        # numeric_col: Numeric DataFrame Column
        # median_type: 1: Median, 2: Median Low, 3: Median High, 4: Median Grouped
        def median(self, dataset, numerical_column_name, median_type=1, interpolation=1):
            if(median_type == 1):
                return median(dataset[numerical_column_name])
            elif(median_type == 2):
                return median_low(dataset[numerical_column_name])
            elif(median_type == 3):
                return median_high(dataset[numerical_column_name])
            elif(median_type == 4):
                if(self.exploratory_data_analysis.is_numerical(dataset, numerical_column_name)):
                    if(self.data_operations.there_any_NaN_values_column(dataset, numerical_column_name) != True):
                        return median_grouped(dataset[numerical_column_name], interval=interpolation)
                    else:
                        return ValueError(f'NaN Value Error: There is missing (NaN) data in {numerical_column_name} ')
                else:
                    return ValueError(f'Variable Type Error: {numerical_column_name} is not a Numerical Variable')
            else:
                return ValueError("Invalid Median Type: Takes a value between 1 and 4")
        

        # Mode
        # numeric_col: Numeric DataFrame Column
        # mode_type: 1: Mode, 2: Count Mode Value
        def mode(self, dataset, numerical_column_name, mode_type=1):
            if(mode_type == 1):
                return mode(dataset[numerical_column_name])
            elif(mode_type == 2):
                return dataset[numerical_column_name].count(mode(dataset[numerical_column_name]))
            else:
                return ValueError("Invalid Mode Type: Takes a value between 1 and 2")
        

        # Variance
        # numeric_col: Numeric DataFrame Column
        # mean_data: If the mean value of the numeric column is not given as a parameter, None should remain
        def variance(self, dataset, numerical_column_name):
            if(self.exploratory_data_analysis.is_numerical(dataset, numerical_column_name)):
                if(self.data_operations.there_any_NaN_values_column(dataset, numerical_column_name) != True):
                    return variance(dataset[numerical_column_name])
                else:
                    return ValueError(f'NaN Value Error: There is missing (NaN) data in {numerical_column_name} ')
            else:
                return ValueError(f'Variable Type Error: {numerical_column_name} is not a Numerical Variable ')
        

        # Standart Deviation
        # numeric_col: Numeric DataFrame Column
        def standart_deviation(self, dataset, numerical_column_name):
            if(self.exploratory_data_analysis.is_numerical(dataset, numerical_column_name)):
                if(self.data_operations.there_any_NaN_values_column(dataset, numerical_column_name) != True):
                    return stdev(dataset[numerical_column_name])
                else:
                    return ValueError(f'NaN Value Error: There is missing (NaN) data in {numerical_column_name} ')
            else:
                return ValueError(f'Variable Type Error: {numerical_column_name} is not a Numerical Variable ')
        

        # Minimum Value
        # numeric_col: Numeric DataFrame Column
        def minimum_value(self, dataset, numerical_column_name):
            return min(dataset[numerical_column_name])
        

        # Maximum Value
        # numeric_col: Numeric DataFrame Column
        def maximum_value(self, dataset, numerical_column_name):
            return max(dataset[numerical_column_name])
        

        # Kurtosis
        # data_matrix: Numeric DataFrame Columns data_matrix = [dataset['number'], dataset['distance']]
        def kurtosis(self, data_matrix):
            df = pd.DataFrame(data=data_matrix)
            return df.kurt(axis=1)
        

        # Skew
        # data_matrix: Numeric DataFrame Columns data_matrix = [dataset['number'], dataset['distance']]
        def skewnewss(self, data_matrix):
            df = pd.DataFrame(data=data_matrix)
            return df.skew(axis=1)
        

        # Quantiles
        # numeric_col: Numeric DataFrame Columns
        # quantile_type: 1: 0.25, 2: 0.50 (Median), 3: 0.75
        def quantiles(self, dataset, numerical_column_name, quantile_type = 1):
            if(self.exploratory_data_analysis.is_numerical(dataset, numerical_column_name)):
                if(self.data_operations.there_any_NaN_values_column(dataset, numerical_column_name) != True):
                    if(quantile_type == 1):
                        return np.percentile(dataset[numerical_column_name], 25)
                    elif(quantile_type == 2):
                        return np.percentile(dataset[numerical_column_name], 50)
                    elif(quantile_type == 3):
                        return np.percentile(dataset[numerical_column_name], 75)
                    else:
                        return ValueError("Invalid Quantile Type: Takes a value between 1 and 3")
                else:
                    return ValueError(f'NaN Value Error: There is missing (NaN) data in {numerical_column_name} ')
            else:
                return ValueError(f'Variable Type Error: {numerical_column_name} is not a Numerical Variable')

        
        # Range Of Change
        # numeric_col: Numeric DataFrame Columns
        def range_of_change(self, dataset, numerical_column_name):
            if(self.exploratory_data_analysis.is_numerical(dataset, numerical_column_name)):
                return max(dataset[numerical_column_name]) - min(dataset[numerical_column_name])
            else:
                return ValueError(f'Variable Type Error: {numerical_column_name} is not a Numerical Variable')

        
        # Covariance
        # numeric_col: Numeric DataFrame Columns
        def covariance(self, dataset, numerical_column_name):
            if(self.exploratory_data_analysis.is_numerical(dataset, numerical_column_name)):
                if(self.data_operations.there_any_NaN_values_column(dataset, numerical_column_name) != True):
                    return np.cov(dataset[numerical_column_name])
                else:
                    return ValueError(f'NaN Value Error: There is missing (NaN) data in {numerical_column_name} ')
            else:
                return ValueError(f'Variable Type Error: {numerical_column_name} is not a Numerical Variable')
        

        # Correlation
        # numerical_columns: Numeric DataFrame Columns ['number', 'distance']
        # method: pearson or kendall
        def correlation(self, dataset, numerical_columns_name_list, method='pearson'):
            for column in numerical_columns_name_list:
                if dataset[column].dtypes not in ["float64", "int64"]:
                    return ValueError(f'Variable Type Error: {column} is not a Numerical Variable')
                else:
                    corr_df = dataset[numerical_columns_name_list].corr(method=method)  
            return corr_df


    class ExploratoryDataAnalysis():
        def __init__(self) -> None:
            pass


        # Copy Dataset
        # dataset: The dataset to be worked on. (DataFrame)
        def copy_dataset(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.copy()
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        
        
        # Show Head & Tail Values
        # dataset: The dataset to be worked on. (DataFrame)
        # n_value: Return top 5 five data by default
        def head_and_tail_values(self, dataset, n_value = 5,):
            if(isinstance(dataset, pd.DataFrame)):
                return pd.DataFrame(dataset.head(n_value)), pd.DataFrame(dataset.tail(n_value))
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Structural Information
        # dataset: The dataset to be worked on. (DataFrame)
        # verbose: If you do not want to get more detailed information, it should be left as None. 
        def structural_information(self, dataset, verbose=True):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.info(verbose=verbose)
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Variable Types 
        # dataset: The dataset to be worked on. (DataFrame)
        def variable_types(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.dtypes
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Change Object Type to Categorical
        # dataset: The dataset to be worked on. (DataFrame)
        # column_name_to_change: Which column is desired to be changed, the name of that column is written.
        def object_to_categorical(self, dataset, column_name_to_change):
            if(dataset[column_name_to_change].dtypes in ["object"]):
                dataset[column_name_to_change] = pd.Categorical(dataset[column_name_to_change])
                return dataset
            else:
                return ValueError(f'Column Type Error: {column_name_to_change} type is not Object')
        

        # Observation Values
        # dataset: The dataset to be worked on. (DataFrame)
        def nb_observation_values(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.shape[0]
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Variable Values
        # dataset: The dataset to be worked on. (DataFrame)
        def nb_variable_values(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.shape[1]
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Dataset Size
        # dataset: The dataset to be worked on. (DataFrame)
        def size_of_dataset(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.size
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')


        # Is the column a numerical column? It controls
        # column_name: The desired column name from the Data Set is written.
        def is_numerical(self, dataset, column_name):
            if(dataset[column_name].dtypes in ["float64", "int64"]):
                return True
            else:
                return False


        # Is the column a categorical column? It controls
        # column_name: The desired column name from the Data Set is written.
        def is_categorical(self, dataset, column_name):
            if(dataset[column_name].dtypes in ["object"]):
                return True
            else:
                return False


        # Dataframe Dimension
        # dataset: The dataset to be worked on. (DataFrame)
        def dimension_of_dataframe(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.ndim
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Series Dimension
        # dataset: The dataset to be worked on. (Series)
        # column_name: The desired column name from the Data Set is written.
        def dimension_of_series(self, dataset, column_name):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset[column_name].ndim
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
                
        
        # Columns Name
        # dataset: The dataset to be worked on. (DataFrame)
        # return_type: 1: Normal, 2: ndarray, 3: List
        def variable_names_of_dataset(self, dataset, return_type=1):
            if(isinstance(dataset, pd.DataFrame)):
                if(return_type == 1):
                    return dataset.columns
                elif(return_type == 2):
                    return dataset.columns.values
                elif(return_type == 3):
                    return list(dataset.columns.values)
                else:
                    return ValueError("Invalid Return Type: Takes a value between 1 and 3")
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Allows us to change column types
        # column_name: Column name to change
        # data_type: Data type to be changed
        def convert_variables_type(self, dataset, column_name, data_type):
            if(isinstance(dataset, pd.DataFrame)):
                if(type(data_type) == str):
                    return dataset[column_name].astype(data_type)
                else:
                    return ValueError("Data Type Variables Must Be str.")
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')


        # Ranks and ranks based on dataset variables
        # column_name: The desired column name from the Data Set is written.
        # method: min, max, average
        def create_and_sort_rank_variables(self, dataset, column_name, method="min"):
            if(isinstance(dataset, pd.DataFrame)):
                dataset["new_col"] = dataset[column_name].rank(method=method)
                return dataset.sort_values("number")
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')


        # Sorts by range within the specified column
        # column_name: The desired column name from the Data Set is written.
        # number 1 ile 3 arasında olanları sıralar. 3 dahil olmaz.
        def select_range_row_by_index(self, dataset, column_name, start_index, end_index):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset[column_name].between(start_index, end_index)
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')


        # Descriptive Statistics
        # dataset: The dataset to be worked on. (DataFrame)
        # transpose: Transpose must be True if you want to see a more detailed output
        # include: "all"
        def descriptive_statistics_of_dataset(self, dataset, transpose = False,  include=None):
            if(isinstance(dataset, pd.DataFrame)):
                if(transpose == False):
                    return dataset.describe(include=include)
                else:
                    return dataset.describe(include=include).T
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Descriptive Statistics - Numerical Variables
        # numeric_columns: A column name with numeric values ​​is written in the Data Set
        def descriptive_statistics_of_numerical_variable(self, dataset, numerical_columns):
            if(isinstance(dataset, pd.DataFrame) and self.is_numerical(dataset, numerical_columns)):
                return rp.summary_cont(dataset[numerical_columns])
            else:
                return ValueError(f'Error: Dataset type is not DataFrame or {numerical_columns} is not numerical')
        

        # Descriptive Statistics - Categorical Variables
        # categoric_columns: A column name with categorical values ​​is written in the Data Set
        def descriptive_statistics_of_categorical_variable(self, dataset, categorical_columns):
            if(isinstance(dataset, pd.DataFrame) and self.is_categorical(dataset, categorical_columns)):
                return rp.summary_cat(dataset[categorical_columns])
            else:
                return ValueError(f'Error: Dataset type is not DataFrame or {categorical_columns} is not categorical')


        # Select categorical data in dataset scope
        def select_categorical_variables(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.select_dtypes(include=["object"])
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Select numerical data in dataset scope
        def select_numerical_variables(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.select_dtypes(include=['float64', 'int64'])
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')


        # Accessing the frequencies of classes of categorical variables
        # categoric_column: The name of the column where we will count the frequencies is written
        # plot_state: Send as True to graph the frequencies
        def frequency_of_categorical_variables(self, dataset, categorical_column):
            if(isinstance(dataset, pd.DataFrame) and self.is_categorical(dataset, categorical_column)):
                return dataset[categorical_column].value_counts()
            else:
                return ValueError(f'Error: Dataset type is not DataFrame or {categorical_column} is not categorical')
        

        # How many unique categorical values ​​are there
        # categoric_column: Write the name of the desired categorical column
        def frequency_unique_categorical_variables(self, dataset, categorical_column):
            if(isinstance(dataset, pd.DataFrame) and self.is_categorical(dataset, categorical_column)):
                return dataset[categorical_column].value_counts().count()
            else:
                return ValueError(f'Error: Dataset type is not DataFrame or {categorical_column} is not categorical')
        

        # What percentage of the dataset is the categorical data in the specified column?
        # categoric_column: Write the name of the desired categorical column
        def categorical_variables_percentage_of_dataset(self, dataset, categorical_column):
            if(isinstance(dataset, pd.DataFrame) and self.is_categorical(dataset, categorical_column)):
                return (dataset[categorical_column].value_counts()/np.product(dataset.shape)) * 100
            else:
                return ValueError(f'Error: Dataset type is not DataFrame or {categorical_column} is not categorical')


        # It is used to access the classes of categorical data and the number of classes in the dataset.
        # categoric_column: Write the name of the desired categorical column
        def find_unique_categorical_variables(self, dataset, categorical_column):
            if(isinstance(dataset, pd.DataFrame) and self.is_categorical(dataset, categorical_column)):
                return dataset[categorical_column].unique()
            else:
                return ValueError(f'Error: Dataset type is not DataFrame or {categorical_column} is not categorical')


        # Statistical measurements of a selected single categorical column with the remaining numeric columns
        # categoric_column: Write the name of the desired categorical column 
        def measure_operations_of_single_categorical_variables_with_all_numerical_variables(self, dataset, categorical_column):
            objectVariable = []
            columns = dataset.drop(columns=categorical_column).columns
            for column in columns:
                if dataset[column].dtypes in ["float64", "int64"]:
                    print(dataset.groupby(categorical_column).agg({column:["mean", "median", 'min', 'max', "std", 'sum', "var"]}))
                else:
                    objectVariable.append(column)
            print(f"Categorical Variables: {objectVariable}")
        

        # Operations of a selected single categorical column with another selected numeric column
        # categoric_column: Write the name of the desired categorical column 
        # numeric_column: Write the name of the desired numerical column 
        def measure_operations_of_single_categorical_variables_single_numerical_variables(self, dataset, categorical_column, numerical_column):
            if(dataset[numerical_column].dtypes in ["float64", "int64"]):
                print(dataset.groupby(categorical_column).agg({numerical_column:["mean", "median", 'min', 'max', "std", 'sum', "var"]}))
            else:
                return ValueError("Not Numeric Data")
        
    
        # Categorical data sent as a list are grouped and operations are performed according to the specified numeric column.
        # categoric_column: Write the name of the desired categorical column 
        # numeric_column: Write the name of the desired numerical column 
        def measure_operations_of_multiple_categorical_variables_single_numerical_variables(self, dataset, categorical_columns, numerical_column):
            if(dataset[numerical_column].dtypes in ["float64", "int64"]):
                print(dataset.groupby(categorical_columns).agg({numerical_column:["mean", "median", 'min', 'max', "std", 'sum', "var"]}))
            else:
                return ValueError("Not Numeric Data")
  

    class DataOperations():
        def __init__(self) -> None:
            pass


        # Is there any missing data?
        # dataset: The dataset to be worked on. (DataFrame)
        def there_any_NaN_values(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.isnull().values.any()
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Is there any missing data in column?
        # column_name: The desired column name from the Dataset is written.
        def there_any_NaN_values_in_column(self, dataset, column_name):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset[column_name].isnull().values.any()
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')


        # Total missing values ​​in all variables
        # dataset: The dataset to be worked on. (DataFrame)
        def total_NaN_values(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.isnull().sum()
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Percentage of null values ​​in dataset
        # dataset: The dataset to be worked on. (DataFrame)
        def percentage_NaN_values(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return (dataset.isnull().sum()/np.product(dataset.shape)) * 100
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Drop NAN values on rows
        # dataset: The dataset to be worked on. (DataFrame)
        def drop_NaN_values_row(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.dropna()
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Drop NAN values on columns, 1 adet eksik veri bile olsa o sütunu komple siler. 
        # dataset: The dataset to be worked on. (DataFrame)
        def drop_NaN_values_column(self, dataset):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.dropna(axis=1)
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')


        # Fill all NAN values ​​with your query
        # dataset: The dataset to be worked on. (DataFrame)
        # column_name: The name of the column we want to fill in the empty values ​​is written.
        # query: Write a query to fill in the blank values ​​in the column
        def fill_NaN_values_column(self, dataset, column_name, query, inplace=True):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset[column_name].fillna(query, inplace=inplace)
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Fill all NAN values with dataset mean value
        # dataset: The dataset to be worked on. (DataFrame)
        # query: Write a query to fill in the blank values ​​in the column
        def fill_NaN_values_dataset(self, dataset, query, inplace=True):
            if(isinstance(dataset, pd.DataFrame)):
                return dataset.fillna(query, inplace=inplace)
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # Filling all the NaN values in the Dataset with the KNN algorithm
        # k = 4 is the default value.
        def fill_NaN_values_knn(self, dataset, k=4):
            np.set_printoptions(suppress=True)
            column_name_list = list(dataset)
            n_df = np.array(dataset)
            knn_df = knnimput.KNN(k = k).complete(n_df)
            knn_df = pd.DataFrame(knn_df, columns=column_name_list)
            return knn_df


        # Filling all the NaN values in the Dataset with the EM algorithm
        def fill_NaN_values_EM(self, dataset):
            np.set_printoptions(suppress=True)
            column_name_list = list(dataset)
            n_df = np.array(dataset)
            EM_df = EM().complete(n_df)
            EM_df = pd.DataFrame(EM_df, columns=column_name_list)
            return EM_df
        

        # It determines the lower and upper limits. It would make sense to support it by drawing a boxplot.
        # private function.
        def __outlier_values_treshold(self, dataset):
            Q1 = dataset.quantile(0.25)
            Q3 = dataset.quantile(0.75)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5*IQR
            upper_limit = Q3 + 1.5*IQR

            return lower_limit, upper_limit
        

        # Returns Outlier Values ​​Below Lower Bound
        # column_name: Type the name of the column for which we want to find the outlier values.
        # values_type: 1: Returns the outliers found in the dataset, 2: Returns outliers values index in the dataset
        def select_lower_outlier_values(self, dataset, column_name, values_type = 1):
            df = dataset[column_name]
            lower_limit, _   = self.__outlier_values_treshold(df)
            lower_outlier_tf = (df < lower_limit)
            if(values_type == 1):
                return df[lower_outlier_tf]
            elif(values_type == 2):
                return df[lower_outlier_tf].index
            else:
                return ValueError("Invalid Values Type: Takes a value between 1 and 2")
        

        # Returns Outlier Values ​​Below Upper Limit
        # column_name: Type the name of the column for which we want to find the outlier values. 
        # values_type: 1: Returns the outliers found in the dataset, 2: Returns outliers values index in the dataset
        def select_upper_outlier_values(self, dataset, column_name, values_type = 1):
            df = dataset[column_name]
            _, upper_limit   = self.__outlier_values_treshold(df)
            upper_outlier_tf = (df > upper_limit)
            if(values_type == 1):
                return df[upper_outlier_tf]
            elif(values_type == 2):
                return df[upper_outlier_tf].index
            else:
                return ValueError("Invalid Values Type: Takes a value between 1 and 2")


        # Returns values ​​with lower and upper bounds together.
        # column_name: Type the name of the column for which we want to find the outlier values.
        # values_type: 1: Returns the outliers found in the dataset, 2: Returns outliers values index in the dataset
        def select_all_outlier_values(self, dataset, column_name, values_type = 1):
            df = dataset[column_name]
            lower_limit, upper_limit   = self.__outlier_values_treshold(df)
            all_outlier_tf = (df < lower_limit) | (df > upper_limit)
            if(values_type == 1):
                return df[all_outlier_tf]
            elif(values_type == 2):
                return df[all_outlier_tf].index
            else:
                return ValueError("Invalid Values Type: Takes a value between 1 and 2")


        # Deleting outliers All outliers on a column of your choice are deleted.
        # column_name: Type the name of the column from which we want to delete outliers.
        def delete_all_outlier_values(self, dataset, column_name):
            df = dataset[column_name]
            lower_limit, upper_limit   = self.__outlier_values_treshold(df)
            df = pd.DataFrame(df)
            clear_df = df[~((df < (lower_limit)) | (df > upper_limit)).any(axis=1)]
            return clear_df


        # The mean fills outliers with the mean of the dataset
        # column_name: Type the name of the column for which we want to find the outlier values.
        def fill_mean_value_all_outlier_values(self, dataset, column_name):
            df = dataset[column_name]
            lower_limit, upper_limit   = self.__outlier_values_treshold(df)
            all_outlier_tf = (df < lower_limit) | (df > upper_limit)
            df[all_outlier_tf] = df.mean()
            return df, all_outlier_tf


        # Fills outliers with suppression method
        # column_name: Type the name of the column for which we want to find the outlier values.
        def fill_suppression_method_all_outlier_values(self, dataset, column_name):
            df = dataset[column_name]
            lower_limit, upper_limit   = self.__outlier_values_treshold(df)
            lower_outlier_tf = (df < lower_limit)
            upper_outlier_tf = (df > upper_limit)
            df[lower_outlier_tf] = lower_limit
            df[upper_outlier_tf] = upper_limit
            return df
        

        # Find and Delete Multivariate Outlier Values
        # Must not be nulls and must be a numeric dataset
        # lof_type: 1:Non-outlier values, 2:Outlier values
        def local_outlier_factor(self, dataset, n_neighbors = 20, contamination = 0.1, threshold_choice = 13, lof_type = 1):
            clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            clf = clf.fit(dataset)
            df_scores = clf.negative_outlier_factor_
        
            threshold_val = np.sort(df_scores)[threshold_choice]

            if(lof_type == 1):
                return dataset[df_scores > threshold_val]
            elif(lof_type == 2):
                return dataset[df_scores < threshold_val]
            else:
                return ValueError("Invalid Return Type: Takes a value between 1 and 2")
        

        # Filling the values ​​obtained with lof, with the suppression method
        def local_outlier_factor_suppression_method(self, dataset, n_neighbors = 20, contamination = 0.1, threshold_choice = 13):
            clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            clf = clf.fit(dataset)
            df_scores = clf.negative_outlier_factor_
        
            threshold_val   = np.sort(df_scores)[threshold_choice]
            suppression_val = dataset[df_scores == threshold_val]
            outlier_tf      = df_scores > threshold_val
            outlier_values  = dataset[~outlier_tf]

            res = outlier_values.to_records(index=False)
            res[:] = suppression_val.to_records(index=False)

            dataset[~outlier_tf] = pd.DataFrame(res, index = dataset[~outlier_tf].index)

            return dataset, dataset[~outlier_tf]
        
        
        # Scale numeric values. Converts values ​​between -3 and +3.
        # A numeric dataset must be sent.
        def numerical_variables_standartization(self, dataset):
            return preprocessing.scale(dataset)
        

        # It normalizes variables. Converts values ​​between 1 and 0.
        # A numeric dataset must be sent.
        def numerical_variables_normalization(self, dataset):
            return preprocessing.normalize(dataset)
        

        # Used to convert between 2 different values
        # init_val: Initial Value
        # end_val : End Value
        # A numeric dataset must be sent.
        def numerical_variables_transformation(self, dataset, init_val, end_val):
            scaler = preprocessing.MinMaxScaler(feature_range=(init_val, end_val))
            return scaler.fit_transform(dataset)
        

        # Converts numeric values to binary numbers
        # threshold: 5 is the default value
        # A numeric dataset must be sent.
        def numerical_variables_binarize(self, dataset, threshold = 5):
            binarizer = preprocessing.Binarizer(threshold=threshold).fit(dataset)
            return binarizer.transform(dataset)
        

        # Converts continuous variables (numeric values) to categorical variables
        # A numeric dataset must be sent.
        def numerical_variables_to_categorical_variables(self, dataset, n_bins=[3, 2, 2]):
            est = preprocessing.KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile').fit(dataset)
            return est.transform(dataset)


        # Converting classes of categorical variable to numeric data | Used when ordering is important
        # categoric_column: Write the name of the desired categorical column
        def ordinal_encoder(self, dataset, categorical_column):
            new_class = "ordinal_encoder"+"_"+categorical_column
            if(isinstance(dataset, pd.DataFrame)):
                categories = pd.Categorical(dataset[categorical_column], categories=list(dataset[categorical_column].unique()), ordered=True)
                values, _ = factorize(categories, sort=True)
                dataset[new_class] = values
                return dataset
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')

        
        # Converting classes of categorical variable to numeric data | Used when ordering is not important
        # categoric_column: Write the name of the desired categorical column
        def label_encoder(self, dataset, categorical_column):
            new_class = "label_encoder"+"_"+categorical_column
            if(isinstance(dataset, pd.DataFrame)):
                dataset[new_class] = preprocessing.LabelEncoder().fit_transform(dataset[categorical_column])
                return dataset
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')
        

        # A value selected from among the categorical variables is given a value of 1, and all remaining values ​​are given a 0.
        # categoric_column: Write the name of the desired categorical column
        # categoric_variable_name: Name of the categorical class to be given a value
        def where_encoder(self, dataset, categorical_column, categorical_class_name):
            new_class = "where_encoder"+"_"+categorical_column
            if(isinstance(dataset, pd.DataFrame)):
                dataset[new_class] = np.where(dataset[categorical_column].str.contains(categorical_class_name), 1, 0)
                return dataset
            else:
                return ValueError(f'Dataset Type Error: Dataset type is not DataFrame')


        # Categorical variables are converted to numeric variables and a column is created for each categorical class
        # categoric_column: Write the name of the desired categorical column 
        def onehot_encoder(self, dataset, categorical_column):
            df_one_hot = pd.get_dummies(dataset, columns=[categorical_column], prefix=[categorical_column])
            return df_one_hot      