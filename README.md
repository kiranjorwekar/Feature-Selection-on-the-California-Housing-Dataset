
![Logo](https://github.com/kiranjorwekar/Linear-Regression-using-California-Dataset/blob/main/Linear_Regression_using_CalHousingData.jpg)


# Project Title - "Feature Selection & Dimensionality on the California Housing Dataset".

An assignement to complete the feature selection & dimenstionality on the California Housing Dataset.

Basically, we can have select features based on the basis of univariate statistics with the given dataset, meaning in short we can consider the dataset columns and relation with each other for determining any inferences out of it.


## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Model: [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: R2 score is - 0.5923464

## Dataset Details

This dataset was obtained from the StatLib repository ([Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the 'sklearn.datasets.fetch_california_housing' function.

- [California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- 20640 samples
- 8 Input Features: 
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
- Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)

## Evaluation and Results
![alt text](https://github.com/123ofai/Demo-Project-Repo/blob/main/results/test.png)

As you can see from the above image, the model has signifcant amount of error in <x, y, z regions>

| Metric        | Value         |
| ------------- | ------------- |
| R2 Score      | 0.6117200789  |
| MSE           | 0.530587150  |
 

The above quant results show that  R2 Score is 0.611 & MSE = 0.530
## Key Takeaways

Based on the  feature selections and finding dimensionality reduction, we need to first find the relation between these column datasets. There 2 methids so far we have studied here as follows - 
a. Filter Based Methods - choose the most valuable information from a large dataset. It is required some pre-processing steps with context of the columns relations within that dataset.
		
		
b. Wrapper Based Methods - It consists of - 
		1. Trian a model on different subset of features and select most accurate or closer one's.
		2. Supervised - means a defined dataset, number , information to measure.
		3. Computational Expensive - dependent on number of experiments & associated sample complexity.
		
c. Recursive Feature Elimination (RFE). - It consists of - 
		1. Given an estimator that assigns weights/coeffecients to the features (eg: linear model),
		2. It starts out by training the model on all the features.
		3. Then, recursively, removes the least important features, and re-trains the model.
		4. This process is repeated until we have the desired number of features.


## How to Run

The code is built on Google Colab on an iPython Notebook. 

```bash
Simply download the repository, upload the notebook and dataset on colab, and hit play!
```


## Roadmap

Will continue learning and evolving further model with different dataset. Will provide the details for further learnings. Thanks!


## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## FAQ

#### How does the linear regression model work?

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the independent and dependent variables. The goal of linear regression is to find the best-fitting straight line that describes the relationship between the variables.
It will be analyzed using - 
1. Data Collections
2. Scatterplot
3. Model Representation - which means relationship between Independant variable (X) and the dependent variable (Y)

#### How do you train the model on a new dataset?

Training a linear regression model for a datataset based on -
1. Data Processing - PRepare the new dataset for traning by perfromaing necessary pre-processing steps like -  handling missing values, scaling or normalizing features and spliting the data into training and testing sets.
2. Model Selection - Decide whether to use Linear Regression Model or multiple regression model(means multiple independent variables). This is completely dependent on the nature of data and releationship between variables.
3. Model traning - Estimating the coefficients that best fit the data using a method of least squares.
4. Model Evalutions
5. Fine-tuning of fetures .. etc 

#### What is the California Housing Dataset?

The California Housing Dataset is a popular dataset often used in machine learning and statistical analysis tutorials and research. It contains data on housing prices and various factors that may influence housing prices in different neighborhoods in California.

## Acknowledgements

All the links, blogs, videos, papers you referred to/took inspiration from for building this project. 

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


## Contact

If you have any feedback/are interested in collaborating, please reach out to me at kiran2777@rediffmail.com


## License



