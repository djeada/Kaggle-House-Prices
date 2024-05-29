# Kaggle House Prices
An exemplary solution for Kaggle's Data Science competition: House Prices - Advanced Regression Techniques.

This regression problem involves forecasting house prices based on various attributes (e.g., size).

![kaggle_5407_media_housesbanner](https://github.com/djeada/Kaggle-House-Prices/assets/37275728/0503dd9a-4379-4473-829a-072156140e34)

## Introduction

> Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

> With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

<a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">Read more.</a>

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/djeada/kaggle-house-prices.git
    ```

2. **Install `virtualenv`**:
    If `virtualenv` is not already installed, you can install it using:
    ```bash
    pip install virtualenv
    ```

3. **Set Up the Virtual Environment**:
    Open the terminal in the project directory and run the following commands to create and activate a virtual environment:
    ```bash
    cd kaggle-house-prices
    virtualenv env
    source env/bin/activate
    ```

4. **Install Dependencies**:
    Install the required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5. **Navigate to Source Directory**:
    Change to the source directory:
    ```bash
    cd src
    ```

6. **Run the Main Script**:
    Execute the main Python script:
    ```bash
    python3 main.py
    ```

## Dataset Description

This dataset contains detailed information about houses and their selling prices. Each row represents a house, and the columns provide various attributes related to the house's physical characteristics, construction details, and other relevant information. Here is a description of each column:

- **Id**: Unique identifier for each house.
- **MSSubClass**: Identifies the type of dwelling involved in the sale.
- **MSZoning**: Identifies the general zoning classification of the sale.
- **LotFrontage**: Linear feet of street connected to the property.
- **LotArea**: Lot size in square feet.
- **Street**: Type of road access to the property.
- **Alley**: Type of alley access to the property.
- **LotShape**: General shape of property.
- **LandContour**: Flatness of the property.
- **Utilities**: Type of utilities available.
- **LotConfig**: Lot configuration.
- **LandSlope**: Slope of property.
- **Neighborhood**: Physical locations within Ames city limits.
- **Condition1**: Proximity to various conditions.
- **Condition2**: Proximity to various conditions (if more than one is present).
- **BldgType**: Type of dwelling.
- **HouseStyle**: Style of dwelling.
- **OverallQual**: Overall material and finish quality.
- **OverallCond**: Overall condition rating.
- **YearBuilt**: Original construction date.
- **YearRemodAdd**: Remodel date.
- **RoofStyle**: Type of roof.
- **RoofMatl**: Roof material.
- **Exterior1st**: Exterior covering on house.
- **Exterior2nd**: Exterior covering on house (if more than one material).
- **MasVnrType**: Masonry veneer type.
- **MasVnrArea**: Masonry veneer area in square feet.
- **ExterQual**: Exterior material quality.
- **ExterCond**: Present condition of the material on the exterior.
- **Foundation**: Type of foundation.
- **BsmtQual**: Height of the basement.
- **BsmtCond**: General condition of the basement.
- **BsmtExposure**: Walkout or garden level basement walls.
- **BsmtFinType1**: Quality of basement finished area.
- **BsmtFinSF1**: Type 1 finished square feet.
- **BsmtFinType2**: Quality of second finished area (if present).
- **BsmtFinSF2**: Type 2 finished square feet.
- **BsmtUnfSF**: Unfinished square feet of basement area.
- **TotalBsmtSF**: Total square feet of basement area.
- **Heating**: Type of heating.
- **HeatingQC**: Heating quality and condition.
- **CentralAir**: Central air conditioning (Y = Yes, N = No).
- **Electrical**: Electrical system.
- **1stFlrSF**: First floor square feet.
- **2ndFlrSF**: Second floor square feet.
- **LowQualFinSF**: Low quality finished square feet (all floors).
- **GrLivArea**: Above grade (ground) living area square feet.
- **BsmtFullBath**: Basement full bathrooms.
- **BsmtHalfBath**: Basement half bathrooms.
- **FullBath**: Full bathrooms above grade.
- **HalfBath**: Half baths above grade.
- **BedroomAbvGr**: Number of bedrooms above basement level.
- **KitchenAbvGr**: Number of kitchens.
- **KitchenQual**: Kitchen quality.
- **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms).
- **Functional**: Home functionality rating.
- **Fireplaces**: Number of fireplaces.
- **FireplaceQu**: Fireplace quality.
- **GarageType**: Garage location.
- **GarageYrBlt**: Year garage was built.
- **GarageFinish**: Interior finish of the garage.
- **GarageCars**: Size of garage in car capacity.
- **GarageArea**: Size of garage in square feet.
- **GarageQual**: Garage quality.
- **GarageCond**: Garage condition.
- **PavedDrive**: Paved driveway (Y = Yes, N = No, P = Partial).
- **WoodDeckSF**: Wood deck area in square feet.
- **OpenPorchSF**: Open porch area in square feet.
- **EnclosedPorch**: Enclosed porch area in square feet.
- **3SsnPorch**: Three season porch area in square feet.
- **ScreenPorch**: Screen porch area in square feet.
- **PoolArea**: Pool area in square feet.
- **PoolQC**: Pool quality.
- **Fence**: Fence quality.
- **MiscFeature**: Miscellaneous feature not covered in other categories.
- **MiscVal**: Value of miscellaneous feature.
- **MoSold**: Month sold.
- **YrSold**: Year sold.
- **SaleType**: Type of sale.
- **SaleCondition**: Condition of sale.
- **SalePrice**: Sale price of the house.

### Sample Data

Here are the first five entries in the dataset:

| Id  | MSSubClass | MSZoning | LotFrontage | LotArea | Street | Alley | LotShape | LandContour | Utilities | LotConfig | LandSlope | Neighborhood | Condition1 | Condition2 | BldgType | HouseStyle | OverallQual | OverallCond | YearBuilt | YearRemodAdd | RoofStyle | RoofMatl | Exterior1st | Exterior2nd | MasVnrType | MasVnrArea | ExterQual | ExterCond | Foundation | BsmtQual | BsmtCond | BsmtExposure | BsmtFinType1 | BsmtFinSF1 | BsmtFinType2 | BsmtFinSF2 | BsmtUnfSF | TotalBsmtSF | Heating | HeatingQC | CentralAir | Electrical | 1stFlrSF | 2ndFlrSF | LowQualFinSF | GrLivArea | BsmtFullBath | BsmtHalfBath | FullBath | HalfBath | BedroomAbvGr | KitchenAbvGr | KitchenQual | TotRmsAbvGrd | Functional | Fireplaces | FireplaceQu | GarageType | GarageYrBlt | GarageFinish | GarageCars | GarageArea | GarageQual | GarageCond | PavedDrive | WoodDeckSF | OpenPorchSF | EnclosedPorch | 3SsnPorch | ScreenPorch | PoolArea | PoolQC | Fence | MiscFeature | MiscVal | MoSold | YrSold | SaleType | SaleCondition | SalePrice |
|----|------------|----------|-------------|---------|--------|-------|----------|-------------|-----------|-----------|-----------|--------------|------------|------------|---------|------------|------------|------------|-----------|--------------|-----------|----------|-------------|-------------|------------|------------|-----------|-----------|------------|----------|----------|--------------|-------------|------------|--------------|------------|-----------|-------------|---------|----------|------------|------------|----------|----------|------------|----------|-------------|-------------|---------|----------|--------------|--------------|------------|---------------|-----------|------------|-------------|-----------|-------------|--------------|-------------|-----------|-------------|------------|------------|-------------|------------|------------|----------|-------|-------|-------------|--------|-------|-------|----------|-------------|----------|
| 1  | 60         | RL       | 65          | 8450    | Pave   | NA    | Reg      | Lvl         | AllPub    | Inside    | Gtl       | CollgCr      | Norm       | Norm       | 1Fam    | 2Story     | 7          | 5          | 2003      | 2003         | Gable     | CompShg  | VinylSd     | VinylSd     | BrkFace    | 196        | Gd        | TA        | PConc      | Gd       | TA       | No           | GLQ         | 706        | Unf          | 0          | 150        | 856         | GasA    | Ex       | Y          | SBrkr      | 856      | 854      | 0           | 1710     | 1           | 0           | 2        | 1        | 3            | 1            | Gd         | 8             | Typ        | 0          | NA          | Attchd    | 2003        | RFn          | 2           | 548         | TA        | TA          | Y          | 0          | 61         | 0            | 0          | 0          | NA       | NA    | NA          | 0            | 2       | 2008  | WD       | Normal      | 208500    |
| 2  | 20         | RL       | 80          | 9600    | Pave   | NA    | Reg      | Lvl         | AllPub    | FR2       | Gtl       | Veenker      | Feedr      | Norm       | 1Fam    | 1Story     | 6          | 8          | 1976      | 1976         | Gable     | CompShg  | MetalSd     | MetalSd     | None       | 0          | TA        | TA        | CBlock     | Gd       | TA       | Gd           | ALQ         | 978        | Unf          | 0          | 284        | 1262        | GasA    | Ex       | Y          | SBrkr      | 1262     | 0        | 0           | 1262     | 0           | 1           | 2        | 0        | 3            | 1            | TA         | 6             | Typ        | 1          | TA          | Attchd    | 1976        | RFn          | 2           | 460         | TA        | TA          | Y          | 298        | 0          | 0            | 0          | 0          | NA       | NA    | NA          | 0            | 5       | 2007  | WD       | Normal      | 181500    |
| 3  | 60         | RL       | 68          | 11250   | Pave   | NA    | IR1      | Lvl         | AllPub    | Inside    | Gtl       | CollgCr      | Norm       | Norm       | 1Fam    | 2Story     | 7          | 5          | 2001      | 2002         | Gable     | CompShg  | VinylSd     | VinylSd     | BrkFace    | 162        | Gd        | TA        | PConc      | Gd       | TA       | Mn           | GLQ         | 486        | Unf          | 0          | 434        | 920         | GasA    | Ex       | Y          | SBrkr      | 920      | 866      | 0           | 1786     | 1           | 0           | 2        | 1        | 3            | 1            | Gd         | 6             | Typ        | 1          | TA          | Attchd    | 2001        | RFn          | 2           | 608         | TA        | TA          | Y          | 0          | 42         | 0            | 0          | 0          | NA       | NA    | NA          | 0            | 9       | 2008  | WD       | Normal      | 223500    |
| 4  | 70         | RL       | 60          | 9550    | Pave   | NA    | IR1      | Lvl         | AllPub    | Corner    | Gtl       | Crawfor      | Norm       | Norm       | 1Fam    | 2Story     | 7          | 5          | 1915      | 1970         | Gable     | CompShg  | Wd Sdng     | Wd Shng     | None       | 0          | TA        | TA        | BrkTil     | TA       | Gd       | No           | ALQ         | 216        | Unf          | 0          | 540        | 756         | GasA    | Gd       | Y          | SBrkr      | 961      | 756      | 0           | 1717     | 1           | 0           | 1        | 0        | 3            | 1            | Gd         | 7             | Typ        | 1          | Gd          | Detchd    | 1998        | Unf          | 3           | 642         | TA        | TA          | Y          | 0          | 35         | 272          | 0          | 0          | NA       | NA    | NA          | 0            | 2       | 2006  | WD       | Abnorml     | 140000    |
| 5  | 60         | RL       | 84          | 14260   | Pave   | NA    | IR1      | Lvl         | AllPub    | FR2       | Gtl       | NoRidge      | Norm       | Norm       | 1Fam    | 2Story     | 8          | 5          | 2000      | 2000         | Gable     | CompShg  | VinylSd     | VinylSd     | BrkFace    | 350        | Gd        | TA        | PConc      | Gd       | TA       | Av           | GLQ         | 655        | Unf          | 0          | 490        | 1145        | GasA    | Ex       | Y          | SBrkr      | 1145     | 1053     | 0           | 2198     | 1           | 0           | 2        | 1        | 4            | 1            | Gd         | 9             | Typ        | 1          | TA          | Attchd    | 2000        | RFn          | 3           | 836         | TA        | TA          | Y          | 192        | 84         | 0            | 0          | 0          | NA       | NA    | NA          | 0            | 12      | 2008  | WD       | Normal      | 250000    |

## Potential Analyses

This dataset allows for a variety of analyses, providing insights into different aspects of house characteristics and their selling prices. Here are some of the key analyses that can be performed:

- **Price Analysis**:
  - Investigate the factors influencing house prices. This involves analyzing how different variables such as lot area, year built, neighborhood, and overall quality affect the selling price.
  - For example, we can determine if houses in certain neighborhoods have higher selling prices compared to others, or if newer houses tend to sell for more.
  - We can also explore the impact of specific features like the presence of a garage, pool, or fireplace on the house price.

- **Demographic Analysis**:
  - Examine the distribution of houses by various characteristics such as overall quality, year built, and lot area.
  - Understand the range of house prices and identify which features are most common among the highest-priced houses.

- **Economic Analysis**:
  - Analyze the relationship between various economic factors and house prices, such as lot area, overall quality, and neighborhood.
  - Determine if larger lots or higher quality materials lead to higher selling prices.
  - Explore the economic differences between houses in different neighborhoods.

- **Family Analysis**:
  - Explore the impact of family-related features (e.g., number of bedrooms, number of bathrooms) on house prices.
  - Investigate whether houses with more bedrooms and bathrooms tend to have higher selling prices.

- **Text Analysis**:
  - Analyze patterns in categorical variables such as house style, roof style, and exterior covering.
  - Identify common features and their impact on house prices.

- **Geographic Analysis**:
  - Study the impact of location (neighborhood) on house prices.
  - Analyze the distribution of house prices across different neighborhoods and identify areas with the highest and lowest prices.


## Project Workflow and Analysis

### Data Preprocessing and Cleaning

1. **Setting Up Paths and Imports**:
    - Imported necessary libraries and modules for data handling, model training, and evaluation.
    - Defined paths to the training and test dataset files.

2. **Preprocessing Steps**:
    - Created a directory for output files to save processed data and models.

3. **Cleaning the Dataset**:
    - Read the raw training and test datasets from CSV files.
    - Applied data cleaning steps:
        - **EncodeCategoricalVariablesFilter**: Encoded categorical variables to numerical values.
        - **FillMissingValuesFilter**: Filled in missing values in the dataset.
    - Cleaned datasets were prepared for further processing.

4. **Splitting the Dataset**:
    - Separated the cleaned training dataset into features (`x_dataset`) and labels (`y_dataset`).
    - Split the data into training and testing sets, saving the split datasets for future use.

### Model Training

5. **Initializing Models**:
    - Defined a list of model types to be trained:
        - `LinearRegression`
        - `MultilayerPerceptron`
        - `RandomForest`
        - `GradientBoost`
        - `Lasso`
    - Created an empty list to store the trained models.

6. **Training Models**:
    - Iterated through each model type, initialized, and trained each model using the training dataset (`train_x` and `train_y`).
    - Saved each trained model to the output directory using joblib.
    - Stored each trained model in the `models` list.

### Model Testing and Evaluation

7. **Evaluating Models**:
    - Initialized an empty list `scores` to store evaluation metrics.
    - For each trained model, generated predictions on the testing dataset (`test_x`).
    - Calculated evaluation metrics:
        - `r2_score`: R-squared score to evaluate the model's performance.
        - `rmse`: Root Mean Squared Error to measure the differences between predicted and actual values.
        - `nrmse`: Normalized Root Mean Squared Error to provide a normalized measure of the error.
    - Stored these metrics in the `scores` list and printed the results.

8. **Identifying the Best Model**:
    - Determined the best model based on the highest `r2_score`.
    - Printed the name of the best-performing model.

### Postprocessing and Predictions

9. **Creating Prediction Tables**:
    - For each model, created a prediction table combining the test features, actual values, and predicted values.
    - Saved these tables for each model to the output directory.

10. **Using the Best Model for Final Predictions**:
    - Selected the best model based on evaluation metrics.
    - Used the best model to predict values on the cleaned test dataset.
    - Prepared a final DataFrame with the predictions and saved it to a CSV file named `predictions.csv` in the output directory.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
