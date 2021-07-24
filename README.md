# kaggle-house-prices
Exemplary solution to Kaggle's Data Science competition: House Prices - Advanced Regression Techniques

<h1>Introduction</h1>

> Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

> With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

<a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">Read more.</a>

<h1>Installation</h1>

Follow the steps:

- Download this repository: 
 
 ```bash 
 git clone https://github.com/djeada/kaggle-titanic.git
 ```
 
- Install <i>virtualenv</i> (if it's not already installed).
- Open the terminal from the project directory and run the following commands:

```bash
virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
cd src
python3 main.py
```

<h1>Cleaning Data</h1>

<h1>General Statistics</h1>
It is always a good idea to take a look at some basic statistics before using any machine learning. Some trends in the data might be obvious and could help us later to understand the predictions of different machine learning algorithms.

![alt text](https://github.com/djeada/kaggle-house-prices/blob/main/resources/number_of_houses_vs_house_prices.png)

![alt text](https://github.com/djeada/kaggle-house-prices/blob/main/resources/numeric_features_correlation.png)

![alt text](https://github.com/djeada/kaggle-house-prices/blob/main/resources/sale_price_vs_GarageArea.png)

![alt text](https://github.com/djeada/kaggle-house-prices/blob/main/resources/sale_price_vs_GarageCars.png)

![alt text](https://github.com/djeada/kaggle-house-prices/blob/main/resources/sale_price_vs_GrLivArea.png)

![alt text](https://github.com/djeada/kaggle-house-prices/blob/main/resources/sale_price_vs_OverallQual.png)

![alt text](https://github.com/djeada/kaggle-house-prices/blob/main/resources/sale_price_vs_TotalBsmtSF.png)

<h1>Results</h1>

![alt text](https://github.com/djeada/kaggle-house-prices/blob/main/resources/model_comparison.png)

A high R-square of more than 60% (0.60) indicates that values can be accurately estimated with some precision.

TODO: graph actual values vs predicted: y - frequency, x - value.
