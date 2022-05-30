


# Matplotlib

  1. Basics:
     1. Plot 2 graphs on top of each other and set the x limits
        1. plt.hist(predicted_weights);
        2. plt.hist(orders2.review_score);
        3. plt.xlim([0,5]);


# Seaborn
  import seaborn as sns

  1. Histplot with kernal distribution estimator:
     1. sns.histplot(df.Series, kde=True)

  2. Regplot of a df
     1. sns.regplot(x='review_score', y='delay_vs_expected', data=orders2, ci=95)
        1. with confidence interval
        2. uses df series names as inputs
        3. specify the x,y, and data
     2. to select a sample, use:
        1. sns.regplot(x='review_score', y='delay_vs_expected', data=orders2.sample(10, random_state=6), ci=95)
  3. Scatter plot of a 2-d
     1. sns.scatterplot(data=X, x=?, y=?);
  4. CountPlot to see counts of the classes in a y_train, for example:
     1. sns.countplot(x=y_train)

  5. Example EDA = Exploratory Data Analysis:
     1. NB this uses sellers a pandas DF
     2. matplotlib as plt
     3. seaborn as sns
   ```python
   plt.figure(figsize=(15,11))
   for (i, col) in enumerate(sellers.describe().columns):#["wait_time", "delay_to_carrier", "avg_review_score", "n_orders", "quantity", "price"]):
    plt.subplot(3,4,i+1)
    sns.histplot(sellers[col], kde=False, stat='density', discrete=[True,None][col in ['share_of_one_stars','share_of_five_stars','sales']]);
    ```

# Plotly

  1. import plotly.express as px
  2. **Scatter plot**, showing circles of difference sizes and colours

      ```python
      import plotly.express as px
      fig = px.scatter(data_frame = sellers[sellers['review_score'] < 4],
          x="wait_time",
          y="delay_to_carrier",
          size="sales",
          color="review_score",
          size_max = 60,
          opacity = 0.5
      )
      fig.show()
      ```

# PANDAS
  1. boxplots
    df.col.boxplot()
    df.boxplot()
  2. histograms
    df.hist()
    df.col.hist()
