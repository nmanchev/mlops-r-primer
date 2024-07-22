# Databricks notebook source
# MAGIC %md
# MAGIC # Training and loggign an R model using MLflow
# MAGIC
# MAGIC In this notebook we train a simple model using the freely available [wine quality]("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality) dataset form the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/). 
# MAGIC
# MAGIC We train 3 separate models and log them in MLflow. We select the best performing model based on \\(R^2\\).

# COMMAND ----------

install.packages("carrier")
install.packages("curl")

# COMMAND ----------

library(mlflow)
library(httr)
library(SparkR)
library(glmnet)
library(carrier)

# COMMAND ----------

reds <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")
whites <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";")
 
wine_quality <- rbind(reds, whites)
 
head(wine_quality)

# COMMAND ----------

## Create a function to train based on different parameters
train_wine_quality <- function(data, alpha, lambda, model_name = "model") {
 
# Split the data into training and test sets. (0.75, 0.25) split.
sampled <- base::sample(1:nrow(data), 0.75 * nrow(data))
train <- data[sampled, ]
test <- data[-sampled, ]
 
# The predicted column is "quality" which is a scalar from [3, 9]
train_x <- as.matrix(train[, !(names(train) == "quality")])
test_x <- as.matrix(test[, !(names(train) == "quality")])
train_y <- train[, "quality"]
test_y <- test[, "quality"]
 
## Define the parameters used in each MLflow run
alpha <- mlflow_param("alpha", alpha, "numeric")
lambda <- mlflow_param("lambda", lambda, "numeric")

with(mlflow_start_run(), {
    model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family= "gaussian", standardize = FALSE)
    l1se <- cv.glmnet(train_x, train_y, alpha = alpha)$lambda.1se
    predictor <- carrier::crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model, s = l1se)
  
    predicted <- predictor(test_x)
 
    rmse <- sqrt(mean((predicted - test_y) ^ 2))
    mae <- mean(abs(predicted - test_y))
    r2 <- as.numeric(cor(predicted, test_y) ^ 2)
 
    message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
    message("  RMSE: ", rmse)
    message("  MAE: ", mae)
    message("  R2: ", mean(r2, na.rm = TRUE))
 
    ## Log the parameters associated with this run
    mlflow_log_param("alpha", alpha)
    mlflow_log_param("lambda", lambda)
  
    ## Log metrics we define from this run
    mlflow_log_metric("rmse", rmse)
    mlflow_log_metric("r2", mean(r2, na.rm = TRUE))
    mlflow_log_metric("mae", mae)
  
    # Save plot to disk
    png(filename = "ElasticNet-CrossValidation.png")
    plot(cv.glmnet(train_x, train_y, alpha = alpha), label = TRUE)
    dev.off()
  
    ## Log that plot as an artifact
    mlflow_log_artifact("ElasticNet-CrossValidation.png")

    mlflow_log_model(predictor, model_name)
  
})

}

# COMMAND ----------

set.seed(1234)
 
model_name = "wine-model"
 
## Run 1
train_wine_quality(data = wine_quality, alpha = 0.03, lambda = 0.98, model_name)
 
## Run 2
train_wine_quality(data = wine_quality, alpha = 0.14, lambda = 0.4, model_name)
 
## Run 3
train_wine_quality(data = wine_quality, alpha = 0.20, lambda = 0.99, model_name)

# COMMAND ----------

# Search runs for best r^2
runs <- mlflow_search_runs(order_by = "metrics.r2 DESC")

# Assuming the first run is the best model according to r^2
best_run_id <- runs$run_id[1]

# Construct model URI
model_uri <- paste0("runs:/", best_run_id, "/model")

# Construct model URI
message(c("run_id        : ", best_run_id))
message(c("experiment_id : ", runs$experiment_id[1]))

model_uri <- paste(runs$artifact_uri[1], model_name, sep = "/")
message(c("Model URI.    : ", model_uri, "\n"))

# COMMAND ----------

## Load the model
best_model <- mlflow_load_model(model_uri = model_uri)
 
## Generate prediction on 5 rows of data 
predictions <- data.frame(mlflow_predict(best_model, data = wine_quality[1:5, !(names(wine_quality) == "quality")]))
                          
names(predictions) <- "wine_quality_pred"
 
## Take a look
display(predictions)
