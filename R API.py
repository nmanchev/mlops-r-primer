# Databricks notebook source
# MAGIC %md
# MAGIC # Calling an R model from MLflow

# COMMAND ----------

clusterId = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
api_url = workspace_url + "/api/1.2"
context_name = "my_execution_context" # Just a name to distinguish the new execution context we'll create 

print("cluster_id        :", clusterId)
print("workspace_url     :", workspace_url)
print("api_url           :", api_url)
print("execution_context :", context_name)

# COMMAND ----------

# MAGIC %md
# MAGIC You need to generate a ppersonal access token, which can be used for secure authentication to the Databricks API instead of passwords. For more details see the [Databricks Documentation](https://learn.microsoft.com/en-gb/azure/databricks/dev-tools/auth/). Once you have the token, input it in the `access_token` widget. 

# COMMAND ----------

# Get the value of the access_token widget
access_token = dbutils.widgets.get("access_token")

# Check if access_token is empty
if not access_token:
    raise ValueError("access_token is empty. Stopping execution.")

# COMMAND ----------

import requests
import json

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# Set the request body
body = {
    "language": "r",
    "clusterId": clusterId,
    "name": context_name
}

# Execute the API call to create the context
response = requests.post(f"{api_url}/contexts/create", headers=headers, data=json.dumps(body))

# Check the response status code
if response.status_code == 200:
    context_id = json.loads(response.text)["id"]
    print(f"Execution context created with ID: {context_id}")
else:
    print(f"Error creating execution context: {response.text}")

# COMMAND ----------

import time

# This is the function that does the actual execution
# It doesn't handle errors or commands that won't terminate properly. It's just an MVP

def execute(clusterId, contextId, command):
  # Set the request body
  body = {
      "language": "r",
      "clusterId": clusterId,
      "command": command,
      "contextId": contextId
  }

  # Execute the API call
  response = requests.post(f"{api_url}/commands/execute", headers=headers, data=json.dumps(body))

  # Check the response status code
  if response.status_code == 200:
      run_id = json.loads(response.text)["id"]
      print(f"Command submitted with run ID: {run_id}")
  else:
      print(f"Error executing command: {response.text}")
  
  if run_id is not None:
    
    # Get the status of the command
    body.pop("command")
    body["commandId"] = run_id
    status = "Running"

    while status == "Running" or status == "Queued":
      print("Command running...")
      time.sleep(1)
      # Execute the API call to get the command info
      response = requests.get(f"{api_url}/commands/status", headers=headers, params=body)

      # Check the response status code
      if response.status_code == 200:
          command_info = json.loads(response.text)
          status = command_info["status"]
      else:
          print(f"Error retrieving command info: {response.text}")

  return command_info

# COMMAND ----------

# Test that we can execute arbitrary R code in the context

code = """
add_two_numbers <- function(first_num, sec_num) {
  return(first_num + sec_num)
}
    
add_two_numbers(11, 10)
add_two_numbers(8, 20)
"""

execute(clusterId, context_id, code)["results"]["data"]

# COMMAND ----------

# Now let's pull the persisted model from MLflow and score some data

model_uri = dbutils.widgets.get("model_uri")

if not model_uri:
    print("model_uri not specified. Will not test calling an MLflow model.")
else:
  code = f'''library(mlflow)
  # Create some test data
  column_names <- c("fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar", "chlorides", "free.sulfur.dioxide", "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol")
  data_matrix <- matrix(data =  c(7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,
                                  7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,
                                  7.8,0.76,0.04,2.3,0.092,15,54,0.997,3.26,0.65,9.8,
                                  11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8,
                                  7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4),
                                  nrow = 5, ncol = length(column_names))
  colnames(data_matrix) <- column_names
  df <- as.data.frame(data_matrix)

  # Connect to MLflow and pull the model. Inject the Databricks host & token for MLflow authentication
  Sys.setenv(DATABRICKS_HOST = "{workspace_url}", "DATABRICKS_TOKEN" = "{access_token}")

  best_model <- mlflow_load_model(model_uri = "{model_uri}")

  # Score the data
  predictions <- data.frame(mlflow_predict(best_model, data = df))          
  toString(predictions[[1]])
  '''
  # Run the R code and print the predictions
  predictions = execute(clusterId, context_id, code)
  print(predictions["results"]["data"])
