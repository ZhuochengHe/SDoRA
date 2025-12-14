import pandas as pd

# 1. Load the CSV file
df = pd.read_csv("results.csv")

# # Quick sanity check of the columns
# # Expected columns: model, checkpoint, dataset, correct, total, accuracy
# print(df.head())

# 2. Create an experiment identifier combining model and checkpoint
#    This lets us treat each checkpoint as a separate model
df["exp_id"] = df["model"] + " | " + df["checkpoint"]

# 3. Pivot table:
#    - rows:    exp_id (model + checkpoint)
#    - columns: dataset name
#    - values:  accuracy
#    If there are multiple rows for the same (exp_id, dataset),
#    we take the mean accuracy.
pivot = df.pivot_table(
    index="exp_id",
    columns="dataset",
    values="accuracy",
    aggfunc="mean",
)

# 4. Compute the average accuracy across all datasets for each exp_id
pivot["avg_accuracy"] = pivot.mean(axis=1)

# 5. Sort experiments by average accuracy (descending)
pivot = pivot.sort_values("avg_accuracy", ascending=False)

# 6. Print the result nicely
pd.set_option("display.max_columns", None)  # show all datasets
pd.set_option("display.width", 160)

print("\n=== Accuracy per model/checkpoint on each dataset and the overall average ===")
# 6. Save the summary to a new CSV file
output_path = "results_summary_by_exp.csv"
pivot.to_csv(output_path)

print(f"Saved summary to {output_path}")