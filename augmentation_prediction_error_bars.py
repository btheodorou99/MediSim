import pickle
import numpy as np
from sklearn import metrics
from tqdm import tqdm

NOISE = 1

def bootstrap_f1_from_confusion_matrix(confusion_matrix, n_iterations=100):
    """Calculate bootstrapped standard error for F1 score from a confusion matrix."""
    # Extract values from confusion matrix
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    total = int(tn + fp + fn + tp)
    
    # Create array of actual results (0s and 1s) based on confusion matrix
    actual = np.array([0]*int(tn + fp) + [1]*int(fn + tp))
    predicted = np.array([0]*int(tn) + [1]*int(fp) + [0]*int(fn) + [1]*int(tp))
    
    f1_scores = []
    # Bootstrap sampling
    for _ in tqdm(range(n_iterations), desc="Bootstrapping Iterations", leave=False):
        indices = np.random.randint(0, total, total)
        sample_actual = actual[indices]
        sample_predicted = predicted[indices]
        if len(np.unique(sample_actual)) < 2:  # Skip if all samples are from one class
            continue
        f1 = metrics.f1_score(sample_actual, sample_predicted)
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    f1 = metrics.f1_score(actual, predicted)  # Original F1 score
    
    # Calculate standard error
    standard_error = np.std(f1_scores, ddof=1)  # ddof=1 for sample standard deviation
    plus_minus = standard_error
    ci_lower = f1 - plus_minus
    ci_upper = f1 + plus_minus
    
    return ci_lower, ci_upper, plus_minus

# Load saved results and calculate confidence intervals
saved_results = pickle.load(open(f"results/augmentation_stats/prediction_stats_missing_{NOISE}.pkl", "rb"))

# Add confidence intervals to results
for task in tqdm(saved_results, desc="Tasks", total=len(saved_results), leave=False):
    if task == "Average":  # Skip the average entry
        continue
    for method in tqdm(saved_results[task], desc="Methods", total=len(saved_results[task]), leave=False):
        if "Confusion Matrix" in saved_results[task][method]:
            conf_matrix = saved_results[task][method]["Confusion Matrix"]
            ci_lower, ci_upper, plus_minus = bootstrap_f1_from_confusion_matrix(conf_matrix)
            saved_results[task][method]["F1_CI"] = (ci_lower, ci_upper)
            saved_results[task][method]["F1_PM"] = plus_minus

# Calculate average and standard error
methods = list(next(iter(saved_results.values())).keys())  # Get list of methods
tasks = [task for task in saved_results.keys() if task != "Average"]

# Sort tasks by prevalence (sum of confusion matrix)
task_prevalence = {}
for task in tasks:
    # Use Real Full Modality Data's confusion matrix for sorting
    conf_matrix = saved_results[task]["Real Full"]["Confusion Matrix"]
    task_prevalence[task] = np.sum(conf_matrix)
sorted_tasks = sorted(tasks, key=lambda x: task_prevalence[x])

for method in methods:
    if "Average" not in saved_results:
        saved_results["Average"] = {}
    
    # Sum confusion matrices across all tasks
    combined_conf_matrix = np.zeros((2, 2))
    for task in tasks:
        combined_conf_matrix += saved_results[task][method]["Confusion Matrix"]
    
    # Calculate bootstrapped CI for combined confusion matrix
    ci_lower, ci_upper, plus_minus = bootstrap_f1_from_confusion_matrix(combined_conf_matrix)
    
    # Calculate F1 score from combined confusion matrix
    tn, fp = combined_conf_matrix[0]
    fn, tp = combined_conf_matrix[1]
    actual = np.array([0]*int(tn + fp) + [1]*int(fn + tp))
    predicted = np.array([0]*int(tn) + [1]*int(fp) + [0]*int(fn) + [1]*int(tp))
    avg_f1 = metrics.f1_score(actual, predicted)
    
    saved_results["Average"][method] = {
        "F1 Score": avg_f1,
        "F1_PM": plus_minus,
        "F1_CI": (ci_lower, ci_upper),
        "Confusion Matrix": combined_conf_matrix
    }

# Save updated results
pickle.dump(saved_results, open(f"results/augmentation_stats/prediction_stats_missing_{NOISE}_with_ci.pkl", "wb"))

# Print example results for verification
for task in saved_results:
    print(f"\nTask: {task}")
    for method in saved_results[task]:
        f1 = saved_results[task][method]["F1 Score"]
        ci_lower, ci_upper = saved_results[task][method]["F1_CI"]
        plus_minus = saved_results[task][method]["F1_PM"]
        print(f"{method}: F1={f1:.3f} ± {plus_minus:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")

# Generate LaTeX table
method_order = [
    "Real Diagnosis Full",
    "Modality Added LR",
    "Modality Added CRA",
    "Modality Added NN",
    "Modality Added",
    "Real Full"
]
method_names = {
    "Real Diagnosis Full": "Real, Diagnosis-Only Data",
    "Modality Added LR": "Logistic Regression",
    "Modality Added CRA": "Cascaded Residual Autoencoder",
    "Modality Added NN": "Neural Network",
    "Modality Added": "\\method",
    "Real Full": "Real, Full Modality Data"
}

print("\n\nLaTeX Table:")
print("\\begin{table*}[]")
print("\\smaller")
print("\\centering")
print("\\caption{Complete Modality Enriched Downstream Performance}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{l" + "c"*len(sorted_tasks) + "c}")
print("\\toprule")

# Header row with rotated task names
header = "& " + " & ".join([f"\\rot{{{task}}}" for task in sorted_tasks]) + " & \\rot{Average} \\\\ \\midrule"
print(header)

for method in method_order:
    display_name = method_names[method]
    values = []
    pm_values = []
    
    for task in sorted_tasks + ["Average"]:
        f1 = saved_results[task][method]["F1 Score"]
        pm = saved_results[task][method]["F1_PM"]
        values.append(f"{f1:.3f}")
        pm_values.append(f"{pm:.3f}")
    
    # Create the row with makecell for each value
    row_values = [f"\\makecell{{{val} \\\\ $\\pm {pm}$}}" for val, pm in zip(values, pm_values)]
    row = f"{display_name} & " + " & ".join(row_values) + " \\\\"
    print(row)
    if method in ["Real Diagnosis Full", "Modality Added NN", "Modality Added"]:
        print("\\hline")

print("\\bottomrule")
print("\\end{tabular}}")
print("\\label{table:FullModalityEnrichment}")
print("\\end{table*}")
