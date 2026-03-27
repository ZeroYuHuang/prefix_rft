import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration Area ---

# 1. Set the path to your root directory
ROOT_DIR = 'rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_7_clip_0.2_25-06-21-Jun-06-1750458680'

# 2. Define the steps to analyze
START_STEP = 128
END_STEP = 256
ACCURACY_STEPS = [128, 160, 192, 224]

# 3. Define filenames
FORWARD_FILE = 'forward.json'
GEN_FILE_PREFIX = 'gen_t1.0'
GEN_FILE_SUFFIX = '.json'

# --- User-Provided Calculation Function ---

def high_entropy_token_loss(example):
    loss = torch.tensor(example["loss"])
    token_entropy = torch.tensor(example["token_entropy"])
    loss_mask = torch.tensor(example["loss_mask"])
    response_losses = torch.masked_select(loss, loss_mask.bool())
    response_length = token_entropy.shape[0]
    k = int(response_length * 0.2)
    k = max(1, k) 
    _, top_k_indices = torch.topk(token_entropy, k)
    top_k_losses = torch.gather(response_losses, 0, top_k_indices)
    example['high_entropy_loss'] = top_k_losses.mean()
    return example

# --- Core Script Logic ---

def find_gen_file(directory: str) -> str | None:
    """Finds the generation result JSON file in the specified directory."""
    for fname in os.listdir(directory):
        if fname.startswith(GEN_FILE_PREFIX) and fname.endswith(GEN_FILE_SUFFIX):
            return os.path.join(directory, fname)
    return None


def load_and_process_high_entropy_loss(filepath: str) -> dict[str, float]:
    """
    Loads the forward data, applies the high_entropy_token_loss function,
    and returns a dictionary mapping {uid: mean_high_entropy_loss}.
    """
    print(f"  - Processing high entropy loss from: {filepath}")
    # try:
    dset = load_dataset('json', data_files=filepath, split='train', cache_dir='./.cache')
    
    # Apply the user's function to each row
    print("    Calculating high entropy loss for each token sequence...")
    dset = dset.map(high_entropy_token_loss, num_proc=os.cpu_count())
    
    print("    Aggregating results by UID...")
    df = dset.to_pandas()
    
    # Group by uid and get the mean of our newly calculated metric
    uid_hel = df.groupby('uid')['high_entropy_loss'].mean()
    return uid_hel.to_dict()
    # except Exception as e:
    #     print(f"    Error processing high entropy loss file {filepath}: {e}")
    #     return {}

def load_and_process_accuracies(filepath: str) -> dict[str, float]:
    # This function remains the same as before
    print(f"  - Loading and processing accuracies from: {filepath}")
    try:
        dset = load_dataset('json', data_files=filepath, split='train', cache_dir='./.cache')
        def calculate_accuracy(example):
            corr_list = example.get('analysis_corr', [])
            return {'accuracy': np.mean(corr_list) if corr_list else 0.0}
        dset = dset.map(calculate_accuracy, num_proc=os.cpu_count())
        return dict(zip(dset['uid'], dset['accuracy']))
    except Exception as e:
        print(f"    Error processing accuracy file {filepath}: {e}")
        return {}


def main():
    """Main execution function"""
    print("🚀 Starting Analysis: High Entropy Loss vs. Accuracy...")

    # --- 1. Data Loading ---
    start_path = os.path.join(ROOT_DIR, f'global_step{START_STEP}', FORWARD_FILE)
    end_path = os.path.join(ROOT_DIR, f'global_step{END_STEP}', FORWARD_FILE)
    
    start_hel_map = load_and_process_high_entropy_loss(start_path)
    end_hel_map = load_and_process_high_entropy_loss(end_path)

    all_accuracies_map = {}
    for step in ACCURACY_STEPS:
        print(f"\nProcessing Step {step} for accuracy...")
        step_dir = os.path.join(ROOT_DIR, f'global_step{step}')
        gen_file_path = find_gen_file(step_dir)
        if gen_file_path:
            all_accuracies_map[step] = load_and_process_accuracies(gen_file_path)
        else:
            print(f"  - Could not find generation file in {step_dir}. Skipping.")

    # --- 2. Data Alignment and Calculation ---
    print("\nAligning data and performing calculations...")
    uids_in_hel = set(start_hel_map.keys()) & set(end_hel_map.keys())
    valid_accuracy_maps = [set(acc_map.keys()) for acc_map in all_accuracies_map.values() if acc_map]
    if not valid_accuracy_maps:
        print("❌ Error: No valid accuracy data could be loaded. Aborting.")
        return
    uids_in_accuracies = set.intersection(*valid_accuracy_maps)
    common_uids = sorted(list(uids_in_hel & uids_in_accuracies))
    
    print(f"Found {len(common_uids)} common UIDs for analysis.")
    hel_changes = [] # High Entropy Loss changes
    mean_accuracies = []
    for uid in tqdm(common_uids, desc="Calculating final metrics"):
        hel_change = end_hel_map[uid] - start_hel_map[uid]
        hel_changes.append(hel_change)
        accs_for_uid = [all_accuracies_map[step][uid] for step in ACCURACY_STEPS if step in all_accuracies_map and uid in all_accuracies_map[step]]
        mean_accuracies.append(np.mean(accs_for_uid) if accs_for_uid else 0)

    # --- 3. Outlier Removal ---
    print("\nFiltering outliers from High Entropy Loss changes...")
    hel_changes_np = np.array(hel_changes)
    mean_accuracies_np = np.array(mean_accuracies)
    q1 = np.percentile(hel_changes_np, 25)
    q3 = np.percentile(hel_changes_np, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    non_outlier_mask = (hel_changes_np >= lower_bound) & (hel_changes_np <= upper_bound)
    filtered_hel_changes = hel_changes_np[non_outlier_mask]
    filtered_accuracies = mean_accuracies_np[non_outlier_mask]
    print(f"Removed {len(hel_changes_np) - len(filtered_hel_changes)} outliers.")

    # --- 4. Trend Line Calculation using LOWESS ---
    print("Calculating LOWESS trend line...")
    lowess_results = sm.nonparametric.lowess(filtered_hel_changes, filtered_accuracies, frac=0.33)
    x_trend, y_trend = lowess_results[:, 0], lowess_results[:, 1]
    
    # --- 5. Visualization ---
    print("\nGenerating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    scatter = ax.scatter(filtered_accuracies, filtered_hel_changes, alpha=0.5, edgecolors='w', s=50, c=filtered_accuracies, cmap='viridis', label='Prompts')
    ax.plot(x_trend, y_trend, color='crimson', linestyle='-', linewidth=3.5, label='LOWESS Trend')
    
    cbar = fig.colorbar(scatter)
    cbar.set_label('Mean Accuracy', rotation=270, labelpad=15)
    ax.set_title(f'Model Confidence Dynamics (Step {START_STEP} to {END_STEP})', fontsize=18, pad=20)
    ax.set_xlabel(f'Mean Accuracy across Steps {ACCURACY_STEPS}', fontsize=14)
    ax.set_ylabel(f'High Entropy Token Loss Change (Loss@{END_STEP} - Loss@{START_STEP})', fontsize=14)
    ax.legend()
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)

    x_margin = (filtered_accuracies.max() - filtered_accuracies.min()) * 0.05
    y_margin = (filtered_hel_changes.max() - filtered_hel_changes.min()) * 0.05
    ax.set_xlim(filtered_accuracies.min() - x_margin, filtered_accuracies.max() + x_margin)
    ax.set_ylim(filtered_hel_changes.min() - y_margin, filtered_hel_changes.max() + y_margin)
    
    fig.text(0.5, 0.01, 
             'Y<0: Increased confidence on ambiguous tokens | Y>0: Decreased confidence\n'
             'This plot shows how the model\'s loss on its most uncertain predictions changes relative to its overall accuracy.',
             ha='center', fontsize=10, style='italic', color='gray')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # --- 6. Saving the Plot ---
    output_filename = f"/data/figs/hel_vs_accuracy_{START_STEP}_to_{END_STEP}.png"
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\n✅ Analysis complete! Plot saved as {output_filename}")
    except Exception as e:
        print(f"\n❌ Error saving file: {e}. Please check write permissions.")
    plt.show()

if __name__ == '__main__':
    main()