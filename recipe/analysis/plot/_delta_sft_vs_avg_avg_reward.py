import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.font_manager as fm

# --- Font & Style Configuration ---
# This approach separates the font properties dictionary from the global rcParams
# configuration to prevent errors.

# 1. Define the font properties to be used by individual plot elements.
#    This dictionary contains only keys that are valid for a 'fontdict'.
FONT_PROPERTIES = {
    'family': 'serif', # This tells matplotlib to use a font from the 'serif' family
    'size': 16
}

# 2. Configure the global settings for which font to use for the 'serif' family.
#    By providing a list, Matplotlib will try each font in order until it finds one
#    that is available on the system. This makes the script more portable.
mpl.rcParams['font.serif'] = [
    'Times New Roman', 
    # 'DejaVu Serif', 'Bitstream Vera Serif', 'Liberation Serif'
]
# We still set the math font globally
mpl.rcParams['mathtext.fontset'] = 'stix'

print("✅ Font properties defined. They will be applied directly to plot elements.")


# --- Main Configuration Area ---

# 1. Set the path to your root directory
ROOT_DIR = '//mnt/jfs2/cth/085b13/_analysis_v2/rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_7_clip_0.8_25-06-21-Jun-06-1750458724'

# 2. Define the steps to analyze
START_STEP = 128
END_STEP = START_STEP + 256
ACCURACY_STEPS = [_ for _ in range(START_STEP, START_STEP + 256, 32)]
# Correctly convert step numbers to strings for the filename
ACCURACY_STEPS_STR = "_".join([str(i) for i in ACCURACY_STEPS])
print(f"Analyzing accuracy steps: {ACCURACY_STEPS}")

# 3. Define filenames
FORWARD_FILE = 'forward.json'
GEN_FILE_PREFIX = 'gen_t1.0'
GEN_FILE_SUFFIX = '.json'

# 4. Set to True to filter outliers, False to include all data points
REMOVE_OUTLIERS = True

# --- Core Script Logic ---

def find_gen_file(directory: str) -> str | None:
    """Finds the generation result JSON file in the specified directory."""
    if not os.path.exists(directory):
        return None
    for fname in os.listdir(directory):
        if fname.startswith(GEN_FILE_PREFIX) and fname.endswith(GEN_FILE_SUFFIX):
            return os.path.join(directory, fname)
    return None

def load_and_process_losses(filepath: str) -> dict[str, float]:
    """
    Loads the forward.json dataset, groups by 'uid', and calculates the mean of 'final_loss'.
    Returns a dictionary mapping {uid: mean_loss}.
    """
    print(f"  - Loading and processing losses from: {filepath}")
    try:
        # Use a cache directory to speed up subsequent runs
        dset = load_dataset('json', data_files=filepath, split='train', cache_dir='./.cache')
        df = dset.to_pandas()
        uid_losses = df.groupby('uid')['final_loss'].mean()
        return uid_losses.to_dict()
    except Exception as e:
        print(f"    Error processing loss file {filepath}: {e}")
        return {}

def load_and_process_accuracies(filepath: str) -> dict[str, float]:
    """
    Loads the gen.json dataset and calculates the mean of the 'analysis_corr' list for accuracy.
    Returns a dictionary mapping {uid: accuracy}.
    """
    print(f"  - Loading and processing accuracies from: {filepath}")
    try:
        dset = load_dataset('json', data_files=filepath, split='train', cache_dir='./.cache')
        
        def calculate_accuracy(example):
            corr_list = example.get('analysis_corr', [])
            if corr_list and isinstance(corr_list, list):
                return {'accuracy': np.mean(corr_list)}
            return {'accuracy': 0.0}

        dset = dset.map(calculate_accuracy, num_proc=os.cpu_count() or 1)
        return dict(zip(dset['uid'], dset['accuracy']))
    except Exception as e:
        print(f"    Error processing accuracy file {filepath}: {e}")
        return {}

def main():
    """Main execution function"""
    print("🚀 Starting Analysis with Hugging Face Datasets...")

    # --- 1. Data Loading ---
    start_loss_path = os.path.join(ROOT_DIR, f'global_step{START_STEP}', FORWARD_FILE)
    end_loss_path = os.path.join(ROOT_DIR, f'global_step{END_STEP}', FORWARD_FILE)
    start_losses_map = load_and_process_losses(start_loss_path)
    end_losses_map = load_and_process_losses(end_loss_path)
    all_accuracies_map = {}
    for step in ACCURACY_STEPS:
        print(f"\nProcessing Step {step} for accuracy...")
        step_dir = os.path.join(ROOT_DIR, f'global_step{step}')
        gen_file_path = find_gen_file(step_dir)
        if not gen_file_path:
            print(f"  - Could not find generation file in {step_dir}. Skipping.")
            continue
        all_accuracies_map[step] = load_and_process_accuracies(gen_file_path)

    # --- 2. Data Alignment and Calculation ---
    print("\nAligning data and performing calculations...")
    uids_in_losses = set(start_losses_map.keys()) & set(end_losses_map.keys())
    valid_accuracy_maps = [set(acc_map.keys()) for acc_map in all_accuracies_map.values() if acc_map]
    if not valid_accuracy_maps:
        print("❌ Error: No valid accuracy data could be loaded. Aborting.")
        return
    uids_in_accuracies = set.intersection(*valid_accuracy_maps)
    common_uids = sorted(list(uids_in_losses & uids_in_accuracies))
    if not common_uids:
        print("❌ Error: No common UIDs found across all files.")
        return
    print(f"Found {len(common_uids)} common UIDs for analysis.")
    sft_loss_changes = []
    mean_accuracies = []
    for uid in tqdm(common_uids, desc="Calculating final metrics"):
        loss_change = end_losses_map[uid] - start_losses_map[uid]
        sft_loss_changes.append(loss_change)
        accs_for_uid = [all_accuracies_map[step][uid] for step in ACCURACY_STEPS if step in all_accuracies_map and uid in all_accuracies_map[step]]
        mean_acc = np.mean(accs_for_uid) if accs_for_uid else 0
        mean_accuracies.append(mean_acc)

    sft_loss_changes_np = np.array(sft_loss_changes)
    mean_accuracies_np = np.array(mean_accuracies)

    # --- 3. Outlier Removal (Conditional) ---
    if REMOVE_OUTLIERS:
        print("\nFiltering outliers from SFT loss changes...")
        q1 = np.percentile(sft_loss_changes_np, 25)
        q3 = np.percentile(sft_loss_changes_np, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        non_outlier_mask = (sft_loss_changes_np >= lower_bound) & (sft_loss_changes_np <= upper_bound)
        
        filtered_loss_changes = sft_loss_changes_np[non_outlier_mask]
        filtered_accuracies = mean_accuracies_np[non_outlier_mask]
        
        num_outliers = len(sft_loss_changes_np) - len(filtered_loss_changes)
        print(f"Removed {num_outliers} outliers based on 1.5 * IQR rule.")
    else:
        print("\nSkipping outlier removal as per configuration.")
        filtered_loss_changes = sft_loss_changes_np
        filtered_accuracies = mean_accuracies_np
    
    # Scale loss changes for y-axis representation
    scaled_filtered_loss_changes = filtered_loss_changes * 1000

    # --- 4. Trend Line Calculation using LOWESS ---
    print("Calculating LOWESS trend line...")
    lowess_results = sm.nonparametric.lowess(scaled_filtered_loss_changes, filtered_accuracies, frac=0.2)
    x_trend = lowess_results[:, 0] 
    y_trend = lowess_results[:, 1]
    
    # --- 5. Visualization ---
    print("\nGenerating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    # Scatter plot of the data points
    scatter = ax.scatter(filtered_accuracies, scaled_filtered_loss_changes,
                        alpha=0.45, edgecolors='w', s=65, c=filtered_accuracies,
                        cmap='viridis')

    # Plot the LOWESS trend line
    ax.plot(x_trend, y_trend, color='royalblue', linestyle='--', linewidth=5)

    # Add a colorbar and set its label font
    cbar = fig.colorbar(scatter, pad=0.02)
    cbar.set_label('Mean Accuracy', rotation=270, labelpad=10, fontdict=FONT_PROPERTIES)

    # Set axis labels using the explicit font properties
    ax.set_ylabel(r'$\Delta$ SFT Loss ($\times 10^{-3}$)', fontdict=FONT_PROPERTIES)
    ax.set_xlabel('Problem Difficulty', fontdict=FONT_PROPERTIES, labelpad=10)
    
    # Set annotation text using the explicit font properties
    ax.text(0.2, -0.05, r'Hard $\longleftarrow$',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            fontdict=FONT_PROPERTIES)
    ax.text(0.8, -0.05, r'$\longrightarrow$ Easy',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontdict=FONT_PROPERTIES)
            
    # Explicitly set the font for tick labels for maximum reliability
    for label in ax.get_xticklabels():
        label.set_fontfamily('serif')
        label.set_fontsize(14)
    for label in ax.get_yticklabels():
        label.set_fontfamily('serif')
        label.set_fontsize(14)
            
    # Final plot adjustments
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    x_margin = (filtered_accuracies.max() - filtered_accuracies.min()) * 0.05
    y_margin = (scaled_filtered_loss_changes.max() - scaled_filtered_loss_changes.min()) * 0.05
    ax.set_xlim(filtered_accuracies.min() - x_margin, filtered_accuracies.max() + x_margin)
    ax.set_ylim(scaled_filtered_loss_changes.min() - y_margin, scaled_filtered_loss_changes.max() + y_margin)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # --- 6. Saving the Plot ---
    output_filename = f"/data/_figs/sft_loss_vs_accuracy_{START_STEP}_to_{END_STEP}_ACC_{ACCURACY_STEPS_STR}.pdf"
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\n✅ Analysis complete! Plot saved as {output_filename}")
    except Exception as e:
        print(f"\n❌ Error saving file: {e}")
        print("Please check if you have write permissions for the /data/_figs/ directory.")

    plt.show()

if __name__ == '__main__':
    main()
