import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import configargparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from metrics import metrics
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc

CMAP = "colorblind"
DARK_MODE = True
AXIS_FONT_SIZE = 18
TICKS_FONT_SIZE = 16
TITLE_FONT_SIZE = 24

SHOW_METRICS = [
    # "TRA", 
    # "DET",
    "1-TRA",
    "1-DET",
    "AOGM", 
    "fp_edges",
    "fn_edges", 
    "False Positive Divisions",
    "False Negative Divisions", 
    "Division F1", 
    "association_accuracy", 
    "Mitotic Branching Correctness"
]
METRICS_INFO = {
    "metrics_to_sum": [
        "AOGM",
        "fp_nodes",
        "fn_nodes",
        "ns_nodes",
        "fp_edges",
        "fn_edges",
        "ws_edges",
        "gt_nodes",
        "pred_nodes",
        "gt_edges",
        "pred_edges",
        "Total GT Divisions",
        "Total Predicted Divisions",
        "True Positive Divisions",
        "False Positive Divisions",
        "False Negative Divisions",
        "Wrong Children Divisions",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
        "aa_match",
        "aa_edge_count",
        ],
    "metrics_to_mean": [
        "DET",
        "1-DET",
        "TRA",
        "1-TRA",
        "Division Recall",
        "Division Precision",
        "Division F1",
        "Mitotic Branching Correctness",
        "idf1",
        "idp",
        "idr",
        "recall",
        "precision",
        "mota",
        "motp",
        "association_accuracy",
    ],
    "ignore": [
        "start",
        "stop",
        "num_frames",
        "num_objects",
    ],
    "keep": [
        "folder",
        "run_name",
    ]
}

_REPLACE_NAMES = {
    "fp_nodes": "False Positive Nodes",
    "fn_nodes": "False Negative Nodes",
    "fp_edges": "False Positive Edges",
    "fn_edges": "False Negative Edges",
    "association_accuracy": "Association Accuracy",
    "recall": "Recall",
    "precision": "Precision",
    "mota": "MOTA",
    "motp": "MOTP",
    "idf1": "ID F1 Score",
    "idp": "ID Precision",
    "idr": "ID Recall",
    "mostly_tracked": "# of Mostly Tracked",
    "partially_tracked": "# of Partially Tracked",
    "num_false_positives": "# of False Positives",
    "num_misses": "# of Misses",
    "num_switches": "# of Switches",
    "num_fragmentations": "# of Fragmentations",
    "aa_match": "Association Accuracy (Matching)",
    "aa_edge_count": "Association Accuracy (Edge Count)",
    "1-TRA": "1 - TRA",
    "1-DET": "1 - DET",
}


def aggregate_metrics(metrics_df, raise_if_unknown=False):
    """Aggregates the metrics dataframe by summing or averaging the metrics over all the frames."""
    summary_df = pd.DataFrame()
    metrics_df = metrics_df.copy()
    metrics_df["1-TRA"] = 1 - metrics_df["TRA"]
    metrics_df["1-DET"] = 1 - metrics_df["DET"]
    for col in metrics_df.columns:
        if col in METRICS_INFO["metrics_to_sum"]:
            summary_df[col] = [metrics_df[col].sum()]
        elif col in METRICS_INFO["metrics_to_mean"]:
            summary_df[col] = [metrics_df[col].mean()]
        elif col in METRICS_INFO["ignore"]:
            continue
        elif col in METRICS_INFO["keep"]:
            summary_df[col] = [metrics_df[col].iloc[0]]
        else:
            if raise_if_unknown:
                raise ValueError(f"Unknown column {col} in metrics dataframe")
            else:
                warnings.warn(f"Unknown column {col} in metrics dataframe will be skipped")
                continue
    return summary_df


def _format_metric_name(names):
    for name in names:
        if name in _REPLACE_NAMES:
            names[names == name] = _REPLACE_NAMES[name]
    return names
    

def plot_metrics_per_model(df, metrics_to_plot: Literal["all", "preset"] | list[str] = "preset", plot_type: Literal["box", "bar"] = "box"):
    """Plots the metrics per model. The metrics are aggregated by summing or averaging them.
    
    Args:
        df (pd.DataFrame): The dataframe containing the metrics.
        metrics_to_plot (str or list): The metrics to plot. If "all", all metrics will be plotted. If "preset", the preset metrics will be plotted. If a list, only the metrics in the list will be plotted, if valid.
        plot_type (str): The type of plot to create. Either "box" or "bar".
    """
    context = nullcontext() if not DARK_MODE else plt.style.context("dark_background")
    
    with context:
        agg_dfs = []
        unique_models = df["run_name"].unique()
        len(unique_models)
        for model in unique_models:
            for folder in df["folder"].unique():
                model_df = df[df["run_name"] == model]
                model_df = model_df[model_df["folder"] == folder]
                agg_df = aggregate_metrics(model_df)
                agg_df["run_name"] = model
                agg_df["folder"] = folder
                agg_dfs.append(agg_df)
            agg_df = pd.concat(agg_dfs, ignore_index=True)
        
        agg_df.reset_index(drop=True, inplace=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=200)
        fig.suptitle("Metrics per model", fontsize=TITLE_FONT_SIZE)
        
        if metrics_to_plot == "all":
            metrics_to_plot = agg_df.columns.to_list()
        elif metrics_to_plot == "preset":
            metrics_to_plot = SHOW_METRICS
        else:
            all_valid = METRICS_INFO["metrics_to_sum"] + METRICS_INFO["metrics_to_mean"]
            metrics_to_plot = [m for m in metrics_to_plot if m in agg_df.columns and m in all_valid]
            
        metrics_to_plot = [m for m in metrics_to_plot if m not in METRICS_INFO["ignore"]]
        metrics_to_plot = [m for m in metrics_to_plot if m not in METRICS_INFO["keep"]]
        
        metrics_sum = [m for m in metrics_to_plot if m in METRICS_INFO["metrics_to_sum"]]
        metrics_mean = [m for m in metrics_to_plot if m in METRICS_INFO["metrics_to_mean"]]
        
        def plot(ax, plot_fn, plot_kwargs, metrics, agg_df, axis_title, disable_legend=False):
            plot_fn(
                data=agg_df.melt(id_vars=["run_name"], value_vars=metrics),
                x="variable",
                y="value",
                hue="run_name",
                ax=ax,
                palette=CMAP,
                **plot_kwargs,
            )
            if plot_type == "box":
                sns.stripplot(
                    data=agg_df.melt(id_vars=["run_name"], value_vars=metrics),
                    x="variable",
                    y="value",
                    hue="run_name",
                    ax=ax,
                    palette=CMAP,
                    dodge=True,
                    marker="o",
                )
                handles, labels = ax.get_legend_handles_labels()

                n_models = len(labels) // 2  # Each model has two entries: one for boxplot and one for stripplot
                paired_handles = [(handles[i], handles[i + n_models]) for i in range(n_models)]
                paired_labels = labels[:n_models]  # Use the first half of the labels for the legend

                ax.legend(
                    handles=paired_handles,
                    labels=paired_labels,
                    loc='best',
                    handlelength=4,
                    handler_map={tuple: HandlerTuple(ndivide=None)}
                )

            formatted_metrics = _format_metric_name(pd.Series(metrics)).tolist()
            ax.set_yscale("log")
            ax.set_ylabel(axis_title, fontsize=AXIS_FONT_SIZE)
            ax.set_xlabel("Metrics", fontsize=AXIS_FONT_SIZE)
            ax.set_xticks(range(len(metrics)))  # Ensure fixed tick positions
            ax.set_xticklabels(formatted_metrics, rotation=45, ha="right", fontsize=TICKS_FONT_SIZE)
            ax.get_legend().set_title("Model", prop={"size": AXIS_FONT_SIZE})
            if disable_legend:
                ax.get_legend().remove()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore UserWarnings
            plot_fn = sns.boxplot if plot_type == "box" else sns.barplot
            
            if plot_type == "box":
                plot_kwargs = {"boxprops": {"alpha": 0.5}, "showfliers": False}
                if DARK_MODE:
                    plot_kwargs.update({
                        "whiskerprops": {"color": "white"},
                        "capprops": {"color": "white"},
                        "medianprops": {"color": "white"},
                        "flierprops": {"markerfacecolor": "white", "markeredgecolor": "white"},
                    })
            elif plot_type == "bar":
                plot_kwargs = {
                    "errorbar": ("ci", 95),
                    "alpha": 0.85,
                    "capsize": 0.4,
                    } 
                if DARK_MODE:
                    plot_kwargs.update({
                    "err_kws": {"color": "white", "linewidth": 2, "alpha": 0.5},
                    })
            
            if metrics_sum:
                plot(
                    ax=ax1,
                    plot_fn=plot_fn,
                    plot_kwargs=plot_kwargs,
                    metrics=metrics_sum,
                    agg_df=agg_df,
                    axis_title="Summed metrics (log)",
                )
                
            if metrics_mean:
                plot(
                    ax=ax2,
                    plot_fn=plot_fn,
                    plot_kwargs=plot_kwargs,
                    metrics=metrics_mean,
                    agg_df=agg_df,
                    axis_title="Averaged metrics (log)",
                    disable_legend=True,
                )
            
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            sns.despine(ax=ax1)
            sns.despine(ax=ax2, right=False, left=True)
            plt.tight_layout()
            
            plt.show()


def load_results_to_df(run_name, results_path):
    """Loads all csvs in the target folder and returns a dataframe with the results."""
    results_path = Path(results_path).resolve()
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results path {results_path} does not exist.")
    
    #  find all folder with "tracked_" in the name
    tracked_folders = [f for f in results_path.iterdir() if f.is_dir() and "tracked_" in f.name]
    
    # for each folder, load the metrics/metrics.csv file
    dfs = []
    
    for folder in tracked_folders:
        metrics_path = folder / "metrics" / "metrics.csv"
        
        if not metrics_path.exists():
            print(f"Metrics path {metrics_path} does not exist.")
            continue
        
        df = pd.read_csv(metrics_path)
        
        # add a column with the folder name
        df["folder"] = folder.name
        
        # add a column with the run name
        df["run_name"] = run_name
        
        dfs.append(df)
        
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.reset_index(drop=True, inplace=True)
    return final_df


def compute_from_config(config_dir, model_dir, outdir, replace_path: dict =None):
    """Compute metrics from a model's folder generated by training."""
    model = Trackastra.from_folder(model_dir)
    
    config_parser = configargparse.YAMLConfigFileParser()
    
    with Path(config_dir).open("r") as f:
        config = config_parser.parse(f)
    
    test_folders = config["input_test"]
    
    ###
    if replace_path is not None:
        for i, f in enumerate(test_folders):
            for k, v in replace_path.items():
                if k in f:
                    test_folders[i] = f.replace(k, v)
    print(f"Test folders: {test_folders}")
    imgs_paths = [Path(f).resolve() / "img" for f in test_folders]
    det_paths = [Path(f).resolve() / "TRA" for f in test_folders]
    
    dfs = {}
    
    for img_path, det_path in zip(imgs_paths, det_paths):
        folder = img_path.parent
        tracking_graph, masks = model.track_from_disk(
            img_path,
            det_path,
            mode="greedy",
        )
        
        tracked_path = outdir / f"tracked_{folder.name}"
        df, masks = graph_to_ctc(tracking_graph, masks, outdir=tracked_path)
        
        metrics_df, _mot_events, _div_events, _edge_errors = metrics(
            gt=det_path,
            pred=tracked_path,
            outdir=tracked_path / "metrics",
            n_workers=8,
        )
        metrics_df["folder"] = folder.name
        dfs[folder.name] = metrics_df
        print(f"Metrics for {folder}:")
        scores_df = aggregate_metrics(metrics_df)
        print(scores_df[SHOW_METRICS].T)
        print("*" * 20)
        
    for folder, df in dfs.items():
        print(f"Metrics for {folder}:")
        scores_df = aggregate_metrics(df)
        print(scores_df[SHOW_METRICS].T)
        print("*" * 20)
        

# if __name__ == "__main__":

#     # config_dir = "/home/achard/code/train_configs/bacteria/sam21_zih_bacteria.yaml"
#     config_dir = Path("./train_scripts/bacteria/sam21_zih_bacteria.yaml").resolve()
#     # model_dir = "/home/achard/trastra_runs/..."
#     model_dir = Path("./eval_scripts/RUNS/EVAL/2025-04-17_11-56-05_vanvliet_wrfeat").resolve()
#     # outdir = "/home/achard/trastra_runs/eval/"
#     outdir = Path("./eval_scripts/metrics_output/").resolve()
    
#     compute_from_config(config_dir, model_dir, outdir)
