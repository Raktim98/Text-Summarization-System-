import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import pandas as pd


class ModelAccuracyAnalyzer:
    def __init__(self):
        self.benchmarks = {
            'BART (Paper)':    {'rouge1': 0.44, 'rouge2': 0.23, 'rougeL': 0.42, 'bleu': 0.22, 'accuracy': 0.88},
            'T5 (Paper)':      {'rouge1': 0.43, 'rouge2': 0.20, 'rougeL': 0.39, 'bleu': 0.20, 'accuracy': 0.82},
            'Pegasus (Paper)': {'rouge1': 0.47, 'rouge2': 0.26, 'rougeL': 0.47, 'bleu': 0.24, 'accuracy': 0.93},
        }

    def generate_comparison_charts(self, current_metrics, model_name="Your Model"):
        data = self.benchmarks.copy()
        data[model_name] = {
            'rouge1':   current_metrics.get('rouge1', 0),
            'rouge2':   current_metrics.get('rouge2', 0),
            'rougeL':   current_metrics.get('rougeL', 0),
            'bleu':     current_metrics.get('bleu', 0),
            'accuracy': current_metrics.get('accuracy', 0),
        }
        df = pd.DataFrame(data).T.reset_index().rename(columns={'index': 'Model'})
        return {
            'metrics_bar':     self._plot_metrics_bar(df),
            'accuracy_gauge':  self._plot_accuracy_comparison(df),
        }

    def _plot_metrics_bar(self, df):
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        df_melt = df.melt(id_vars="Model",
                          value_vars=['rouge1', 'rouge2', 'rougeL', 'bleu'],
                          var_name="Metric", value_name="Score")
        ax = sns.barplot(x="Metric", y="Score", hue="Model",
                         data=df_melt, palette="viridis")
        plt.title("Performance Comparison: Real-Time vs Benchmarks", fontsize=14)
        plt.ylim(0, 1.0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        return self._to_base64(plt)

    def _plot_accuracy_comparison(self, df):
        plt.figure(figsize=(8, 5))
        sns.set_style("whitegrid")
        ax = sns.barplot(x="Model", y="accuracy", data=df, palette="magma")
        plt.title("Model Accuracy (ROUGE-L > Threshold)", fontsize=14)
        plt.ylabel("Accuracy % (0.0 - 1.0)")
        plt.ylim(0, 1.0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        return self._to_base64(plt)

    def _to_base64(self, plt_obj):
        buf = io.BytesIO()
        plt_obj.savefig(buf, format='png', bbox_inches='tight')
        plt_obj.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
