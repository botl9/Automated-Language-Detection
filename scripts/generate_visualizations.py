from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "language-detection.csv"
OUTPUT_DIR = ROOT / "web" / "assets" / "visualizations"


def save_chart(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="#f8f6f1")
    plt.close()


def style_plot():
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": "#f8f6f1",
            "axes.facecolor": "#f8f6f1",
            "axes.edgecolor": "#d8d0c2",
            "grid.color": "#e2dccc",
            "text.color": "#111111",
            "axes.labelcolor": "#6f685d",
            "xtick.color": "#6f685d",
            "ytick.color": "#6f685d",
            "font.family": "DejaVu Sans",
        },
    )


def load_data():
    data = pd.read_csv(DATASET_PATH)
    data = data.dropna(subset=["Text", "Language"])
    data["Text"] = data["Text"].astype(str).str.strip()
    return data[data["Text"] != ""]


def train_model(data):
    x_train, x_test, y_train, y_test = train_test_split(
        data["Text"],
        data["Language"],
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 3),
                    max_features=5000,
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return y_test, predictions, accuracy


def plot_language_distribution(data):
    counts = data["Language"].value_counts().sort_values()
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x=counts.values, y=counts.index, color="#8c713e")
    ax.set_title("Dataset Language Distribution", fontsize=20, weight="bold", pad=18)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, padding=4, fontsize=9, color="#6f685d")
    save_chart(OUTPUT_DIR / "language-distribution.png")


def plot_confusion_matrix(y_test, predictions):
    labels = sorted(y_test.unique())
    matrix = confusion_matrix(y_test, predictions, labels=labels, normalize="true")
    plt.figure(figsize=(11, 9))
    ax = sns.heatmap(
        matrix,
        cmap="crest",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Normalized Accuracy"},
    )
    ax.set_title("Normalized Confusion Matrix", fontsize=20, weight="bold", pad=18)
    ax.set_xlabel("Predicted Language")
    ax.set_ylabel("Actual Language")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    save_chart(OUTPUT_DIR / "confusion-matrix.png")


def plot_language_performance(y_test, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=sorted(y_test.unique()),
        zero_division=0,
    )
    performance = pd.DataFrame(
        {
            "Language": sorted(y_test.unique()),
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    ).sort_values("F1 Score")

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(data=performance, x="F1 Score", y="Language", color="#7697b2")
    ax.set_xlim(0, 1.02)
    ax.set_title("Per-Language F1 Score", fontsize=20, weight="bold", pad=18)
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=4, fontsize=9, color="#6f685d")
    save_chart(OUTPUT_DIR / "language-performance.png")


def plot_accuracy_summary(accuracy, language_count, sample_count):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axis("off")
    fig.text(0.08, 0.78, "Model Performance Summary", fontsize=22, weight="bold")
    fig.text(0.08, 0.52, f"{accuracy:.2%}", fontsize=64, weight="bold", family="serif")
    fig.text(0.08, 0.37, "Overall test accuracy", fontsize=16, color="#6f685d")
    fig.text(0.62, 0.56, f"{language_count}", fontsize=36, weight="bold", family="serif")
    fig.text(0.62, 0.45, "Languages", fontsize=14, color="#6f685d")
    fig.text(0.62, 0.28, f"{sample_count:,}", fontsize=36, weight="bold", family="serif")
    fig.text(0.62, 0.17, "Dataset samples", fontsize=14, color="#6f685d")
    save_chart(OUTPUT_DIR / "accuracy-summary.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style_plot()

    data = load_data()
    y_test, predictions, accuracy = train_model(data)

    plot_accuracy_summary(accuracy, data["Language"].nunique(), len(data))
    plot_language_distribution(data)
    plot_confusion_matrix(y_test, predictions)
    plot_language_performance(y_test, predictions)

    print(f"Saved visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
