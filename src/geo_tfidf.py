"""geo-tfidf: per-class mean TF-IDF over AG News.

Classical NLP baseline from the GEO paper (Aggarwal et al., 2024). Loads AG News,
fits a TF-IDF vectorizer, and reports the top terms per news category.

Run with:  uv run python src/geo_tfidf.py
"""

from pathlib import Path
import html

import numpy as np
import pandas as pd
import datasets
import plotnine as p9
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# paths (project root is the parent of src/)
ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = ROOT / "img"

# config
DATASET = "wangrongsheng/ag_news"
SPLIT = "train[:5000]"
# World (0), Sports (1), Business (2), Sci/Tech (3)
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
# news-wire boilerplate / generic words on top of sklearn's English stop words
DOMAIN_STOPS = {"reuters", "ap", "afp", "said"}
TOP_N = 5


def load_news() -> pd.DataFrame:
    """Load AG News and return a cleaned (text, label) DataFrame."""
    ds = datasets.load_dataset(DATASET, split=SPLIT)
    df = pd.DataFrame(ds)
    return df.assign(
        # unescape html entities e.g. &#39; before tokenizing
        text=lambda d: d.text.map(html.unescape),
        label=lambda d: d.label.astype("category").cat.rename_categories(LABEL_NAMES),
    )


def vectorize(texts: pd.Series):
    """Fit TF-IDF and return (matrix, vocabulary)."""
    vec = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOPS),
        token_pattern=r"(?u)\b[a-z][a-z]+\b",  # letters only -> drops "39", "11", etc.
        ngram_range=(1, 2),  # uni-grams and bi-grams
        min_df=5,  # ignore terms in fewer than 5 docs -> kills one-off bigrams
    )
    X = vec.fit_transform(texts)
    return X, vec.get_feature_names_out()


def top_terms_per_class(X, vocab, labels: pd.Series, top_n: int = TOP_N) -> pd.DataFrame:
    """Mean TF-IDF per class, keeping the top_n terms for each label."""
    records = []
    for label, idx in labels.groupby(labels).groups.items():
        means = np.asarray(X[idx].mean(axis=0)).ravel()
        for j in means.argsort()[::-1][:top_n]:
            records.append({"label": label, "term": vocab[j], "mean_tfidf": means[j]})
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["label", "mean_tfidf"], ascending=[True, False])
        .reset_index(drop=True)
    )


def plot(top_terms: pd.DataFrame) -> p9.ggplot:
    """Scatter of mean TF-IDF, sized by weight, term vs class."""
    return (
        p9.ggplot(top_terms, p9.aes(x="label", y="term", size="mean_tfidf"))
        + p9.geom_point()
        + p9.theme_grey()
        + p9.labs(title="Mean TF-IDF Per Class")
    )


def main() -> None:
    df = load_news()
    print(df["label"].value_counts().sort_values())

    X, vocab = vectorize(df["text"])
    print(f"tf-idf matrix: {X.shape[0]} docs x {X.shape[1]} terms")

    top_terms = top_terms_per_class(X, vocab, df["label"])
    print(top_terms.to_string())

    PLOT_DIR.mkdir(exist_ok=True)
    out = PLOT_DIR / "plot.png"
    plot(top_terms).save(out)
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
