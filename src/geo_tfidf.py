"""geo-tfidf: per-class mean TF-IDF over AG News.

Classical NLP baseline from the GEO paper (Aggarwal et al., 2024). Loads AG News,
fits a TF-IDF vectorizer, and reports the top terms per news category.

Pipeline:
    1. Load + clean the AG News dataset.
    2. Fit a TF-IDF vectorizer -> document-term matrix + vocabulary.
    3. Take the per-class mean TF-IDF and keep the top terms per class.
    4. Plot and save the result.

Run with:  uv run python src/geo_tfidf.py
"""

from pathlib import Path
import html

import numpy as np
import numpy.typing as npt
import pandas as pd
import datasets
import plotnine as p9
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# paths (project root is the parent of src/)
ROOT: Path = Path(__file__).resolve().parent.parent
PLOT_DIR: Path = ROOT / "img"

# config
DATASET: str = "wangrongsheng/ag_news"
SPLIT: str = "train[:5000]"
# World (0), Sports (1), Business (2), Sci/Tech (3)
LABEL_NAMES: dict[int, str] = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
# news-wire boilerplate / generic words on top of sklearn's English stop words
DOMAIN_STOPS: set[str] = {"reuters", "ap", "afp", "said"}
TOP_N: int = 5


def load_news() -> pd.DataFrame:
    """Load AG News and return a cleaned DataFrame.

    Returns:
        pd.DataFrame with columns:
            text  (str)      -- HTML-unescaped article text
            label (category) -- one of World / Sports / Business / Sci/Tech
    """
    # 1. download the split as a HuggingFace Dataset
    ds = datasets.load_dataset(DATASET, split=SPLIT)
    # 2. to DataFrame, then clean text + name the integer labels
    df = pd.DataFrame(ds)
    return df.assign(
        # unescape html entities e.g. &#39; before tokenizing
        text=lambda d: d.text.map(html.unescape),
        label=lambda d: d.label.astype("category").cat.rename_categories(LABEL_NAMES),
    )


def vectorize(texts: pd.Series) -> tuple[csr_matrix, npt.NDArray[np.str_]]:
    """Fit a TF-IDF vectorizer over the corpus.

    Args:
        texts: pd.Series[str] -- one document per row.

    Returns:
        X     -- scipy.sparse.csr_matrix, shape (n_docs, n_terms), float64 weights.
        vocab -- np.ndarray[str], shape (n_terms,), term for each matrix column.
    """
    # 1. instantiate the vectorizer
    vec = TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOPS),
        token_pattern=r"(?u)\b[a-z][a-z]+\b",  # letters only -> drops "39", "11", etc.
        ngram_range=(1, 2),  # uni-grams and bi-grams
        min_df=5,  # ignore terms in fewer than 5 docs -> kills one-off bigrams
    )
    # 2. fit + transform -> sparse document-term matrix
    X = vec.fit_transform(texts)
    # 3. column index -> term lookup
    return X, vec.get_feature_names_out()


def top_terms_per_class(
    X: csr_matrix,
    vocab: npt.NDArray[np.str_],
    labels: pd.Series,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Mean TF-IDF per class, keeping the strongest terms for each label.

    Args:
        X:      scipy.sparse.csr_matrix, shape (n_docs, n_terms) -- TF-IDF weights.
        vocab:  np.ndarray[str], shape (n_terms,) -- term per matrix column.
        labels: pd.Series[category] -- class label per row of X.
        top_n:  int -- number of terms to keep per class.

    Returns:
        pd.DataFrame with columns:
            label      (category) -- class
            term       (str)      -- vocabulary term
            mean_tfidf (float64)  -- mean weight of that term within the class
    """
    # 1. build a (document x term) table with the class label attached
    tfidf = pd.DataFrame(X.toarray(), columns=vocab)
    tfidf["label"] = labels.values

    # 2. collapse to (class x term) by averaging within each class
    means = tfidf.groupby("label").mean()
    means.columns.name = "term"  # names the level that stack() creates

    # 3. reshape to long form and take the top_n terms per class
    return (
        means.stack()
        .rename("mean_tfidf")  # Series.name sets the value column after reset_index
        .reset_index()  # -> label, term, mean_tfidf
        .sort_values(["label", "mean_tfidf"], ascending=[True, False])
        .groupby("label")
        .head(top_n)
        .reset_index(drop=True)
    )


def plot(top_terms: pd.DataFrame) -> p9.ggplot:
    """Build the scatter of mean TF-IDF, sized by weight, term vs class.

    Args:
        top_terms: pd.DataFrame from `top_terms_per_class`
                   (columns: label, term, mean_tfidf).

    Returns:
        p9.ggplot -- unrendered plot; call `.save(path)` to write it.
    """
    return (
        p9.ggplot(top_terms, p9.aes(x="label", y="term", size="mean_tfidf"))
        + p9.geom_point()
        + p9.theme_grey()
        + p9.labs(title="Mean TF-IDF Per Class")
    )


def main() -> None:
    """Run the full pipeline and save the plot to img/plot.png."""
    # 1. load + clean the dataset
    df = load_news()
    print(df["label"].value_counts().sort_values())

    # 2. fit TF-IDF -> document-term matrix + vocabulary
    X, vocab = vectorize(df["text"])
    print(f"tf-idf matrix: {X.shape[0]} docs x {X.shape[1]} terms")

    # 3. per-class mean -> top terms per class
    top_terms = top_terms_per_class(X, vocab, df["label"])
    print(top_terms.to_string())

    # 4. plot + save
    PLOT_DIR.mkdir(exist_ok=True)
    out = PLOT_DIR / "plot.png"
    plot(top_terms).save(out)
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
