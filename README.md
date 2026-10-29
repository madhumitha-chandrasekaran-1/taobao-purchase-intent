# User Behavior & Purchase Prediction

Exploratory analysis and purchase-intent modeling on a sample of the Taobao **UserBehavior** dataset, released by Alibaba on [Tianchi](https://tianchi.aliyun.com/dataset/649) (100M events in full; [Kaggle mirror](https://www.kaggle.com/datasets/marwa80/userbehavior)). See [Data source and licence](#data-source-and-licence).

**Research question:** does enriching a simple view-signal baseline with deeper engagement features (favorites, cart adds, repeat visits, recency, funnel breadth) meaningfully improve purchase prediction?

**Short answer:** yes. Moving from V1 (logistic regression on a binary view signal) to V2 (XGBoost on 10 engagement features) raises test PR-AUC from **0.093 to 0.613**.

That gain needs **both** the engagement features and a nonlinear model:
- Either change alone gets only part of the way (see [Model vs features](#model-vs-features-what-drives-the-gain)).
- No single feature is essential. Removing `session_count` and `depth_score` together costs only 0.023 PR-AUC, because the engagement features overlap (see [Ablation](#ablation-what-the-model-needs)).
- The four features added in round 2 add nothing measurable.

All scores are **optimistic**, because the features use each pair's full history, including the purchase day (see [Limitations](#limitations-and-caveats)).

---

## Contents

1. [Dataset](#dataset)
2. [Pipeline overview](#pipeline-overview)
3. [Data cleaning](#data-cleaning)
4. [Exploratory analysis](#exploratory-analysis)
5. [Feature engineering](#feature-engineering)
6. [Modeling](#modeling)
7. [Results](#results)
8. [Model vs features: what drives the gain](#model-vs-features-what-drives-the-gain)
9. [Ablation: what the model needs](#ablation-what-the-model-needs)
10. [Threshold selection](#threshold-selection)
11. [Feature importance](#feature-importance)
12. [Train vs test](#train-vs-test)
13. [Limitations and caveats](#limitations-and-caveats)
14. [Data source and licence](#data-source-and-licence)
15. [Setup and troubleshooting](#setup-and-troubleshooting)
16. [Files](#files)
17. [Future work](#future-work)

---

## Dataset

| Field | Description |
|---|---|
| `user_id` | Anonymized user identifier |
| `item_id` | Product identifier |
| `category_id` | Product category |
| `behavior_type` | One of `pv` (page view), `fav` (favorite), `cart` (add to cart), `buy` (purchase) |
| `timestamp` | Unix timestamp. The platform's local time is **Beijing time (UTC+8)**. |

- **Full dataset:** 100,150,807 events from 987,994 users.
- **Documented collection window:** 2017-11-25 to 2017-12-03 (9 days, Beijing time).
- **Sample used here:** 10,000 users drawn at random (`np.random.seed(42)`), giving 1,009,809 events before cleaning and **839,613 after**. That is under 1% of the full dataset.

---

## Pipeline overview

```
UserBehavior.csv (100M events)
  → sample 10,000 users
  → convert timestamps to Beijing time, keep 2017-11-25 … 2017-12-03
  → validate (nulls, duplicates, behavior types)
  → remove IQR activity outliers
  → EDA (funnel, depth, category conversion, hour/day patterns)
  → reshape to one row per user–item pair
  → engineer features → model_ready.csv
  → 80/20 split → category feature rebuilt from training pairs only
  → V1 logistic regression vs V2 XGBoost (+ GridSearchCV)
  → threshold chosen on training-set CV predictions, reported once on test
  → model-vs-features comparison and ablation (CV on training set)
```

---

## Data cleaning

| Step | Rows | Users | Notes |
|---|---|---|---|
| Raw sample | 1,009,809 | 10,000 | |
| Out-of-window timestamps removed | 1,009,227 | 10,000 | 582 rows removed |
| IQR outlier users removed | **839,613** | **9,529** | 471 users removed |

**Nulls, duplicates and behavior types:** 0 nulls, 0 duplicate rows, 0 invalid `behavior_type` values.

**Timezone and out-of-window timestamps.** These are two separate issues:

1. **The timestamps are in Beijing time (UTC+8), not UTC.** In UTC, activity jumps from about 100 events per hour to 3,722 at 2017-11-24 16:00 UTC, which is exactly midnight on Nov 25 in Beijing. It stops at 2017-12-03 16:00 UTC, which is midnight on Dec 4 in Beijing. The notebook adds 8 hours when converting timestamps. Treating them as UTC would:
   - shift the hour-of-day patterns by 8 hours, and
   - make a date-window filter drop the first 8 hours of real data (11,764 events in this sample).
2. **582 rows fall outside the collection window:** 548 before it (as early as 2017-08-01) and 34 after it (Dec 4–5). A simple sanity check for dates before 2000 or after 2025 doesn't catch them. If kept, they would push `days_since_last` up to 126 days and `engagement_span_days` up to 14 days, both impossible in a 9-day window. They are removed.

**Activity outliers (IQR method on events per user):**

| Statistic | Value |
|---|---|
| Mean events per user | 100.9 |
| Median | 75.5 |
| Q1 / Q3 | 39 / 136 |
| IQR | 97 |
| Upper bound (Q3 + 1.5×IQR) | 281.5 |
| Lower bound | negative, so no low-activity users are removed |
| 95th / 99th percentile / max | 277 / 400 / 713 |

471 users above the upper bound were removed: whole users, not individual events. Unusually heavy activity is more likely to come from bots or resellers than from typical shoppers. See [Limitations](#limitations-and-caveats) for the trade-off.

---

## Exploratory analysis

All figures below are after cleaning (839,613 events, 9,529 users).

### Engagement funnel (event counts)

| Stage | Events | % of page views |
|---|---|---|
| Page view (`pv`) | 748,690 | 100% |
| Add to cart | 47,854 | 6.39% |
| Favorite | 24,465 | 3.27% |
| Purchase | 18,604 | 2.48% |

### Deepest engagement reached per user

| Deepest stage | Users | Share |
|---|---|---|
| Purchased | 6,425 | 67.4% |
| Added to cart | 2,019 | 21.2% |
| Viewed only | 612 | 6.4% |
| Favorited | 473 | 5.0% |

- Most *users* buy something over 9 days, even though most *item views* go nowhere.
- Buyers made **2.90 purchases** on average. The most purchases by one user was **34**.

### Category conversion rates

There are 201 categories with at least 500 page views. The 500-view minimum keeps tiny categories from producing noisy rates. The top 15 by purchases per view:

| Category | Views | Purchases | Conversion |
|---|---|---|---|
| 800581 | 641 | 71 | 11.08% |
| 174239 | 550 | 52 | 9.45% |
| 4159072 | 1,729 | 163 | 9.43% |
| 866301 | 972 | 86 | 8.85% |
| 3231625 | 612 | 53 | 8.66% |
| 2495340 | 625 | 51 | 8.16% |
| 3747017 | 616 | 49 | 7.95% |
| 1595193 | 503 | 40 | 7.95% |
| 267209 | 517 | 38 | 7.35% |
| 4919077 | 582 | 42 | 7.22% |
| 366112 | 501 | 36 | 7.19% |
| 50601 | 766 | 53 | 6.92% |
| 1973012 | 683 | 47 | 6.88% |
| 2297500 | 511 | 34 | 6.65% |
| 846990 | 1,109 | 70 | 6.31% |

Conversion varies several-fold by category, which is why both models get the `category_avg_depth` control feature.

### Time patterns (Beijing time)

- **Hour of day:** views, cart adds and purchases all peak at **21:00**, and favorites peak at **22:00**. The quietest hour is **03:00**, with 3,573 page views against 64,144 at 21:00.
- **Day of week. This chart needs care.** The 9-day window has **two Saturdays and two Sundays but only one of each weekday**, so the raw day-of-week totals roughly double-count weekends: Saturday has 207,141 events against 78,456 on Tuesday. Daily totals are a fairer view:

| Date | Day | Events |
|---|---|---|
| 2017-11-25 | Sat | 88,520 |
| 2017-11-26 | Sun | 88,094 |
| 2017-11-27 | Mon | 83,692 |
| 2017-11-28 | Tue | 78,456 |
| 2017-11-29 | Wed | 85,972 |
| 2017-11-30 | Thu | 87,359 |
| 2017-12-01 | Fri | 92,155 |
| 2017-12-02 | Sat | 118,621 |
| 2017-12-03 | Sun | 116,744 |

The second weekend is about 33% busier than the first. That may reflect shoppers warming up for the "Double 12" (Dec 12) sale, but this dataset can't confirm it.

---

## Feature engineering

**Unit of analysis:** one row per **user–item pair**: **644,064 pairs**, of which **17,841 (2.77%)** include a purchase.

**Target:** `purchased = 1` if the pair's behaviors include `buy`.

| Feature | Description | Used in |
|---|---|---|
| `has_view` | 1 if the user viewed the item | V1 |
| `view_count` | Number of page views | V2 |
| `has_fav` / `fav_count` | Whether / how many times favorited | V2 (`fav_count` added in round 2) |
| `has_cart` / `cart_count` | Whether / how many times added to cart | V2 uses `has_cart` |
| `depth_score` | Weighted engagement: pv×1 + fav×2 + cart×3 (purchases excluded) | V2 |
| `session_count` | Number of distinct days the user interacted with the item (the data has no session IDs) | V2 |
| `category_avg_depth` | Mean `depth_score` of the item's category (control variable). **Rebuilt from training pairs only** at modeling time | V1 and V2 |
| `days_since_last` | *(new)* Fractional days from the user's last interaction with the item to the dataset's last timestamp | V2 |
| `engagement_span_days` | *(new)* Calendar days between first and last interaction (0 if all on one day) | V2 |
| `funnel_steps` | *(new)* Number of distinct **pre-purchase** behaviors reached (pv, fav, cart). For example pv+cart = 2 | V2 |

**Excluding `buy` from features:** `depth_score` and `funnel_steps` don't count purchases, because that would put the target directly into a feature. They still include *views after a purchase*, which is a leakage path (see [Limitations](#limitations-and-caveats)).

**What the new features look like:**

| | Mean | Median | 75th pct | Max |
|---|---|---|---|---|
| `days_since_last` | 3.96 | 3.63 | 6.32 | 9.00 |
| `engagement_span_days` | 0.23 | 0 | 0 | 8 |

| `funnel_steps` | Pairs | Purchase rate |
|---|---|---|
| 0 | 4,992 | **100%** (bought with no view, favorite or cart event) |
| 1 | 607,411 | 1.59% |
| 2 | 30,993 | 9.88% |
| 3 | 668 | 17.66% |

The `funnel_steps = 0` group is always a purchase by construction. These are "buy-only" pairs, and the other features already identify them, since `view_count`, `has_fav` and `has_cart` are all 0.

**Repeat visits, illustrated** (notebook, last cell): among pairs with exactly 5 views, those on **1 day** convert at **13.0%** (818 pairs), while those spread over **3+ days** convert at **25.3%** (1,080 pairs), about 2.0×.

The engineered table is saved to `model_ready.csv` (644,064 rows). It includes a full-data `category_avg_depth` for reference, which the modeling section replaces.

---

## Modeling

- **Split:** 80/20 stratified train/test (`random_state=42`). Train has 515,251 rows and test has 128,813 rows (3,568 purchases). Both have a 2.77% purchase rate.
- **Category feature from training only:** after the split, `category_avg_depth` is recomputed from training pairs and mapped onto test pairs. The 236 test pairs in categories never seen in training get the overall training average.
- **V1 baseline:** logistic regression (`class_weight='balanced'`, `max_iter=1000`) on `has_view` + `category_avg_depth`.
- **V2:** XGBoost on the 10 features `view_count, has_fav, fav_count, has_cart, depth_score, session_count, category_avg_depth, days_since_last, engagement_span_days, funnel_steps`.
  - Class imbalance is handled with `scale_pos_weight = (1 − rate) / rate`, about 35.
  - Untuned settings: `n_estimators=200, max_depth=4, learning_rate=0.05`.
- **Tuning:** `GridSearchCV` with 3 folds, scored on `average_precision` (PR-AUC).
  - Grid: `n_estimators ∈ {100, 200, 300}`, `max_depth ∈ {3, 4, 6}`, `learning_rate ∈ {0.01, 0.05, 0.1}`, `subsample ∈ {0.8, 1.0}`. That is 54 candidates and 162 fits.
  - **Best parameters:** `learning_rate=0.05, max_depth=6, n_estimators=300, subsample=1.0`.
  - **Best CV PR-AUC:** 0.6024.
- **Test-set discipline:** the test set is used only to report the final V1 and V2 models, and the V2 metrics at the cutoff chosen on validation. Every comparison (model vs features, ablation, cutoff choice) uses 3-fold cross-validation on the training set: `StratifiedKFold(shuffle=True, random_state=42)`.

**Why PR-AUC is the headline metric:** only 2.77% of pairs are purchases. ROC-AUC can look strong just by ranking the many non-buyers correctly. PR-AUC measures how well the model finds the rare buyers, and a random model would score about 0.028.

---

## Results

### Headline metrics (test set)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| V1 baseline (logistic regression) | 0.6781 | 0.0933 |
| V2 untuned (XGBoost) | 0.9068 | 0.5976 |
| **V2 tuned (XGBoost)** | **0.9122** | **0.6127** |

| Lift over V1 | ROC-AUC | PR-AUC |
|---|---|---|
| V2 untuned (as printed in the notebook) | +33.7% | +540.2% |
| V2 tuned | +34.5% | +556.7% (≈6.6×) |

This lift compares two different models *and* two different feature sets at once. See the next section for how much each one contributes.

### Classification at the default 0.5 threshold

| Model | Precision (buy) | Recall (buy) | F1 (buy) | Accuracy |
|---|---|---|---|---|
| V1 | 0.125 | 0.356 | 0.185 | 0.91 |
| V2 tuned | 0.172 | 0.790 | 0.282 | 0.89 |

| Confusion matrix at 0.5 | TN | FP | FN | TP |
|---|---|---|---|---|
| V1 | 116,315 | 8,930 | 2,297 | 1,271 |
| V2 tuned | 111,677 | 13,568 | 750 | 2,818 |

### Curves

- **ROC curve** and **precision–recall curve:** V1 against V2 tuned, with the AUC values in the legends.
- The PR curve includes the random baseline at the positive rate, 0.028.

---

## Model vs features: what drives the gain

V1 → V2 changes the model (logistic regression → XGBoost) **and** the features (2 → 10). To separate the two, every combination was scored with 3-fold CV PR-AUC on the training set:

| | V1 features (2) | V2 features (10) |
|---|---|---|
| **Logistic regression** | 0.0830 | 0.2201 |
| **XGBoost** | 0.1424 | **0.6018** |

- **Features alone** (logistic regression, 2 → 10 features): 0.083 → 0.220.
- **Model alone** (V1's features, logistic regression → XGBoost): 0.083 → 0.142.
- **Both together:** 0.602. The gain is mostly an *interaction*: the engagement features pay off when a nonlinear model can combine them, for example as count thresholds and interactions between views, carts and repeat days.

How the comparison was set up:
- The logistic regression rows use a `StandardScaler` so the solver behaves well on the count features.
- The XGBoost rows reuse the hyperparameters tuned on the 10-feature set, which slightly favors that cell.

---

## Ablation: what the model needs

Feature importance shows what the model *used*. Ablation shows what it *needs*: retrain without a feature group and measure the drop. All rows use the tuned XGBoost, scored with 3-fold CV PR-AUC on the training set.

| Feature set | Features | CV PR-AUC | Change vs all |
|---|---|---|---|
| All 10 features | 10 | 0.6018 | — |
| Without `session_count` | 9 | 0.5809 | −0.0210 |
| Without `depth_score` | 9 | 0.5989 | −0.0030 |
| Without `session_count` and `depth_score` | 8 | 0.5785 | −0.0233 |
| Without the 4 round-2 features | 6 | 0.5999 | −0.0020 |

What this shows:
- **`session_count` matters most, but it isn't essential.** Removing it costs 0.021, about 3.5% of the score.
- **`depth_score` is almost fully redundant** (−0.003). It is a weighted sum of `view_count`, `fav_count` and the cart count, and the model can rebuild it from those.
- **The engagement features overlap heavily.** Removing the two most-used features together costs only 0.023, so the remaining features (views, cart/fav flags, `funnel_steps`, `engagement_span_days`) carry most of the same information.
- **The round-2 features add nothing measurable** (−0.002, within the fold-to-fold spread of about ±0.005).

---

## Threshold selection

V2's scores are pushed upward by `scale_pos_weight`, so 0.5 is not a useful cutoff. **The cutoff is chosen without looking at the test set:**

1. Get out-of-fold predictions on the training set from 3-fold CV with the tuned parameters. Every training pair is scored by a model that never saw it.
2. Sweep cutoffs from 0.05 to 0.95 in steps of 0.01 on those predictions, and pick the one with the highest F1: **0.92** (validation F1 0.576).
3. Apply that fixed cutoff **once** to the test set:

| On the test set | Cutoff 0.5 | Cutoff 0.92 (chosen on validation) |
|---|---|---|
| Precision | 0.172 | **0.790** |
| Recall | 0.790 | **0.466** |
| F1 | 0.282 | **0.586** |
| Pairs flagged | 16,386 | 2,103 |
| TN / FP / FN / TP | 111,677 / 13,568 / 750 / 2,818 | 124,804 / 441 / 1,906 / 1,662 |

- F1 roughly doubles compared with 0.5.
- Pick a lower cutoff when missing a buyer is costly, and a higher one when each flag costs money.
- These are model-evaluation numbers, not a business result. They inherit the leakage described below.

---

## Feature importance

Tuned V2, 10 features (XGBoost gain-based importance):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `session_count` | 0.6088 |
| 2 | `depth_score` | 0.2227 |
| 3 | `view_count` | 0.0732 |
| 4 | `funnel_steps` *(new)* | 0.0445 |
| 5 | `category_avg_depth` | 0.0186 |
| 6 | `has_cart` | 0.0095 |
| 7 | `has_fav` | 0.0075 |
| 8 | `days_since_last` *(new)* | 0.0073 |
| 9 | `engagement_span_days` *(new)* | 0.0048 |
| 10 | `fav_count` *(new)* | 0.0034 |

Read this alongside the [ablation](#ablation-what-the-model-needs). `depth_score` has 22% of the importance, but removing it costs almost nothing, because importance credits whichever of several overlapping features the trees happened to split on.

---

## Train vs test

| Model | Train ROC-AUC | Test ROC-AUC | Train PR-AUC | Test PR-AUC | PR-AUC gap |
|---|---|---|---|---|---|
| V1 Logistic Regression | 0.6850 | 0.6781 | 0.0829 | 0.0933 | −0.0105 |
| V2 XGBoost (tuned) | 0.9273 | 0.9122 | 0.6278 | 0.6127 | 0.0151 |

V2's train, test and CV scores (0.628 / 0.613 / 0.602) are close, so the model isn't memorizing training pairs. **This does not show the scores are trustworthy.** The split is random, so train and test share the same users and days, and the leakage below affects both halves equally. A small gap can't detect it.

---

## Limitations and caveats

1. **Target leakage makes all scores optimistic.** Every feature is computed over the pair's *entire* history, including events at or after the purchase:
   - the buy day counts toward `session_count`;
   - views after buying count toward `view_count`, `depth_score` and `category_avg_depth`;
   - the buy event can be the "last interaction" in `days_since_last` and `engagement_span_days`.

   The model partly learns "this pair went deep in the funnel", which comes bundled with buying. This likely helps V2 (many engagement features) more than V1 (only `has_view`), so it probably inflates the V1 → V2 gap too. The fix: build features only from events before a cutoff date and predict purchases after it.
2. **Random split, not time-based.** The same users and days appear in train and test, so the test set doesn't measure how well the model predicts future behavior.
3. **CV folds share the category feature.** `category_avg_depth` is built from the full training set, so inside cross-validation each validation fold contributed to it. This slightly flatters the CV numbers (comparisons, ablation, cutoff choice), but not the test results.
4. **Uncalibrated scores.** `scale_pos_weight` turns V2's outputs into ranking scores rather than probabilities. That is why the useful cutoff is 0.92.
5. **The IQR rule removes the heaviest users.** These 471 users may include bots, but also the most engaged real shoppers.
6. **Single sample.** One 10,000-user sample (seed 42), under 1% of the full data. The fold-to-fold spread is about ±0.005 PR-AUC, and differences smaller than that shouldn't be read as real.
7. **Buy-only pairs** (`funnel_steps = 0`, 4,992 pairs) are purchases by construction.

---

## Data source and licence

- **Original source:** the Taobao *UserBehavior* dataset, released by Alibaba on the Tianchi platform: [tianchi.aliyun.com/dataset/649](https://tianchi.aliyun.com/dataset/649).
- **Copy used here:** the Kaggle mirror [marwa80/userbehavior](https://www.kaggle.com/datasets/marwa80/userbehavior), downloaded with `kagglehub`.
- **Licence: not yet verified.**
  - The Tianchi page didn't show licence terms to an automated check, and the Kaggle mirror doesn't state one.
  - Tianchi's [licensing guide](https://tianchi.aliyun.com/specials/promotion/license) says terms vary by dataset, and Alibaba datasets are often limited to research use.
  - Before publishing derived data (`model_ready.csv.gz`), open the Tianchi page while logged in and confirm that its terms allow redistribution.
- **Citation:** use whatever the Tianchi page asks for. Papers using this dataset often cite Zhu et al., *"Learning Tree-based Deep Model for Recommender Systems"* (KDD 2018). Confirm that on the Tianchi page before relying on it.

---

## Setup and troubleshooting

```bash
pip install kagglehub pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

- Run the notebook top to bottom.
  - Cell 0 installs packages with `%pip`, so they go into the kernel's own environment.
  - Cell 1 downloads the dataset with `kagglehub`, which caches it under `~/.cache/kagglehub/`.
- It writes `model_ready.csv` to `~/data/`.
- **Memory:** loading the full 100M-row CSV needs a few GB of RAM. It was tested on an 8 GB Mac.
- **Runtime:** about 20 minutes, mostly the grid search (162 fits) and the CV comparisons.

### Dashboard

```bash
pip install -r requirements.txt
cd ~/data && streamlit run app.py
```

`app.py` is a 4-tab Streamlit dashboard (EDA, Features, Model Results, Insights). It does **not** retrain anything:
- The EDA charts and the User A vs User B example are computed from `model_ready.csv` (or `model_ready.csv.gz`).
- Model results are hardcoded from the final notebook run, in the `RESULTS` dict at the top of `app.py`. Update that dict if you re-run the notebook.
- The ROC curves are approximated from the AUC values. The notebook has the exact curves.

**Dashboard vs notebook charts:**

| Chart / table | Notebook (matplotlib) | Dashboard (Plotly) |
|---|---|---|
| Engagement funnel | ✓ | ✓ same numbers |
| Purchase rate by depth score / by session count | — | ✓ |
| User A vs User B | printed numbers | ✓ |
| Feature importance | ✓ | ✓ same values |
| Confusion matrices (cutoff 0.5) | ✓ | ✓ same counts |
| ROC curve | ✓ exact | ≈ approximated from AUC |
| Precision–recall curve, threshold sweep plot | ✓ | — (dashboard shows the chosen cutoff's test metrics) |
| Train vs test, model vs features, ablation | ✓ tables | ✓ tables |
| Events-per-user boxplots, deepest stage, category conversion, hour/day patterns | ✓ | — |

**Deploying (Streamlit Community Cloud, free):** check the [licence](#data-source-and-licence) first, because deploying publishes `model_ready.csv.gz`.
1. Push this folder to its own public GitHub repo.
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app** and pick the repo, the branch `main` and the file `app.py`.
3. Put the resulting `*.streamlit.app` URL in `README_URL` (in `app.py`) and in `PORTFOLIO.md`.

Troubleshooting:

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'kagglehub'` | Packages aren't installed in the kernel's environment | Run the install cell (cell 0) first. It uses `%pip`, so packages go into the kernel's environment |
| `XGBoostError: ... libomp.dylib` (macOS) | XGBoost needs the OpenMP runtime | `brew install libomp`, then restart the kernel |
| `ModuleNotFoundError: No module named 'numpy'` when running `app.py` | VS Code used a different Python (Homebrew), and `python app.py` doesn't start Streamlit | Select the Python that has the packages; run `streamlit run app.py` |
| Notebook edits "disappear" in VS Code | VS Code kept an older copy of the notebook open | **Revert File** (Cmd+Shift+P) or close and reopen without saving |

---

## Files

```
data/
  userbehavior.ipynb   # Main notebook
  app.py               # Streamlit dashboard (display only, no training)
  requirements.txt     # Dashboard dependencies (used by Streamlit Cloud)
  model_ready.csv      # Feature-engineered user–item table (generated; git-ignored)
  model_ready.csv.gz   # Compressed copy for the dashboard
  README.md            # This file
  PORTFOLIO.md         # Portfolio case study
  .gitignore
```

---

## Future work

- **Remove leakage:** build features from days 1–7 only and predict purchases on days 8–9. Report how much of the score survives, and how much of the V1 → V2 gap.
- **Time-based or user-grouped split.**
- **Compute `category_avg_depth` inside each CV fold** (for example with a scikit-learn pipeline) to remove the small CV leak.
- **Calibrate V2's scores** (Platt or isotonic scaling).
- **Repeat with several user samples** to get confidence intervals.
- **Time-of-day features:** the evening peak is unused by the model.
