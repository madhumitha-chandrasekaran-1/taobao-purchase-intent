# Attention Depth Score — Predicting Purchases from How People Engage

**Dashboard:** [taobao-purchase-intent.streamlit.app](https://taobao-purchase-intent-nporuhbvrvk4wor5dcjmdn.streamlit.app/) · **Code:** [github.com/madhumitha-chandrasekaran-1/taobao-purchase-intent](https://github.com/madhumitha-chandrasekaran-1/taobao-purchase-intent) · **Full technical write-up:** [README.md](README.md)

> Clicks and views treat a half-second glance the same as someone who comes back three days in a row. I tested whether *depth* and *persistence* of attention predict purchases better than a simple "did they view it?" signal, using real e-commerce behavior data from Taobao.

---

## At a glance

| | |
|---|---|
| **Data** | Taobao UserBehavior (Alibaba, via [Tianchi](https://tianchi.aliyun.com/dataset/649)): 100M events, ~1M users, 9 days in Nov–Dec 2017 |
| **What I modeled** | A 10,000-user sample: 839,613 cleaned events (under 1% of the full data) → 644,064 user–item pairs, 2.77% purchased |
| **Models** | V1 logistic regression (binary view signal) vs V2 XGBoost (10 engagement features) |
| **Headline** | Test PR-AUC **0.093 → 0.613**, ROC-AUC **0.678 → 0.912** |
| **What drives it** | The *combination* of engagement features and a nonlinear model. Either alone gets only part of the way |
| **Caveat** | Scores are optimistic: features use each pair's full history, including the purchase day |
| **Stack** | Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn (notebook), Streamlit + Plotly (dashboard) |

### Short description (for a portfolio card or LinkedIn)

Modeled purchase intent on a 10,000-user sample of Alibaba's 100M-event Taobao dataset. Engineered engagement-depth features (repeat visits, funnel depth, recency) and showed, with a controlled model-vs-features comparison and ablations, where the predictive gain actually comes from. Along the way I found and fixed two data bugs that standard validation had missed, and I documented the leakage that makes the absolute scores optimistic.

### Resume bullets

- Modeled purchase intent on a 10,000-user sample (644K user–item pairs, 2.8% positive) of Alibaba's 100M-event Taobao dataset. XGBoost on engineered engagement features raised test PR-AUC from 0.093 (logistic regression baseline) to 0.613.
- Separated model effects from feature effects with a 2×2 cross-validated comparison: engagement features alone reached 0.22 PR-AUC, the nonlinear model alone 0.14, and both together 0.60. Ablations showed no single feature is essential.
- Found two data bugs that standard checks missed:
  - 582 out-of-window timestamps had pushed a recency feature to an impossible 126 days.
  - A UTC vs Beijing-time mismatch had shifted hourly patterns by 8 hours, and a filter built on it had dropped 11,764 valid events.
- Chose the classification cutoff on cross-validated training predictions, not the test set. Applied once to test, it doubled F1 from 0.28 to 0.59.
- Deployed a 4-tab [Streamlit + Plotly dashboard](https://taobao-purchase-intent-nporuhbvrvk4wor5dcjmdn.streamlit.app/) presenting the EDA, features, model comparisons and limitations.

---

## The problem

Most engagement measurement is binary: did someone view or click? That treats very different behavior as identical. One person glances at a product once. Another views it on three separate days, favorites it and adds it to their cart. A model that only sees "viewed: yes" can't tell them apart.

**Question:** if a model sees *how* a user engaged with an item (how deeply, how often, how recently), how much better can it predict whether they'll buy it?

---

## What I did, step by step

### 1. Loading the data on a laptop
- I downloaded the data with `kagglehub`, which caches it so later runs skip the download.
- I loaded the 100M rows with compact types (`int32` for IDs and timestamps, `category` for the behavior column) to fit within 8 GB of RAM.
- **I sampled 10,000 users, not random rows** (seed 42). Sampling by user keeps each person's *complete* history, which matters because every feature describes a user's full relationship with an item.
- I deleted the full table from memory once the sample was taken.

### 2. Cleaning: two bugs the standard checks missed
- Basic checks passed: 0 nulls, 0 duplicates, 0 invalid behavior types.
- **Out-of-window timestamps.** 582 events fell outside the documented 9-day window, some as early as August. The original check only flagged dates before 2000 or after 2025. These rows are what pushed `days_since_last` to 126 days in a 9-day dataset. I removed them.
- **Timezone.** The timestamps are in Beijing time (UTC+8). I spotted it when hourly activity jumped from about 100 to 3,722 events exactly at midnight Beijing time. Treating them as UTC caused two problems:
  - it shifted every hour-of-day pattern by 8 hours, and
  - a date filter in UTC threw away 11,764 real events from the first night.

  The notebook converts timestamps to Beijing time first.
- **Outlier users.** I used the IQR rule on events per user (upper bound about 282 events) and removed 471 users. I removed whole users, so no one's history is left half-deleted. The trade-off: this may also remove genuinely heavy shoppers.

### 3. Exploring the data
- **Funnel:** 748,690 views → 47,854 cart adds (6.4%) → 18,604 purchases (2.5%).
- **Users vs items:** 67% of users bought *something* during the 9 days, but only 2.8% of user–item pairs ended in a purchase. The problem is "which item?", not "which user?".
- **Categories:** conversion ranges from about 6% to 11% among categories with at least 500 views. The floor avoids noisy small categories. This variation is why both models get a category control feature.
- **Time:** shopping peaks at 21:00 Beijing time.
- **A chart trap:** the 9-day window has two Saturdays and two Sundays but one of each weekday, so a raw day-of-week chart makes weekends look about twice as busy. Daily totals are the honest view.

### 4. Reshaping: one row per user–item pair
The question is "will *this user* buy *this item*?", so I grouped the event log into one row per user–item pair. The target is whether that pair includes a purchase.

### 5. Feature engineering

| Feature | What it captures | Key decision |
|---|---|---|
| `view_count`, `fav_count` | How many times viewed / favorited | Plain counts |
| `has_view`, `has_fav`, `has_cart` | Whether each action happened | V1 uses only `has_view` |
| `depth_score` | view×1 + favorite×2 + cart×3 | Weights rise with commitment. Purchases are excluded, since counting them would leak the answer |
| `session_count` | Distinct *days* engaged with the item | No session IDs in the data, so a calendar day stands in for a session |
| `category_avg_depth` | Average depth in the item's category | A control in both models, **computed from training data only** |
| `days_since_last` | Days from the last interaction to the end of the data | Fractional days |
| `engagement_span_days` | Days between first and last interaction | 0 if all on one day |
| `funnel_steps` | Distinct pre-purchase stages reached | Purchases excluded |

### 6. Modeling
- **Split:** 80/20 *stratified* train/test (seed 42), so both halves keep the 2.77% purchase rate.
- **V1:** logistic regression on `has_view` + the category control, with balanced class weights.
- **V2:** XGBoost on all 10 features, with `scale_pos_weight` ≈ 35 for the imbalance.
- **Metric: PR-AUC.** With 97% non-buyers, accuracy is meaningless and ROC-AUC flatters the model. PR-AUC measures how well it finds the rare buyers. Random guessing scores 0.028.
- **Tuning:** a grid search over 54 combinations with 3-fold CV on PR-AUC. The best was 300 trees, depth 6, learning rate 0.05.
- **Test-set discipline:** the test set is used only for the final reported numbers. Every comparison and every choice (feature sets, ablations, cutoff) uses cross-validation on the training set.

### 7. Where does the gain come from?
V1 → V2 changes the model *and* the features at once, so I tested every combination (3-fold CV PR-AUC on the training set):

| | 2 view features | 10 engagement features |
|---|---|---|
| Logistic regression | 0.083 | 0.220 |
| XGBoost | 0.142 | **0.602** |

The features help on their own, and so does the model. But the big jump needs **both**: a nonlinear model that can combine the engagement signals.

### 8. Which features does the model actually need?
Feature importance says `session_count` (61%) and `depth_score` (22%) dominate. Importance only shows what the model *used*, so I ran ablations: retrain without a feature and measure the loss.

| Removed | CV PR-AUC change |
|---|---|
| `session_count` | −0.021 |
| `depth_score` | −0.003 |
| both | −0.023 |
| the 4 features I added in round 2 | −0.002 (noise) |

**No single feature is essential.** The engagement features overlap heavily, so the model rebuilds the missing signal from the others. The four features I added later contribute nothing measurable, and I report that as a negative result.

### 9. Choosing a cutoff without touching the test set
Class weighting pushes V2's scores up, so 0.5 is a poor cutoff. I took cross-validated (out-of-fold) predictions on the training set and picked the cutoff that maximizes F1 there: **0.92**. Then I applied it once to the test set:

| On the test set | Cutoff 0.5 | Cutoff 0.92 |
|---|---|---|
| Precision | 17% | 79% |
| Recall | 79% | 47% |
| F1 | 0.28 | **0.59** |

A cheap action like a reminder email suits a lower cutoff. An expensive one like a discount suits a higher one.

### 10. Train vs test
V2 scores 0.628 PR-AUC on train, 0.613 on test and 0.602 in cross-validation. The model isn't memorizing its training data. A close train/test match **can't** rule out leakage here, though, because the random split puts the same users and days on both sides.

---

## Results

| | V1 baseline | V2 | Change |
|---|---|---|---|
| ROC-AUC (test) | 0.678 | **0.912** | +34% |
| PR-AUC (test) | 0.093 | **0.613** | ≈6.6× |
| Buyers caught at cutoff 0.5 | 1,271 of 3,568 | **2,818 of 3,568** | 2.2× |

**An example of the signal:** among pairs with exactly 5 views, those spread over 3+ days convert at 25.3%, against 13.0% when all 5 views happen on one day (1,080 and 818 pairs).

> **The signal isn't presence — it's persistence.** Repeat visits are the model's most-used signal. The ablation shows the same information is also spread across the other engagement features.

---

## Limitations (I lead with these in interviews)

1. **The scores are optimistic because of target leakage.** Features use each pair's whole history, including the purchase day and views after buying. This probably helps V2 more than V1, so it may also inflate the gap between them. The fix: build features from days 1–7 only and predict purchases on days 8–9.
2. **Random split, not time-based.** The same users and days appear in train and test.
3. **A small sample.** 10,000 users (under 1% of the data), one random seed, with the heaviest users removed.
4. **Uncalibrated scores.** The outputs work for ranking but aren't probabilities, so the cutoff has to be chosen deliberately.

---

## Likely interview questions

**Why PR-AUC instead of accuracy?** Only 2.8% of pairs are purchases. Predicting "no" every time gives 97% accuracy.

**Is the V1 → V2 gain from the model or the features?** Both. The 2×2 comparison shows features alone reach 0.22, the model alone 0.14, and both together 0.60.

**Why does `session_count` have 61% importance but cost only 0.021 when removed?** Importance credits whichever overlapping feature the trees split on. Ablation measures what's lost, and the other engagement features carry most of the same signal.

**How did you choose the cutoff?** On cross-validated predictions from the training set. Then I applied it once to test, so the reported numbers are an honest estimate.

**Is 0.91 ROC-AUC realistic?** Not for real-time prediction, because of leakage. The *relative* comparisons are fair, since all models share the same setup. The absolute numbers would drop with a time-based design.

**Why sample users instead of rows?** Memory, plus sampling by user keeps each user's history intact.

**What was the hardest bug?** The timezone. Every standard check passed, but hourly activity jumped exactly at midnight Beijing time.

**What would you do next?** A leakage-free time split, calibration, repeated samples for confidence intervals, and time-of-day features.

---

## The dashboard

The Streamlit app has four tabs:
- **EDA:** stat tiles, the engagement funnel, and purchase rate by depth and by repeat visits.
- **Features:** a feature dictionary, importance, and the 5-views example.
- **Model Results:** metrics, confusion matrices, ROC curves, train vs test, the chosen cutoff, model vs features, and the ablation.
- **Insights:** the core finding, an ad-tech analogy, and limitations.

It doesn't retrain anything:
- Data-driven charts are computed from the engineered dataset.
- Model numbers come from the final notebook run.
- ROC curves are approximated from the real AUC values. The notebook has the exact curves.

**Design choices:**
- A colorblind-safe blue/orange pair for V1 vs V2.
- The 100%-by-definition "buy-only" bar is hidden by default, so it doesn't flatten the rest of the chart.
- Log-scaled colors on the confusion matrices, so the small buyer cells stay visible.
- Hover tooltips with pair counts, so viewers can tell a well-supported rate from a small bucket.

---

## Data credit

The Taobao UserBehavior dataset was released by Alibaba on [Tianchi](https://tianchi.aliyun.com/dataset/649). I used the [Kaggle mirror](https://www.kaggle.com/datasets/marwa80/userbehavior). Licence terms are still to be confirmed on the Tianchi page; see the [README](README.md#data-source-and-licence).
