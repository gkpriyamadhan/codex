import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
inter = pd.read_csv("interactions.csv")

# 1. DATE PARSING
for df in [train, test, inter]:
    df['service_date'] = pd.to_datetime(df['service_date'], dayfirst=True)

train = train.sort_values('service_date').reset_index(drop=True)

# 2. BASIC FEATURE ENGINEERING
train['route_id'] = train['origin_hub_id'].astype(str) + "_" + train['destination_hub_id'].astype(str)
test['route_id']  = test['origin_hub_id'].astype(str) + "_" + test['destination_hub_id'].astype(str)

train['dow'] = train['service_date'].dt.dayofweek
test['dow']  = test['service_date'].dt.dayofweek

# 3. TRAIN / VALIDATION SPLIT
split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx]

# 4. ROUTE + WEEKDAY ANCHOR
print("Building anchor model...")
anchor_map = (
    train_part
    .groupby(['route_id', 'dow'])['final_service_units']
    .median()
    .to_dict()
)

global_median = train_part['final_service_units'].median()

def get_anchor(row):
    return anchor_map.get((row['route_id'], row['dow']), global_median)

train['anchor_pred'] = train.apply(get_anchor, axis=1)
test['anchor_pred']  = test.apply(get_anchor, axis=1)

# 5. MULTI-DAY INTERACTION SIGNAL (≤ 15 days)
signal = (
    inter[inter['days_before_service'] <= 15]
    .groupby(['service_date','origin_hub_id','destination_hub_id'])
    .agg({'cumulative_commitments':'max'})
    .reset_index()
)

train = train.merge(
    signal, 
    on=['service_date','origin_hub_id','destination_hub_id'], 
    how='left'
).fillna(0)

test = test.merge(
    signal, 
    on=['service_date','origin_hub_id','destination_hub_id'], 
    how='left'
).fillna(0)

# 6. ROUTE HISTORY FEATURES
route_stats = (
    train_part
    .groupby('route_id')['final_service_units']
    .agg(route_mean='mean', route_median='median')
    .reset_index()
)

train = train.merge(route_stats, on='route_id', how='left')
test  = test.merge(route_stats, on='route_id', how='left')

# 7. RESIDUAL TARGET
train['residual'] = train['final_service_units'] - train['anchor_pred']

features = [
    'anchor_pred',
    'cumulative_commitments',
    'route_mean',
    'route_median',
    'dow'
]

# 8. EXTRA TREES ADJUSTER
adjuster = ExtraTreesRegressor(
    n_estimators=600,
    max_depth=14,
    min_samples_leaf=8,
    max_features=0.7,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

print("Training residual model...")
adjuster.fit(
    train[features].iloc[:split_idx],
    train['residual'].iloc[:split_idx]
)

# 9. VALIDATION
val_adj = adjuster.predict(train[features].iloc[split_idx:])
val_preds = train['anchor_pred'].iloc[split_idx:] + val_adj

cap = train_part['final_service_units'].quantile(0.98)
val_preds = np.clip(val_preds, 0, cap)

mae = mean_absolute_error(
    train['final_service_units'].iloc[split_idx:],
    val_preds
)

print("\n--- TEAM CODEX PERFORMANCE REPORT ---")
print(f"VALIDATION MAE: {mae:.2f}")
print("------------------------------------\n")

# 10. FINAL TRAINING ON FULL DATA
adjuster.fit(train[features], train['residual'])

test_adj = adjuster.predict(test[features])
test_final = test['anchor_pred'] + test_adj
test_final = np.maximum(0, np.round(test_final)).astype(int)

# 11. SUBMISSION FILE
submission = pd.DataFrame({
    'final_service_units': test_final
})

submission.to_csv("codex_hybrid_final.csv", index=False)

print("SUCCESS")