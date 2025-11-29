#!/bin/bash
# Complete workflow example for cyst classifier with feature caching

echo "=========================================="
echo "Cyst Classifier - Complete Workflow"
echo "=========================================="

# Step 1: Create train/test split
echo -e "\n[1/6] Creating train/test split..."
python3 -c "
from cyst_classifier.data_utils import generate_train_test_split
import pandas as pd

df = pd.read_csv('all_scans.csv')
train_idx, test_idx = generate_train_test_split(df, test_size=0.2, random_state=42)
df.iloc[train_idx].to_csv('train_images.csv', index=False)
df.iloc[test_idx].to_csv('test_images.csv', index=False)

print(f'Train: {len(train_idx)} scans')
print(f'Test: {len(test_idx)} scans')
"

# Step 2: Extract features from training set
echo -e "\n[2/6] Extracting features from training set (this may take a while)..."
python3 -m cyst_classifier.extract_features_script \
    --data train_images.csv \
    --output features_train.csv \
    --min-voxels 10

# Step 3: Extract features from test set
echo -e "\n[3/6] Extracting features from test set..."
python3 -m cyst_classifier.extract_features_script \
    --data test_images.csv \
    --output features_test.csv \
    --min-voxels 10

# Step 4: Train logistic regression model
echo -e "\n[4/6] Training logistic regression model..."
python3 -m cyst_classifier.main train \
    --data features_train.csv \
    --model logistic \
    --output models/logistic_regression.pkl

# Step 5: Train decision tree model
echo -e "\n[5/6] Training decision tree model..."
python3 -m cyst_classifier.main train \
    --data features_train.csv \
    --model tree \
    --tree-depth 5 \
    --output models/decision_tree.pkl

# Step 6: Evaluate both models
echo -e "\n[6/6] Evaluating models..."

echo "  - Evaluating logistic regression..."
python3 -m cyst_classifier.main eval \
    --data features_test.csv \
    --model models/logistic_regression.pkl \
    --output-dir results/logistic_regression/

echo "  - Evaluating decision tree..."
python3 -m cyst_classifier.main eval \
    --data features_test.csv \
    --model models/decision_tree.pkl \
    --output-dir results/decision_tree/

echo -e "\n=========================================="
echo "Workflow complete!"
echo "=========================================="
echo "Results saved in:"
echo "  - results/logistic_regression/"
echo "  - results/decision_tree/"
echo ""
echo "Check these directories for:"
echo "  - metrics.txt (performance metrics)"
echo "  - roc_curve.png (ROC curve)"
echo "  - confusion_matrix.png (confusion matrix)"
echo "=========================================="

# Optional: Quick experimentation with different feature subsets
echo -e "\n[BONUS] Experimenting with simplified feature set..."
python3 -m cyst_classifier.main train \
    --data features_train.csv \
    --model logistic \
    --features mean_hu std_hu frac_below_20hu \
    --output models/simple_model.pkl

python3 -m cyst_classifier.main eval \
    --data features_test.csv \
    --model models/simple_model.pkl \
    --output-dir results/simple_model/

echo "Done! Simple model results in results/simple_model/"
