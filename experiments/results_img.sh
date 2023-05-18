export PYTHONPATH=.

echo "InvSIR_image:"
python -u experiments/InvSIR_img.py \
  --hidden_dim=256 \
  --lr=0.001 \
  --selected_features=256 \
  --slices=10 \
  --steps=501

echo "IRM_image:"
python -u experiments/IRM_img.py \
  --hidden_dim=256 \
  --l2_regularizer_weight=0.001 \
  --lr=0.001 \
  --penalty_anneal_iters=250 \
  --penalty_weight=100000 \
  --steps=501