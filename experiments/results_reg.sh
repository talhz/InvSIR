export PYTHONPATH=.

echo "IRM_reg_example1_setting1:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=10 \
  --l2_regularizer_weight=0.001 \
  --lr=0.001 \
  --penalty_anneal_iters=250 \
  --penalty_weight=100000 \
  --steps=501 \
  --example_setting=11

echo "ERM_reg_example1_setting1:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=10 \
  --l2_regularizer_weight=0 \
  --lr=0.001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=501 \
  --example_setting=11

echo "IRM_reg_example1_setting2:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=10 \
  --l2_regularizer_weight=0.001 \
  --lr=0.001 \
  --penalty_anneal_iters=250 \
  --penalty_weight=100000 \
  --steps=501 \
  --example_setting=12

echo "ERM_reg_example1_setting2:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=10 \
  --l2_regularizer_weight=0 \
  --lr=0.001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=501 \
  --example_setting=12

echo "IRM_reg_example1_setting3:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=10 \
  --l2_regularizer_weight=0.001 \
  --lr=0.001 \
  --penalty_anneal_iters=250 \
  --penalty_weight=100000 \
  --steps=501 \
  --example_setting=13

echo "ERM_reg_example1_setting3:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=10 \
  --l2_regularizer_weight=0 \
  --lr=0.001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=501 \
  --example_setting=13

echo "IRM_reg_example2_setting1:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=0 \
  --l2_regularizer_weight=0.001 \
  --lr=0.001 \
  --penalty_anneal_iters=250 \
  --penalty_weight=100000 \
  --steps=501 \
  --example_setting=21

echo "ERM_reg_example2_setting1:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=0 \
  --l2_regularizer_weight=0 \
  --lr=0.001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=501 \
  --example_setting=21

echo "IRM_reg_example2_setting2:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=0 \
  --l2_regularizer_weight=0.001 \
  --lr=0.001 \
  --penalty_anneal_iters=250 \
  --penalty_weight=100000 \
  --steps=501 \
  --example_setting=22

echo "ERM_reg_example2_setting2:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=0 \
  --l2_regularizer_weight=0 \
  --lr=0.001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=501 \
  --example_setting=22

echo "IRM_reg_example3_setting1:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=0 \
  --l2_regularizer_weight=0.001 \
  --lr=0.001 \
  --penalty_anneal_iters=250 \
  --penalty_weight=100000 \
  --steps=501 \
  --example_setting=31

echo "ERM_reg_example3_setting1:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=0 \
  --l2_regularizer_weight=0 \
  --lr=0.001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=501 \
  --example_setting=31

echo "IRM_reg_example3_setting2:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=0 \
  --l2_regularizer_weight=0.001 \
  --lr=0.001 \
  --penalty_anneal_iters=250 \
  --penalty_weight=100000 \
  --steps=501 \
  --example_setting=32


echo "ERM_reg_example3_setting2:"
python -u experiments/IRM_reg.py \
  --hidden_dim=64 \
  --input_size=10 \
  --spurious_size=0 \
  --l2_regularizer_weight=0 \
  --lr=0.001 \
  --penalty_anneal_iters=0 \
  --penalty_weight=0.0 \
  --steps=501 \
  --example_setting=32
