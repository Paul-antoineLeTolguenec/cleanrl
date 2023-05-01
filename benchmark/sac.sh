poetry install -E "mujoco_py pybullet"
poetry run python -c "import mujoco_py"
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids Hopper-v3 \
    --command "poetry run python cleanrl/sac_continuous_action.py --track --wandb-project-name cleanRL --seed 3 --autotune False" \
    --num-seeds 1 \
    --workers 1