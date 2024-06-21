poetry install -E "mujoco_py pybullet"
poetry run python -c "import mujoco_py"
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids Hopper-v2 \
    --command "poetry run python cleanrl/sac_continuous_action.py --track" \
    --num-seeds 1 \
    --workers 1