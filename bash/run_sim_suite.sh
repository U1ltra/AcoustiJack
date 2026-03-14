SESSION_NAME="simulation"
PY_ENV="${HOME}/Documents/pyVir/px4-venv-sysPy"

tmux new-session -d -s $SESSION_NAME -n 'Dashboard'

tmux new-window -t $SESSION_NAME:1 -n 'XRCE'
tmux send-keys -t $SESSION_NAME:1 'cd ~/Documents/github/Micro-XRCE-DDS-Agent' C-m
tmux send-keys -t $SESSION_NAME:1 'MicroXRCEAgent udp4 -p 8888' C-m


# tmux new-window -t $SESSION_NAME:2 -n 'QGC'
# tmux send-keys -t $SESSION_NAME:2 '~/Documents/QGroundControl.AppImage' C-m

tmux new-window -t $SESSION_NAME:3 -n 'PX4 GZ'
tmux send-keys -t $SESSION_NAME:3 'cd ~/PX4-Autopilot' C-m
tmux send-keys -t $SESSION_NAME:3 'source ~/Documents/pyVir/px4-venv-sysPy/bin/activate' C-m
tmux send-keys -t $SESSION_NAME:3 'GZ_IP=127.0.0.1 HEADLESS=1 PX4_GZ_WORLD=urban_car PX4_SYS_AUTOSTART=4002 PX4_GZ_MODEL_POSE="-75.0,-52.0,0.0,0.0,0.0,0.0" PX4_SIM_MODEL=x500_gimbal ./build/px4_sitl_default/bin/px4' C-m
# GZ_PARTITION=px4 

# Wait to avoid CUDA OOM Exceptions
# sleep 15

# tmux new-window -t $SESSION_NAME:5 -n 'Follow'
# tmux send-keys -t $SESSION_NAME:5 "source ${PY_ENV}/bin/activate" C-m
# tmux send-keys -t $SESSION_NAME:5 "cd ${HOME}/Documents/github/uav-attacks" C-m
# tmux send-keys -t $SESSION_NAME:5 "python3 object-follower.py" C-m

# tmux new-window -t $SESSION_NAME:6 -n 'Gimbal'
# tmux send-keys -t $SESSION_NAME:6 "source ${PY_ENV}/bin/activate" C-m
# tmux send-keys -t $SESSION_NAME:6 "cd ${HOME}/Documents/github/uav-attacks" C-m
# tmux send-keys -t $SESSION_NAME:6 "python3 gimbal-control.py" C-m

# tmux new-window -t $SESSION_NAME:7 -n 'Attack'
# tmux send-keys -t $SESSION_NAME:7 "source ${PY_ENV}/bin/activate" C-m
# tmux send-keys -t $SESSION_NAME:7 "cd ${HOME}/Documents/github/uav-attacks" C-m
# tmux send-keys -t $SESSION_NAME:7 "python3 attack/motionAtk.py" C-m

# tmux new-window -t $SESSION_NAME:8 -n 'Tracking'
# tmux send-keys -t $SESSION_NAME:8 "source ${PY_ENV}/bin/activate" C-m
# tmux send-keys -t $SESSION_NAME:8 "cd ${HOME}/Documents/github/uav-attacks" C-m
# tmux send-keys -t $SESSION_NAME:8 "python3 uav_mot.py" C-m

tmux select-window -t $SESSION_NAME:3
# tmux attach -t $SESSION_NAME
