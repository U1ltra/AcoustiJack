cd /root/AcoustiJack
source env.sh

GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_dasiam --world_file field --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_ucmc --world_file field --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200