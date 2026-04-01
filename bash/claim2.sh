cd /root/AcoustiJack
source env.sh

GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_dasiam_uav2 --world_file field --gim_max_speed 1.0 4.0 1.0 --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_ucmc_uav2 --world_file field --gim_max_speed 1.0 4.0 1.0 --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atker_angle 0 --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_race_ucmc_uav2 --world_file raceway --gim_max_speed 1.0 4.0 1.0 --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atker_angle 0 --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200