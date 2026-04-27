cd /root/AcoustiJack
source env.sh

# ===== Setting 1: HighEndDrone =====

# SiamRPN
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_siam --world_file urban --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_siam --world_file field --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_siam --world_file warehouse --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_siam --world_file raceway --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_siam --world_file urban_car --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60

# DaSiamRPN
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_dasiam --world_file urban --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_dasiam --world_file field --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_dasiam --world_file warehouse --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_dasiam --world_file raceway --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_dasiam --world_file urban_car --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60

# KCF
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_kcf --world_file urban --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_kcf --world_file field --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_kcf --world_file warehouse --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_kcf --world_file raceway --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_kcf --world_file urban_car --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60

# SORT
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_sort --world_file urban --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_sort --world_file field --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_sort --world_file warehouse --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_sort --world_file raceway --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_sort --world_file urban_car --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60

# UCMCTrack
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_ucmc --world_file urban --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_ucmc --world_file field --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_ucmc --world_file warehouse --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_ucmc --world_file raceway --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_ucmc --world_file urban_car --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200 --atker_angle 60


# ===== Setting 2: MidEndDrone =====

# SiamRPN
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_siam_uav2 --world_file urban --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_siam_uav2 --world_file field --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_siam_uav2 --world_file warehouse --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_siam_uav2 --world_file raceway --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_siam_uav2 --world_file urban_car --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200

# DaSiamRPN
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_dasiam_uav2 --world_file urban --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_dasiam_uav2 --world_file field --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_dasiam_uav2 --world_file warehouse --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_dasiam_uav2 --world_file raceway --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_dasiam_uav2 --world_file urban_car --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200

# KCF
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_kcf_uav2 --world_file urban --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_kcf_uav2 --world_file field --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_kcf_uav2 --world_file warehouse --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_kcf_uav2 --world_file raceway --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_kcf_uav2 --world_file urban_car --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type KCF --surrogate_model KCF --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200

# SORT
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_sort_uav2 --world_file urban --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_sort_uav2 --world_file field --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_sort_uav2 --world_file warehouse --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_sort_uav2 --world_file raceway --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_sort_uav2 --world_file urban_car --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200

# UCMCTrack
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_ucmc_uav2 --world_file urban --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_ucmc_uav2 --world_file field --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_ucmc_uav2 --world_file warehouse --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_raceway_ucmc_uav2 --world_file raceway --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name car_urban_ucmc_uav2 --world_file urban_car --gim_max_speed 0.1 3.0 0.1 --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atker_angle 0 --atk_cycle 4 --atk_duration 30 --attack_interval_method simdt --timeout 200