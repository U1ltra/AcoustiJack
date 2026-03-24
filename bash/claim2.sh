cd /root/AcoustiJack
nohup bash -c '

GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_dasiam_uav2 --world_file field --gim_max_speed 1.0 4.0 1.0 --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_dasiam_uav2 --world_file warehouse --gim_max_speed 1.0 4.0 1.0 --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200

GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_ucmc_uav2 --world_file field --gim_max_speed 1.0 4.0 1.0 --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_ucmc_uav2 --world_file warehouse --gim_max_speed 1.0 4.0 1.0 --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200

echo "All experiments completed!"
' > all_experiments.log 2>&1 &