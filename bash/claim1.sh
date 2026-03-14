cd /root/uav-attacks
nohup bash -c '

GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_siam --world_file field --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_siam --world_file urban --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_siam --world_file warehouse --realtime_factor 1.0 --tracker_type SiamRPN --surrogate_model SiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200

GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_dasiam --world_file field --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_dasiam --world_file urban --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_dasiam --world_file warehouse --realtime_factor 1.0 --tracker_type DaSiamRPN --surrogate_model DaSiamRPN --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200

GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_sort --world_file field --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_sort --world_file urban --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_sort --world_file warehouse --realtime_factor 1.0 --tracker_type SORT --surrogate_model SORT --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200

GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_field_ucmc --world_file field --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_urban_ucmc --world_file urban --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200
GZ_IP=127.0.0.1 python -u launch.py --num_exps 18 --exp_name ped_warehouse_ucmc --world_file warehouse --realtime_factor 1.0 --tracker_type UCMCTrack --surrogate_model UCMCTrack --atk_cycle 4 --atk_duration 60 --attack_interval_method simdt --timeout 200

echo "All experiments completed!"
' > all_experiments.log 2>&1 &