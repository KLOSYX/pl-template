from nni.experiment import Experiment

search_space = {
    "lr": {"_type": "loguniform", "_value": [0.00001, 0.1]},
    "batch_size": {"_type": "choice", "_value": [4, 8, 16, 32]},
    "weight_decay": {"_type": "uniform", "_value": [0.01, 0.1]},
    "dropout": {"_type": "uniform", "_value": [0.1, 0.5]}
}

if __name__ == '__main__':
    experiment = Experiment('local')
    experiment.config.experiment_name = 'hw1'
    experiment.config.search_space = search_space
    experiment.config.trial_command = "python main.py --use_nni --num_workers=2 --project_name=hw1 --log_dir=pl_log --batch_size=8 --num_workers=2 --gpus=1 --max_epochs=10 --lr=0.001 --monitor=val_f1_score --data_path=data/raw/hw1_data --no_early_stop"
    experiment.config.trial_code_directory = '.'
    experiment.config.experiment_working_directory = f'./pl_log/{experiment.config.experiment_name}/nni'
    experiment.config.tuner.name = 'Random'
    experiment.config.max_trial_number = 30
    experiment.config.trial_concurrency = 1
    experiment.config.max_experiment_duration = '1h'
    experiment.config.nni_manager_ip = 'local'
    
    experiment.run(8080)