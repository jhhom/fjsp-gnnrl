import torch

from ppo import PPO
from params import config, TRAINING_ONESHOT, TRAINING_RESUME, TRAINING_SAVE

def save_progress(training_log, validation_log, validation_result, record, model: PPO):
    path = config.progress_config.path_to_save_progress

    validation_log.append(validation_result)

    saved = {
        "model_state_dict": model.policy.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        "validation_log": validation_log,
        "training_log": training_log,
        "best_record": record
    }

    with open(f'{path}/validation_log.txt', 'w') as logfile:
        logfile.write(str(validation_log))

    with open(f'{path}/training_log.txt', 'w') as logfile:
        logfile.write(str(training_log))

    if validation_result < record:
        saved['best_record'] = validation_result
        torch.save(model.policy.state_dict(), f'{path}/best_weight.pth')

    if config.progress_config.training_mode == TRAINING_SAVE or config.progress_config.training_mode == TRAINING_RESUME:
        torch.save(saved, f'{path}/saved.pth')

'''
Notes:

The folder name will be MK01 / MK02
The subfolder name you can determine yourself


keep a fixed format

Suggestion:

<Experiment_ID>

Things to save

1. a serialized config object in the folder.
2. training log
3. validation log
4. best_weight
5. last_weight
6. last_optimizer_weights


'''