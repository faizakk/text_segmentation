import argparse
import subprocess
import os
import multiprocessing
import pickle
import csv
from twilio.rest import Client
import json
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))

import rationale_net.utils.parsing as parsing

EXPERIMENT_CRASH_MSG = "ALERT! job:[{}] has crashed! Check logfile at:[{}]"
CONFIG_NOT_FOUND_MSG = "ALERT! {} config {} file does not exist!"
RESULTS_PATH_APPEAR_ERR = 'results_path should not appear in config. It will be determined automatically per job'
SUCESSFUL_SEARCH_STR = "SUCCESS! Grid search results dumped to {}. Best dev loss: {},  dev accuracy: {:.3f}"

RESULT_KEY_STEMS = ['{}_loss', '{}_obj_loss', '{}_k_selection_loss',
        '{}_k_continuity_loss','{}_metric']

LOG_KEYS = ['results_path', 'model_path', 'log_path']
SORT_KEY = 'dev_loss'

parser = argparse.ArgumentParser(description='OncoNet Grid Search Dispatcher. For use information, see `doc/README.md`')
parser.add_argument("--experiment_config_path", required=True, type=str, help="Path of experiment config")
parser.add_argument("--alert_config_path", type=str, default='configs/alert_config.json', help="Path of alert config")
parser.add_argument('--log_dir', type=str, default="logs", help="path to store logs and detailed job level result files")
parser.add_argument('--result_path', type=str, default="results/grid_search.csv", help="path to store grid_search table. This is preferably on shared storage")
parser.add_argument('--rerun_experiments', action='store_true', default=False, help='whether to rerun experiments with the same result file location')


def send_text_msg(msg, alert_config, twilio_config):
    '''
    Send a text message using twilio acct specified twilio conf to numbers
    specified in alert_conf.
    If suppress_alerts is turned on, do nothing
    :msg: - body of text message
    :alert_config: - dictionary with a list fo numbers to send message to
    :twilio-config: - dictionary with twilio SID, TOKEN, and phone number
    '''
    if alert_config['suppress_alerts']:
        return
    client = Client(twilio_config['ACCOUNT_SID'], twilio_config['AUTH_TOKEN'])
    for number in [alert_config['alert_nums']]:
        client.messages.create(
            to=number, from_=twilio_config['twilio_num'], body=msg)


def launch_experiment(gpu, flag_string, alert_conf, twilio_conf):
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    Alert of something goes wrong.
    :gpu: gpu to run this machine on.
    :flag_string: flags to use for this model run. Will be fed into
    scripts/main.py
    '''
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_name = parsing.md5(flag_string)
    log_stem = os.path.join(args.log_dir, log_name)
    log_path = '{}.txt'.format(log_stem)
    results_path = "{}.results".format(log_stem)

    experiment_string = "CUDA_VISIBLE_DEVICES={} python -u scripts/main.py {} --results_path {}".format(
        gpu, flag_string, results_path)

    # forward logs to logfile
    shell_cmd = "{} > {} 2>&1".format(experiment_string, log_path)
    print("Lauched exp: {}".format(shell_cmd))
    if not os.path.exists(results_path) or args.rerun_experiments:
        subprocess.call(shell_cmd, shell=True)

    if not os.path.exists(results_path):
        # running this process failed, alert me
        job_fail_msg = EXPERIMENT_CRASH_MSG.format(experiment_string, log_path)
        send_text_msg(job_fail_msg, alert_conf, twilio_conf)

    return results_path, log_path


def worker(gpu, job_queue, done_queue, alert_config, twilio_config):
    '''
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.
    :gpu - gpu this worker can access.
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(gpu, params, alert_config, twilio_config))


if __name__ == "__main__":

    args = parser.parse_args()
    if not os.path.exists(args.experiment_config_path):
        print(CONFIG_NOT_FOUND_MSG.format("experiment", args.experiment_config_path))
        sys.exit(1)
    experiment_config = json.load(open(args.experiment_config_path, 'r'))

    if 'results_path' in experiment_config['search_space']:
        print (RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)

    if not os.path.exists(args.alert_config_path):
        print(CONFIG_NOT_FOUND_MSG.format("alert", args.alert_config_path))
        sys.exit(1)
    alert_config = json.load(open(args.alert_config_path, 'r'))

    twilio_conf_path = alert_config['path_to_twilio_secret']
    if not os.path.exists(twilio_conf_path):
        print(CONFIG_NOT_FOUND_MSG.format("twilio", twilio_conf_path))

    twilio_config = None
    if not alert_config['suppress_alerts']:
        twilio_config = json.load(open(twilio_conf_path, 'r'))

    job_list, experiment_axies = parsing.parse_dispatcher_config(experiment_config)
    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for job in job_list:
        job_queue.put(job)
    print("Launching Dispatcher with {} jobs!".format(len(job_list)))
    print()
    for gpu in experiment_config['available_gpus']:
        print("Start gpu worker {}".format(gpu))
        multiprocessing.Process(target=worker, args=(gpu, job_queue, done_queue, alert_config, twilio_config)).start()
    print()

    summary = []
    result_keys = []
    for mode in ['train','dev','test']:
        result_keys.extend( [k.format(mode) for k in RESULT_KEY_STEMS ])
    for _ in range(len(job_list)):
        result_path, log_path = done_queue.get()
        assert result_path is not None
        try:
            result_dict = pickle.load(open(result_path, 'rb'))
        except Exception as e:
            print("Experiment failed! Logs are located at: {}".format(log_path))
            continue

        result_dict['log_path'] = log_path
        # Get results from best epoch and move to top level of results dict
        best_epoch_indx = result_dict['epoch_stats']['best_epoch']
        present_result_keys = []
        for k in result_keys:
            if (k in result_dict['test_stats'] and len(result_dict['test_stats'][k])>0) or (k in result_dict['epoch_stats'] and len(result_dict['epoch_stats'][k])>0):
                present_result_keys.append(k)
                if 'test' in k:
                    result_dict[k] = result_dict['test_stats'][k][0]
                else:
                    result_dict[k] = result_dict['epoch_stats'][k][best_epoch_indx]


        summary_columns = experiment_axies + present_result_keys + LOG_KEYS
        # Only export keys we want to see in sheet to csv
        summary_dict = {}
        for key in summary_columns:
            summary_dict[key] = result_dict[key]
        summary.append(summary_dict)
    summary = sorted(summary, key=lambda k: k[SORT_KEY])

    dump_result_string = SUCESSFUL_SEARCH_STR.format(
        args.result_path, summary[0]['dev_loss'], summary[0]['dev_metric']
    )
    # Write summary to csv
    with open(args.result_path, 'w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=summary_columns)
        writer.writeheader()
        for experiment in summary:
            writer.writerow(experiment)

    print(dump_result_string)
    send_text_msg(dump_result_string, alert_config, twilio_config)
