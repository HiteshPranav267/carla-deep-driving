import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')
_LOG_PATH = os.path.join(_MODELS_DIR, 'training_log.json')

def main():
    if not os.path.exists(_LOG_PATH):
        print("No log found.")
        return

    with open(_LOG_PATH, 'r') as f:
        log = json.load(f)

    models = ['baseline_cnn', 'cnn_gru', 'gru_only']
    out_data = {}

    for m in models:
        mses = []
        tmaes = []
        smaes = []
        bmaes = []
        times = []
        for i in range(3):
            key = f"{m}_est{i}"
            if key in log and len(log[key]) > 0:
                best_epoch = min(log[key], key=lambda x: x['val_loss'])
                mses.append(best_epoch['val_loss'])
                tmaes.append(best_epoch['val_thr_mae'])
                smaes.append(best_epoch['val_str_mae'])
                bmaes.append(best_epoch['val_brk_mae'])
                times.append(sum(e.get('time_s', 0) for e in log[key]) / 60)
        
        if len(mses) > 0:
            avg_mse = sum(mses) / len(mses)
            avg_tmae = sum(tmaes) / len(tmaes)
            avg_smae = sum(smaes) / len(smaes)
            avg_bmae = sum(bmaes) / len(bmaes)
            avg_time = sum(times) / len(times)
            out_data[m] = {
                "mse": avg_mse,
                "tmae": avg_tmae,
                "smae": avg_smae,
                "bmae": avg_bmae,
                "time": avg_time
            }

    with open('parsed_metrics.json', 'w') as f:
        json.dump(out_data, f, indent=2)

if __name__ == "__main__":
    main()
