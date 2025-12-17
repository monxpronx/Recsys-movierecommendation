import argparse
from recbole.quick_start import run_recbole

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--save_path', type=str, default='saved')

    args = parser.parse_args()

    run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_files=args.config,
        config_dict={
            'data_path': args.data_path,
            'checkpoint_dir': args.save_path
        }
    )

if __name__ == "__main__":
    main()
