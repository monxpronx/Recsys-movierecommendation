import argparse
from recbole.quick_start import run_recbole

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    run_recbole(
        config_file_list=[args.config]
    )
if __name__ == "__main__":
    main()
