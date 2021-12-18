import os.path
import subprocess
from datetime import datetime

import click

RANDOM_CLICK_MONKEY_TYPE = "random-clicks"
RANDOM_WIDGET_MONKEY_TYPE = "random-widgets"


@click.command()
@click.option("-p", "--number-of-processes", type=int, required=True,
              help="Number of parallel processes that shall generate the data")
@click.option("-t", "--time", "stop_mode", flag_value="time", default=True,
              help="Use elapsed time in seconds to stop the data generation")
@click.option("-i", "--iterations", "stop_mode", flag_value="iterations",
              help="Use the number of iterations to stop the data generation")
@click.option("--amount", type=int,
              help="Amount on how long the data generation shall run (seconds or number of iterations, depending on "
                   "the stop_mode")
@click.option("-m", "--monkey-type", default=RANDOM_WIDGET_MONKEY_TYPE,
              type=click.Choice([RANDOM_CLICK_MONKEY_TYPE, RANDOM_WIDGET_MONKEY_TYPE]),
              show_default=True, help="Choose which type of random monkey tester to use")
@click.option("--root-dir", type=str, default="datasets/gui_env",
              help="In this directory, subfolders are automatically created based on time to store the generated data")
def main(number_of_processes: int, stop_mode: str, amount: int, monkey_type: str, root_dir: str):

    python_commands = ["python", "data/data_generation.py"]

    if stop_mode == "time":
        python_commands.append("-t")
    else:
        python_commands.append("-i")

    python_commands.append(f"--amount={amount}")
    python_commands.append(f"--monkey-type={monkey_type}")

    xvfb_command = ["xvfb-run", "-s", "-screen 0 448x448x24"]

    base_dir = os.path.join(root_dir, monkey_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    processes = []
    for i in range(number_of_processes):
        current_dir = os.path.join(base_dir, f"{i}")
        current_python_command = python_commands
        current_python_command.append(f"--directory={current_dir}")

        current_xvfb_command = xvfb_command + [f"--server-num={99 + i}"]
        command = current_xvfb_command + current_python_command

        p = subprocess.Popen(command)
        processes.append(p)

        print(f"Started {i}")

    for i, p in enumerate(processes):
        p.wait()
        print(f"Finished process {i}")

    print("Finished all processes, exiting.")


if __name__ == "__main__":
    main()
