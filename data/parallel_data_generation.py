import os.path
import subprocess
from datetime import datetime

import click

from data.data_generation import data_generation_options


@click.command()
@click.option("-s", "--number-of-sequences", type=int, required=True,
              help="Number of sequences that shall be generated")
@click.option("-p", "--number-of-processes", type=int, default=8,
              help="Number of parallel processes that shall generate the data")
@click.option("--root-dir", type=str, default="datasets/gui_env",
              help="In this directory, subfolders are automatically created based on time to store the generated data")
@data_generation_options
def main(number_of_sequences: int, number_of_processes: int, root_dir: str,
         stop_mode: str, amount: int, monkey_type: str, random_click_prob: float, log: bool, html_report: bool):

    python_commands = ["python", "data/data_generation.py"]

    if stop_mode == "time":
        python_commands.append("-t")
    else:
        python_commands.append("-i")

    python_commands.append(f"--amount={amount}")
    python_commands.append(f"--monkey-type={monkey_type}")
    if random_click_prob is not None:
        python_commands.append(f"--random-click-prob={random_click_prob}")
    if not log:
        python_commands.append(f"--no-log")
    if not html_report:
        python_commands.append(f"--no-html-report")

    xvfb_command = ["xvfb-run", "-s", "-screen 0 448x448x24"]

    base_dir = os.path.join(root_dir, monkey_type, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    commands = []
    for i in range(number_of_sequences):
        current_dir = os.path.join(base_dir, f"{i}")
        current_python_command = python_commands
        current_python_command.append(f"--directory={current_dir}")

        current_xvfb_command = xvfb_command + [f"--server-num={99 + i}"]
        command = current_xvfb_command + current_python_command

        commands.append(command)

    processes = []
    for p_id, command in enumerate(commands):
        if len(processes) >= number_of_processes:
            for i, p in enumerate(processes):
                p.wait()
                print(f"Finished process {i}")
                processes.remove(p)

        p = subprocess.Popen(command)
        processes.append(p)
        print(f"Started {p_id}")

    for i, p in enumerate(processes):
        p.wait()
        print(f"Finished process {i}")

    print("Finished all processes, exiting.")


if __name__ == "__main__":
    main()
