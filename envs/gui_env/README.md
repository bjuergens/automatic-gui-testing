# Required packages

- gym
- numpy
- PySide6
- Coverage.py
- Install with `pip install coverage pyside6 numpy gym`

# Start the application (without OpenAI Gym)

- In the root folder run `PYTHONPATH=$(pwd) python envs/gui_env/src/main_window.py`

# Start as an OpenAI Gym environment

- You need to import the environment and then use it like a normal gym environment
- To enable generating a html report of the coverage, use `generate_html_report=True` as a parameter in the constructor
- There are three environments:
    * `GUIEnv` requires x and y coordinates as the action
    * `GUIEnvRandomWidget` does not require an action, it selects either a random widget for a click or a random click
       itself
    * `GUIEnvRandomClick` does not require an action, it always chooses a random click (so random coordinates)
- Example code for a random monkey tester:
```python
import time

from envs.gui_env.gui_env import GUIEnvRandomClick


def main():
    env = GUIEnvRandomClick(generate_html_report=True)
    ob = env.reset()
    
    rew_sum = 0
    
    start_time = time.time()
    timeout = 3600  # Run for an hour
    i = 0
    while time.time() < start_time + timeout:
        rew, ob, done, info = env.step()
        rew_sum += rew
        
        if i % 500 == 0:
            print(f"{i}: Current reward '{rew_sum}', time remaining '{start_time + timeout - time.time():.0f}'")
    
        i += 1
    
    env.close()


if __name__ == '__main__':
    main()

```

# Headless setup

The environments can also be started on a headless server or on your local machine with a virtual desktop. This has the
advantage of not interfering with something else on your desktop. To do this, install `xvfb`, for example with
`sudo apt install xvfb` and then run the following command in the root directory of this project, where
`monkey_tester.py` contains the example script from above:

`PYTHONPATH=$(pwd) xvfb-run -s "-screen 0 448x448x24" python monkey_tester.py`