import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite
from dm_control import viewer

# Choose a task:
# The DMC suite includes environments like `cartpole`, `cheetah`, `walker`, etc.
# Each environment has different tasks (e.g., 'swingup', 'run').
env = suite.load(domain_name="walker", task_name="run")

# Set the camera ID for rendering. Different environments may have different camera setups.
camera_id = 0

# Reset the environment to get the initial observation
obs = env.reset()

# print(obs)
# Extract pixel-based observation (rendered image from the environment)
pixels = env.physics.render(camera_id=camera_id, height=240, width=320)

# print(pixels, type(pixels))
# Display the pixels as an image using matplotlib
plt.imshow(pixels)
plt.title("Initial Observation (Pixel-Based)")
plt.axis('off')
plt.show()

i = 0
# Interaction Loop (e.g., 100 steps)
while True:   
 # Sample a random action from the action space
    action = np.random.uniform(low=-1.0, high=1.0, size=env.action_spec().shape)
    i += 1
    # Take a step in the environment
    time_step = env.step(action)
    
    print(time_step.last())
    # Render and display the current pixel observation
    pixels = env.physics.render(camera_id=camera_id, height=240, width=320)
    if time_step.last():
        print(i)
    #     break
    # plt.imshow(pixels)
    # plt.axis('off')
    # plt.pause(0.01)  # Small pause to allow image display

# If you want to launch an interactive viewer, you can use the built-in viewer from dm_control
# viewer.launch(env)