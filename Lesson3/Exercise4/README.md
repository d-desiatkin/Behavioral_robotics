In this exercise we must understand the importance of correct reward function.
We will work with two bullet environments, their key parameters are listed bellow.

register(id='HopperBulletEnv-v0',
         entry_point='pybullet_envs.gym_locomotion_envs:HopperBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='HalfCheetahBulletEnv-v0',
         entry_point='pybullet_envs.gym_locomotion_envs:HalfCheetahBulletEnv',
         max_episode_steps=1000,
         reward_threshold=3000.0)

As we see we should get reward value 2500 for fully evolved hopper robot, and 3000 for fully evolved halfcheetah robot.

Specially edited environment files was provided to us. Their contains modification that will allow to find solution with GA (Genetic Algorithms) instead of RL (Reinforcement Learning).

We was asked to explain why original functions is not fit for GA. My initial guess (I haven't looked at the code at all) is that they will create broad search space with strong attractors or rewrard local minimums. So our modifications will help to overcome this phenomena.

Now lets look on the code.

.........
.........
Now look at diff files.
.........

Lets look on training results.

## Results

### Hopper
![Graph of original hopper](https://github.com/d-desiatkin/Behavioral_robotics/blob/master/Lesson3/Exercise4/hopper/original.png)

![Graph of modified hopper](https://github.com/d-desiatkin/Behavioral_robotics/blob/master/Lesson3/Exercise4/hopper/modified.png)
