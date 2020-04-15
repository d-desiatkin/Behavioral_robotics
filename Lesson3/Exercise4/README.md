# Report

## Exercise 4 description

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
#### original
![Graph of original hopper](https://github.com/d-desiatkin/Behavioral_robotics/blob/master/Lesson3/Exercise4/hopper/original.png)
#### modified
![Graph of modified hopper](https://github.com/d-desiatkin/Behavioral_robotics/blob/master/Lesson3/Exercise4/hopper/modified.png)

### Halfcheetah
#### original
![Graph of original halfcheetah](https://github.com/d-desiatkin/Behavioral_robotics/blob/master/Lesson3/Exercise4/halfcheetah/original.png)
#### modified
![Graph of modified halfcheetah](https://github.com/d-desiatkin/Behavioral_robotics/blob/master/Lesson3/Exercise4/halfcheetah/modified.png)

## Discussion
If you will launch simmulations it seems that punishment for energy usage is too great for original reward of both robots. They stabilise themselves in downward position right before the jump, but not take any further actions. Modified robots have not such problem. They behave pretty well.

It seems that energy condition in reward create a vast minimum, and it for GA to get free from it.
