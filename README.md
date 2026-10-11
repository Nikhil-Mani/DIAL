# Pendulum sampling MPC

Implemented components: pendulum trajectory rollout and configurable swing-up
cost, Gaussian sampling, Metropolis-Hastings search and simulated tempering in Python, using only the standard library.

The general flow is:
- Optimizer picks a torque sequence over the horizon (2 seconds)
- We noise it by sampling around it
- Check behavior over the next few steps
- Update

State is `(angle_radians, angular_velocity_radians_per_second)`; torque is not part of the state, and a control input in Nm.
Zero angle points downward; pi points upward. Angles are unwrapped.
The model is a point mass on a massless rod, with optional damping.
Each torque is held constant for one timestep, integrated with RK4.

Currently, the samplers fail to converge to swing-up. Next steps are:
- Decreasing time between samples
- Try switching to a hybrid control model, to prevent hysterisis and decouple the swing up and stabilize portions

The goal is to implement a DIAL-MPC style update on a Unitree, but we are starting with the pendulum to learn the basics.
