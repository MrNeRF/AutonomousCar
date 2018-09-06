# Model Predictive Control applied to autonomous wheeled mobile robot
This is the repository with the C++ implementation of Model Predictive Control on a simple self-made wheeled mobile robot (WMR)  - a skid-steering drive - as done in my bachelor thesis in applied mathematics. The controller lets the WMR follow any given reference path that can be parameterized. The controller is called mpc_controller in the src folder. The computer vision system "cv_system" provides the WMR with its current state information (x-,y-coordinates, heading). An example of the controller in action can be found in the following youtube video.

[![MPC Horizon 15](https://img.youtube.com/vi/7Zckkx6ERC8/0.jpg)](https://www.youtube.com/watch?v=7Zckkx6ERC8 "Model Predictive Control Horizon 15")

The controller requests its initial state from the cv_system, so that cv_system has to be started first.
