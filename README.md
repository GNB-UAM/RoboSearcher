# RoboSearcher
Simulation platform to study and optimize random search strategies. This is supplementary material for [our article](https://doi.org/10.1038/s41598-021-03826-3):

> C. Garcia-Saura, E. Serrano, F.B. Rodriguez, P. Varona. 2021. **Intrinsic and environmental factors modulating autonomous robotic search under high uncertainty. [Scientific Reports 11: 24509.](https://doi.org/10.1038/s41598-021-03826-3)**

![demo animation](Media/example.gif)


Instructions
-
1) Modify the parameters in `runSim.py` and `simulevy.py`, adapt or uncomment the sections as needed.
2) Install the dependency libraries that are imported at the top of each file.
3) Execute the code with `python3 runSim.py`.

Note: We recommend Numpy Pickle (.npy) to export the desired parameters for their later representation.


Future work
-

Next step would be to simplify the setup of the simulations by implementing a command line interface or a visual GUI. Another option would be to integrate this into a standard library that can be used within other applications.


How to cite
-
Upon use of this software please remember to cite the following publication:

- C. Garcia-Saura, E. Serrano, F.B. Rodriguez, P. Varona. 2021. **Intrinsic and environmental factors modulating autonomous robotic search under high uncertainty. [Scientific Reports 11: 24509.](https://doi.org/10.1038/s41598-021-03826-3)**

Other relevant publications:

- C. Garcia-Saura, E. Serrano, F.B. Rodriguez, P. Varona. 2017. **Effects of Locomotive Drift in Scale-Invariant Robotic Search Strategies. [Lect Notes in Comput. Sc 10384: 161-169.](https://doi.org/10.1007/978-3-319-63537-8_14)**

- C. Garcia-Saura, F.B. Rodriguez, P. Varona. 2014. **Design Principles for Cooperative Robots with Uncertainty-Aware and Resource-Wise Adaptive Behavior. [Lect Notes in Comput. Sc 8608: 108-117.](http://dx.doi.org/10.1007/978-3-319-09435-9_10)**

