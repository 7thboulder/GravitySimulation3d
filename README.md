For most of the interactive simulations, the controls are W, A, S, and D to go forward, left, back, and right respectively, the arrow keys are to change the angle of the camera. Space bar will stop time but this is only relavant in the newtonian simulations of bodies like the solar system or the two body system I have set up. ] will increase the speed of the camera and [ will slow down the speed of the camera, speed is listed in the bottom corner. Also if you want to change the time step in the newtonian simulations then change the dt value in the code, too much higher than 75000 can break the code but if you want to represent some slow moving bodies than it shouldn't be too much of a problem.

WARNING: IF YOU USE THE VISUAL SIMULATION, IT TAKES A LOT OF CPU POWER AND BE CAREFUL WITH NOT USING TO MANY CPU THREADS TO RENDER IT - the CPU threads can be changed at the top of the PlanarBlackHoleVisual.py code.

The GPU version of the PlanarBlackHoleVisual file runs much faster and is much less resource intensive but as of now it does not work.

For people who just want to see what each file does and run them:
- main.py is a simulation of the solar system, if you read that code its pretty easy to understand how to add more object into the solar system or create whole new systems. 
- binaryStarSystem.py is a simulation of a planar stable binary star system, I haven't played around with it much but go ahead and see what cool stuff can be done with it.
- SchwarzschildBlackHoleMain.py is a visual for how spacetime curves around a black hole with light rays to show why light curves around a black hole.
- PlanarBlackHoleVisual is a visual simulation of a black hole, this is what I'm currently working on. Feel free to play with some of the values at the top, but do be warned that the visuals take a while to render and be careful with how many CPU threads your using to do the simulation because without a decent CPU, the current 12 threads I have it set up at will probably crash whatever your using to run it.

As of 4/11/2026:

This is a basic newtonian physics simulation for how massive objects interact via gravity. Currently I have the solar system modeled in the main.py file and a binary star system modeled in the binaryStarSystem.py file.

The objects that are commented out in the main file are the sun and a black hole with mass of 100 solar masses, if you comment out the central anchored mass code and un comment the black hole and sun, it gives a pretty interesting simulation.

The xyzPlanet.py file is a class that represents all the data for each individual object, you give it a name, a velocity vector in m/s, a position vector in meters, mass in kg, radius in meters, and color of the object.

Its pretty interesting and fun to add new objects and/or change mass values or velocity values for planets in the solar system. Although basic as of now, I plan on expanding this and making it more accurate in the future. As of now I plan on changing the physics engine to a Runge-Kutta 4th Order algorithm which will make it a bit more accurate. I also plan on using the Schwartzchild metric to add accurate deflection of light in a strong gravitational field.

As of 4/30/2026:

I've made a lot of progress, I still have the file that simulates the orbit of planets but now I also have a file that simulates how light follows the curvature of spacetime - the file is SchwarzschildBlackHoleMain.py.

I am currently working on a visual simulation of a black hole, there are several test files for this but the one that is working best at the moment is PlanarBlackHoleVisual.py. This file generates a pretty good image, but it does take a long time to render.

**Also disclaimer, I do use AI to aid in the coding and help me understand more about the math involved but I do try to had write as much of this as I can to keep it as authentic as possible.
