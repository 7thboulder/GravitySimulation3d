As of 4/11/2026:

This is a basic newtonian physics simulation for how massive objects interact via gravity. Currently I have the solar system modeled in the main.py file and a binary star system modeled in the binaryStarSystem.py file.

The objects that are commented out in the main file are the sun and a black hole with mass of 100 solar masses, if you comment out the central anchored mass code and un comment the black hole and sun, it gives a pretty interesting simulation.

The xyzPlanet.py file is a class that represents all the data for each individual object, you give it a name, a velocity vector in m/s, a position vector in meters, mass in kg, radius in meters, and color of the object.

Its pretty interesting and fun to add new objects and/or change mass values or velocity values for planets in the solar system. Although basic as of now, I plan on expanding this and making it more accurate in the future. As of now I plan on changing the physics engine to a Runge-Kutta 4th Order algorithm which will make it a bit more accurate. I also plan on using the Schwartzchild metric to add accurate deflection of light in a strong gravitational field.
