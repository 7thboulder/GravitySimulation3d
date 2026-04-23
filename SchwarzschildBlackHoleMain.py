import SchwarzschildBlackHoleSimulation

def main():
    # `System` owns the simulation loop, PyVista scene, camera, and input.
    solar_system = SchwarzschildBlackHoleSimulation.BlackHoleSimulation(
        central_mass=1.31268720076e41,
        central_color='yellow',
        central_name='Sun',

        # Change in time every frame in seconds. If the value is too high the physics will break.
        dt=100.0,
    )
    solar_system.setup_scene()
    solar_system.run()

if __name__ == '__main__':
    # Run the app when this file is executed directly.
    main()

