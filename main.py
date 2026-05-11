import xyzPlanet
import xyzSystem


def create_planets():
    # Build the initial body list from the ephemeris pulled from https://ssd.jpl.nasa.gov/horizons/.
    # Positions and velocities are stored in meters and meters/second.
    return [
        xyzPlanet.Planet(
            'Neptune',
            [(-1.846252051580562e-1) * 1000, 5.466210363415099 * 1000, (-1.078488329550216e-1) * 1000],
            [4.467030372207629e9 * 1000, 1.226149913689110e8 * 1000, (-1.054722726035721e8) * 1000],
            1.0241e26,
            2.4764e7,
            '#5b5ddf',
        ),
        xyzPlanet.Planet(
            'Uranus',
            [(-5.985404103417621) * 1000, 3.021176194263185 * 1000, 8.887080130858549e-2 * 1000],
            [1.427722715087428e9 * 1000, 2.538215855216387e9 * 1000, (-9.069650256654501e6) * 1000],
            8.68103e25,
            2.5362e7,
            '#ACE5EE',
        ),
        xyzPlanet.Planet(
            'Saturn',
            [(-1.338278370529056) * 1000, 9.603638258034600 * 1000, (-1.131427240770324e-1) * 1000],
            [1.412903414675380e9 * 1000, 1.183752582179902e8 * 1000, (-5.831412221269201e7) * 1000],
            5.683e26,
            6.0268e7,
            '#e2bf7d',
        ),
        xyzPlanet.Planet(
            'Jupiter',
            [(-1.179937990903205e1) * 1000, (-5.310485582497766) * 1000, 2.860840520874088e-1 * 1000],
            [(-3.558856707583244e8) * 1000, 6.989381675574148e8 * 1000, 5.065337118746012e6 * 1000],
            1.89813e27,
            7.1492e7,
            '#b07f35',
        ),
        xyzPlanet.Planet(
            'Mars',
            [7.548251851081988 * 1000, 2.538470558899317e1 * 1000, 3.468925331668515e-1 * 1000],
            [1.985259643482910e8 * 1000, (-5.719715755432040e7) * 1000, (-6.040564070416190e6) * 1000],
            6.41693e23,
            3.396e6,
            '#993D00',
        ),
        xyzPlanet.Planet(
            'Earth',
            [8.627101148479060 * 1000, (-2.846965740220493e1) * 1000, 1.121198486597308e-3 * 1000],
            [(-1.429457076128366e8) * 1000, (-4.661779230871242e7) * 1000, 2.164948998766020e4 * 1000],
            5.972e24,
            6.371e6,
            '#0000A0',
        ),
        xyzPlanet.Planet(
            'Venus',
            [(-3.328790003759111e1) * 1000, 1.103232227747965e1 * 1000, 2.072685278297524 * 1000],
            [3.412111090050375e7 * 1000, 1.013594429793452e8 * 1000, (-5.678705814348906e5) * 1000],
            4.86732e24,
            6.052e6,
            '#F8E2B0',
        ),
        xyzPlanet.Planet(
            'Mercury',
            [3.889411174817296e1 * 1000, 6.879994500917141e-2 * 1000, (-3.560961317278396) * 1000],
            [(-3.807375969705307e6) * 1000, (-7.007834838832895e7) * 1000, -5.325942599030249e6 * 1000],
            3.30104e23,
            2.44e6,
            '#E5E5E5',
        ),
        # xyzPlanet.Planet(
        #     'Black Hole',
        #     [0,75000,0],
        #     [4.997565809872080e8 * 1000, -2e+12, 1.496e+11],
        #     1.989e+32,
        #     6.957e8,
        #     'white'
        # ),
        xyzPlanet.Planet('Sun',
                         [1.196533483317350e-2*1000,1.962500152515327e-3*1000,-2.435289461162395e-4*1000],
                         [-3.564238245924839e5*1000,-8.184134297931885e5*1000,1.768188292095705e4*1000],
                         1.989e30,
                         6.957e8,
                         'yellow'
        ),
    ]


def main():
    # `System` owns the simulation loop, PyVista scene, camera, and input.
    solar_system = xyzSystem.System(
        create_planets(),

        # Change in time every frame in seconds. If the value is too high the physics will break.
        dt=100000.0,

        # Comment these three lines out if you don't want the system anchored to a center mass, ex: if you wanted to build a binary star system.
        # central_mass=1.989e30,
        # central_radius=6.957e8,
        # central_color='yellow',
        # central_name = 'Sun',
        #

        render_scale=1e10,
    )
    solar_system.setup_scene()
    solar_system.run()


if __name__ == '__main__':
    # Run the app when this file is executed directly.
    main()
