import xyzPlanet
import xyzSystem


def create_planets():
    # Build the initial body list from the ephemeris pulled from https://ssd.jpl.nasa.gov/horizons/.
    # Positions and velocities are stored in meters and meters/second.
    return [
        xyzPlanet.Planet(
            'Neptune',
            [(-1.965905399912297e-1) * 1000, 5.464247863262584 * 1000, (-1.076053040089056e-1) * 1000],
            [4.467386796032222e9 * 1000, 1.234334047987042e8 * 1000, (-1.054899544864930e8) * 1000],
            1.0241e26,
            2.4764e7,
            '#5b5ddf',
        ),
        xyzPlanet.Planet(
            'Uranus',
            [(-5.997369438250795) * 1000, 3.019213694110670 * 1000, 8.911433025470172e-2 * 1000],
            [1.428079138912021e9 * 1000, 2.539034268646180e9 * 1000, (-9.087332139575481e6) * 1000],
            8.68103e25,
            2.5362e7,
            '#ACE5EE',
        ),
        xyzPlanet.Planet(
            'Saturn',
            [(-1.350243705362230) * 1000, 9.601675757882083 * 1000, (-1.128991951309164e-1) * 1000],
            [1.413259838499972e9 * 1000, 1.191936716477834e8 * 1000, (-5.833180409561297e7) * 1000],
            5.683e26,
            6.0268e7,
            '#e2bf7d',
        ),
        xyzPlanet.Planet(
            'Jupiter',
            [(-1.181134524386522e1) * 1000, (-5.312448082650281) * 1000, 2.863275810335251e-1 * 1000],
            [(-3.555292469337319e8) * 1000, 6.997565809872080e8 * 1000, 5.047655235825032e6 * 1000],
            1.89813e27,
            7.1492e7,
            '#b07f35',
        ),
        xyzPlanet.Planet(
            'Mars',
            [7.536286516248815 * 1000, 2.538274308884066e1 * 1000, 3.471360621129680e-1 * 1000],
            [1.988823881728835e8 * 1000, (-5.637874412452721e7) * 1000, (-6.058245953337148e6) * 1000],
            6.41693e23,
            3.396e6,
            '#993D00',
        ),
        xyzPlanet.Planet(
            'Earth',
            [8.615135813645887 * 1000, (-2.847161990235745e1) * 1000, 1.364727432711987e-3 * 1000],
            [(-1.425892837882441e8) * 1000, (-4.579937887891923e7) * 1000, 3.967607066705823e3 * 1000],
            5.972e24,
            6.371e6,
            '#0000A0',
        ),
        xyzPlanet.Planet(
            'Venus',
            [(-3.329986537242428e1) * 1000, 1.103035977732714e1 * 1000, 2.072928807243641 * 1000],
            [3.447753472509624e7 * 1000, 1.021778564091384e8 * 1000, (-5.855524643558413e5) * 1000],
            4.86732e24,
            6.052e6,
            '#F8E2B0',
        ),
        xyzPlanet.Planet(
            'Mercury',
            [3.888214641333979e1 * 1000, 6.683744485665621e-2 * 1000, (-3.560717788332280) * 1000],
            [(-3.450952145112824e6) * 1000, (-6.925993495853576e7) * 1000, (-5.343624481951203e6) * 1000],
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
        # xyzPlanet.Planet('Sun',
        #                  [0,0,0],
        #                  [0,0,0],
        #                  1.989e30,
        #                  6.957e8,
        #                  'yellow'
        # ),
    ]


def main():
    # `System` owns the simulation loop, PyVista scene, camera, and input.
    solar_system = xyzSystem.System(
        create_planets(),

        # Change in time every frame in seconds. If the value is too high the physics will break.
        dt=75000.0,

        # Comment these three lines out if you don't want the system anchored to a center mass, ex: if you wanted to build a binary star system.
        central_mass=1.989e30,
        central_radius=6.957e8,
        central_color='yellow',
        #

        render_scale=1e10,
    )
    solar_system.setup_scene()
    solar_system.run()


if __name__ == '__main__':
    # Run the app when this file is executed directly.
    main()
