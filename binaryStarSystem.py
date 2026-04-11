import xyzPlanet
import xyzSystem


def create_planets():
    return [
        xyzPlanet.Planet(
            'Sun 1',
            [0, 12161, 0],
            [4.99e10, 0, 0],
            1.989e30,
            6.957e8,
            'yellow',
        ),
        xyzPlanet.Planet(
            'Sun 2',
            [0, -24323, 0],
            [-9.97e10, 0, 0],
            9.945e29,
            (0.79*6.957e8),
            'yellow',
        ),
    ]


def main():
    solar_system = xyzSystem.System(
        create_planets(),
        dt=1000.0,
        render_scale=1e10,
    )
    solar_system.setup_scene()
    solar_system.run()


if __name__ == '__main__':
    main()
