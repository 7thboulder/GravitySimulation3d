import xyzPlanet
import SchwarzschildMetricSystem


# You must add a central body for this first version of the NewtonGRSystem
def main():

    system = SchwarzschildMetricSystem.SchwarzschildSystem([],1, 1, 1)

    geod = system.calculate_full_orbit(8, 8, 0, 0, 0, 1)
    system.plot_orbit_with_mass(geod)


if __name__ == '__main__':
    # Run the app when this file is executed directly.
    main()
