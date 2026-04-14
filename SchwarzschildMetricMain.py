import xyzPlanet
import SchwarzschildMetricSystem


# You must add a central body for this first version of the NewtonGRSystem
def main():

    system = SchwarzschildMetricSystem.SchwarzschildSystem([],1, 1, 1)

    system.calculate_circular_orbit(8)


if __name__ == '__main__':
    # Run the app when this file is executed directly.
    main()
