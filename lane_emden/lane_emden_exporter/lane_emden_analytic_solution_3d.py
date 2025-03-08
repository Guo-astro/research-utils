#!/usr/bin/env python3
import math
import csv

# Global constant: first zero of theta(xi) for n = 3/2.
XI1_GLOBAL = 3.65375


def compute_lane_emden(num_steps=10000):
    """
    Compute the Lane–Emden solution for n = 3/2 from xi = 1e-6 to XI1_GLOBAL using RK4.

    Returns:
        A list of tuples (xi, theta) containing the computed solution.
    """
    # Initial condition: use series expansion at small xi.
    xi = 1e-6
    theta = 1.0 - (xi * xi) / 6.0
    dtheta = -xi / 3.0

    # Compute the step size h from xi = 1e-6 to XI1_GLOBAL.
    h = (XI1_GLOBAL - 1e-6) / num_steps

    # Prepare the result list and store the first value.
    results = [(xi, theta)]

    # Define the derivative functions.
    def f1(xi, theta, dtheta):
        # d(theta)/d(xi) = dtheta.
        return dtheta

    def f2(xi, theta, dtheta):
        # d(dtheta)/d(xi) = - theta^(3/2) - (2/xi)*dtheta.
        return - math.pow(theta, 1.5) - (2.0 / xi) * dtheta

    # RK4 integration loop.
    for _ in range(num_steps):
        k1_theta = h * f1(xi, theta, dtheta)
        k1_dtheta = h * f2(xi, theta, dtheta)

        k2_theta = h * f1(xi + h / 2, theta + k1_theta / 2, dtheta + k1_dtheta / 2)
        k2_dtheta = h * f2(xi + h / 2, theta + k1_theta / 2, dtheta + k1_dtheta / 2)

        k3_theta = h * f1(xi + h / 2, theta + k2_theta / 2, dtheta + k2_dtheta / 2)
        k3_dtheta = h * f2(xi + h / 2, theta + k2_theta / 2, dtheta + k2_dtheta / 2)

        k4_theta = h * f1(xi + h, theta + k3_theta, dtheta + k3_dtheta)
        k4_dtheta = h * f2(xi + h, theta + k3_theta, dtheta + k3_dtheta)

        theta += (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6.0
        dtheta += (k1_dtheta + 2 * k2_dtheta + 2 * k3_dtheta + k4_dtheta) / 6.0
        xi += h

        results.append((xi, theta))

    return results


def export_lane_emden_csv(filename="lane_emden_data.csv", num_steps=10000):
    """
    Exports the Lane–Emden solution data to a CSV file.

    The CSV file will have a header "xi,theta" followed by the computed values.
    """
    data = compute_lane_emden(num_steps=num_steps)
    try:
        with open(filename, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["xi", "theta"])
            writer.writerows(data)
        print(f"Lane–Emden solution exported to '{filename}' with {len(data)} entries.")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")


if __name__ == "__main__":
    export_lane_emden_csv()
