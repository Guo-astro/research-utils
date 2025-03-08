#!/usr/bin/env python3
import math
import csv
import matplotlib.pyplot as plt

# Global constant: upper bound for xi in 2D for n = 3/2 (beyond the first zero).
# Adjust XI1_GLOBAL to match the physical zero (e.g., around 2.752) if needed.
XI1_GLOBAL = 4.0  # Set to capture the solution; adjust if needed.


def compute_lane_emden(num_steps=10000):
    """
    Compute the Lane-Emden solution for n = 3/2 in 2D from xi = 1e-6 to XI1_GLOBAL using RK4.

    Returns:
        A list of tuples (xi, theta) containing the computed solution.
    """
    # Use a consistent global value (adjust as needed)
    xi_max = XI1_GLOBAL  # e.g., for physical consistency, you might use ~2.752 for n=3/2 in 2D.

    # Initial conditions using series expansion at small xi for 2D
    xi = 1e-6
    theta = 1.0 - (xi * xi) / 4.0  # 2D: theta ≈ 1 - (1/4) xi^2
    dtheta = -xi / 2.0  # 2D: dtheta/dxi ≈ -(1/2) xi

    # Compute step size
    h = (xi_max - 1e-6) / num_steps

    # Store results
    results = [(xi, theta)]

    # Derivative functions for 2D Lane-Emden equation
    def f1(xi, theta, dtheta):
        return dtheta  # d(theta)/d(xi) = dtheta

    def f2(xi, theta, dtheta):
        if theta < 0:
            return 0  # Prevent negative theta from causing errors
        return -math.pow(theta, 1.5) - (1.0 / xi) * dtheta  # d(dtheta)/d(xi)

    # RK4 integration loop
    for _ in range(num_steps):
        # RK4 steps
        k1_theta = h * f1(xi, theta, dtheta)
        k1_dtheta = h * f2(xi, theta, dtheta)

        k2_theta = h * f1(xi + h / 2, theta + k1_theta / 2, dtheta + k1_dtheta / 2)
        k2_dtheta = h * f2(xi + h / 2, theta + k1_theta / 2, dtheta + k1_dtheta / 2)

        k3_theta = h * f1(xi + h / 2, theta + k2_theta / 2, dtheta + k2_dtheta / 2)
        k3_dtheta = h * f2(xi + h / 2, theta + k2_theta / 2, dtheta + k2_dtheta / 2)

        k4_theta = h * f1(xi + h, theta + k3_theta, dtheta + k3_dtheta)
        k4_dtheta = h * f2(xi + h, theta + k3_theta, dtheta + k3_dtheta)

        # Update theta and dtheta
        theta_new = theta + (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6.0
        dtheta_new = dtheta + (k1_dtheta + 2 * k2_dtheta + 2 * k3_dtheta + k4_dtheta) / 6.0

        # Stop if theta becomes negative
        if theta_new < 0:
            break

        theta = theta_new
        dtheta = dtheta_new
        xi += h

        results.append((xi, theta))

    return results


def export_lane_emden_csv(filename="lane_emden_2d_data.csv", num_steps=10000):
    """
    Exports the Lane–Emden solution data for 2D to a CSV file.

    The CSV file will have a header "xi,theta" followed by the computed values.
    """
    data = compute_lane_emden(num_steps=num_steps)
    try:
        with open(filename, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["xi", "theta"])
            writer.writerows(data)
        print(f"Lane–Emden solution for 2D exported to '{filename}' with {len(data)} entries.")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")


def plot_lane_emden(num_steps=10000):
    """
    Plot the Lane-Emden profile computed from the RK4 integration.
    """
    data = compute_lane_emden(num_steps=num_steps)
    xis = [point[0] for point in data]
    thetas = [point[1] for point in data]

    plt.figure()
    plt.plot(xis, thetas, label='Lane-Emden Profile')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\theta$')
    plt.title('Lane-Emden Profile for n=3/2 (2D)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Export CSV file if needed.
    export_lane_emden_csv()
    # Plot the Lane-Emden profile.
    plot_lane_emden()
