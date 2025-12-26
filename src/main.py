import random
import time
import matplotlib.pyplot as plt
import numpy as np
from generators import generate_polygon, visualize_polygon
from algorithms import gauss_area, monte_carlo_area


def task1_generate_test_polygons():
    print("\n=== Task 1: Generating Test Polygons ===")
    random.seed(42)

    vertex_counts = [10, 50, 100]

    for n in vertex_counts:
        poly = generate_polygon(num_points=n, radius=50)
        filename = f"../images/polygon_n{n}.png"
        visualize_polygon(poly, filename=filename)
        print(f"Polygon with {n} vertices generated. Shapely Area: {poly.area:.4f}")


def task2_monte_carlo_accuracy_study():
    print("\n=== Task 2: Monte-Carlo Accuracy Study (N=50) ===")
    random.seed(42)

    poly = generate_polygon(num_points=50, radius=50)
    shapely_area = poly.area

    sample_counts = [100, 1000, 10000, 100000]
    errors = []

    print(f"Shapely reference area: {shapely_area:.4f}")

    for M in sample_counts:
        mc_area = monte_carlo_area(poly, num_samples=M)
        error_percent = abs(mc_area - shapely_area) / shapely_area * 100
        errors.append(error_percent)

        print(f"M={M:>6}: MC Area = {mc_area:>10.4f}, Error = {error_percent:>6.3f}%")

    plt.figure(figsize=(8, 6))
    plt.plot(sample_counts, errors, marker='o', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Number of samples (M)', fontsize=12)
    plt.ylabel('Relative error (%)', fontsize=12)
    plt.title('Monte-Carlo Accuracy vs Number of Samples (N=50)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('../images/error_plot.png')
    print("Error plot saved: ../images/error_plot.png")
    plt.close()


def task3_performance_benchmark():
    print("\n=== Task 3: Performance Benchmark ===")
    random.seed(42)

    vertex_counts = [10, 50, 100, 1000]
    num_mc_samples = 100000

    results = []

    for n in vertex_counts:
        poly = generate_polygon(num_points=n, radius=50)

        start = time.perf_counter()
        shapely_area = poly.area
        time_shapely = time.perf_counter() - start

        start = time.perf_counter()
        gauss_result = gauss_area(poly)
        time_gauss = time.perf_counter() - start

        start = time.perf_counter()
        mc_result = monte_carlo_area(poly, num_samples=num_mc_samples)
        time_mc = time.perf_counter() - start

        results.append({
            'N': n,
            'Shapely (s)': time_shapely,
            'Gauss (s)': time_gauss,
            'Monte-Carlo (s)': time_mc,
            'Shapely Area': shapely_area,
            'Gauss Area': gauss_result,
            'MC Area': mc_result
        })

        print(f"N={n:>4}: Shapely={time_shapely:.6f}s, Gauss={time_gauss:.6f}s, MC={time_mc:.6f}s")

    print("\n--- Benchmark Results Table ---")
    print(f"{'N':<6} {'Shapely (s)':<15} {'Gauss (s)':<15} {'Monte-Carlo (s)':<15}")
    print("-" * 51)
    for r in results:
        print(f"{r['N']:<6} {r['Shapely (s)']:<15.8f} {r['Gauss (s)']:<15.8f} {r['Monte-Carlo (s)']:<15.8f}")

    vertices = [r['N'] for r in results]
    time_shapely = [r['Shapely (s)'] for r in results]
    time_gauss = [r['Gauss (s)'] for r in results]
    time_mc = [r['Monte-Carlo (s)'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(vertices, time_shapely, marker='o', label='Shapely', linewidth=2)
    plt.plot(vertices, time_gauss, marker='s', label='Gauss', linewidth=2)
    plt.plot(vertices, time_mc, marker='^', label='Monte-Carlo (M=10^5)', linewidth=2)
    plt.xlabel('Number of vertices (N)', fontsize=12)
    plt.ylabel('Execution time (s)', fontsize=12)
    plt.title('Performance Benchmark: Area Calculation Methods', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('../images/time_benchmark.png')
    print("\nTime benchmark plot saved: ../images/time_benchmark.png")
    plt.close()

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Lab 6: Polygon Area Calculation Methods")
    print("=" * 60)

    task1_generate_test_polygons()
    task2_monte_carlo_accuracy_study()
    benchmark_results = task3_performance_benchmark()

    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)