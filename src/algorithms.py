import random
from shapely.geometry import Polygon, Point


def gauss_area(polygon: Polygon) -> float:
    coords = list(polygon.exterior.coords)
    n = len(coords) - 1

    area_sum = 0.0
    for i in range(n):
        x_i, y_i = coords[i]
        x_next, y_next = coords[i + 1]
        area_sum += (x_i * y_next - x_next * y_i)

    return abs(area_sum) / 2.0


def monte_carlo_area(polygon: Polygon, num_samples: int = 10000) -> float:
    minx, miny, maxx, maxy = polygon.bounds

    box_area = (maxx - minx) * (maxy - miny)

    inside_count = 0
    for _ in range(num_samples):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        point = Point(x, y)

        if polygon.contains(point):
            inside_count += 1

    estimated_area = box_area * (inside_count / num_samples)

    return estimated_area


if __name__ == "__main__":
    from generators import generate_polygon
    random.seed(42)

    test_polygon = generate_polygon(num_points=50, radius=50)

    shapely_area = test_polygon.area
    print(f"Shapely Area: {shapely_area:.4f}")

    gauss_result = gauss_area(test_polygon)
    print(f"Gauss Area: {gauss_result:.4f}")
    print(f"Gauss Error: {abs(gauss_result - shapely_area) / shapely_area * 100:.4f}%")

    mc_result = monte_carlo_area(test_polygon, num_samples=100000)
    print(f"Monte-Carlo Area (100k samples): {mc_result:.4f}")
    print(f"Monte-Carlo Error: {abs(mc_result - shapely_area) / shapely_area * 100:.4f}%")
