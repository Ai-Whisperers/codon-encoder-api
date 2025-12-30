/**
 * Hyperbolic Geometry Utilities
 */
import * as THREE from 'three';

// Convert depth to Poincare disk radius (d=0 outer, d=9 center)
export function depthToPoincareRadius(depth) {
    // Map d=0 to r=0.9, d=9 to r=0.1
    return 0.9 - (depth / 9) * 0.8;
}

// Get position on Poincare disk from depth and angle
export function poincarePosition(depth, angle, height = 0) {
    const r = depthToPoincareRadius(depth);
    return new THREE.Vector3(
        r * Math.cos(angle),
        height,
        r * Math.sin(angle)
    );
}

// Compute hyperbolic geodesic between two points in Poincare disk
export function computeGeodesicArc(p1, p2, segments = 20) {
    const points = [];

    // Project to 2D (x, z plane)
    const x1 = p1.x, z1 = p1.z;
    const x2 = p2.x, z2 = p2.z;

    // For Poincare disk, geodesics are circular arcs
    // Simplified: use quadratic bezier approximation
    const midX = (x1 + x2) / 2;
    const midZ = (z1 + z2) / 2;

    // Compute perpendicular offset for arc curvature
    const dx = x2 - x1;
    const dz = z2 - z1;
    const dist = Math.sqrt(dx * dx + dz * dz);

    // Handle same point case
    if (dist < 0.0001) {
        return [p1.clone()];
    }

    // Arc curvature based on distance from center
    const avgRadius = (Math.sqrt(x1*x1 + z1*z1) + Math.sqrt(x2*x2 + z2*z2)) / 2;
    const curvature = avgRadius * 0.5;

    // Perpendicular direction (pointing outward from center)
    const perpX = -dz / dist * curvature;
    const perpZ = dx / dist * curvature;

    // Control point for quadratic bezier
    const ctrlX = midX + perpX;
    const ctrlZ = midZ + perpZ;

    // Interpolate height
    const y1 = p1.y, y2 = p2.y;

    for (let i = 0; i <= segments; i++) {
        const t = i / segments;
        const t1 = 1 - t;

        // Quadratic bezier
        const x = t1 * t1 * x1 + 2 * t1 * t * ctrlX + t * t * x2;
        const z = t1 * t1 * z1 + 2 * t1 * t * ctrlZ + t * t * z2;
        const y = t1 * y1 + t * y2;

        points.push(new THREE.Vector3(x, y, z));
    }

    return points;
}

// Get position based on projection mode
export function getProjectedPosition(point, index, mode) {
    switch (mode) {
        case 'pca':
            // Use PCA projection directly from server
            if (point.projection && point.projection.length >= 3) {
                return new THREE.Vector3(
                    point.projection[0],
                    point.projection[1],
                    point.projection[2]
                );
            }
            // Fallback to poincare if no projection
            return poincarePosition(point.depth, point.angle || (index/64)*Math.PI*2, 0);

        case 'hemisphere':
            // Map Poincare disk to hemisphere
            const r = point.poincare_radius !== undefined
                ? point.poincare_radius
                : depthToPoincareRadius(point.depth);
            const angle = point.angle || (index/64)*Math.PI*2;

            // Hemisphere: x = r*cos(θ), z = r*sin(θ), y = sqrt(1 - r²)
            const x = r * Math.cos(angle);
            const z = r * Math.sin(angle);
            const y = Math.sqrt(Math.max(0, 1 - r * r));
            return new THREE.Vector3(x, y, z);

        case 'poincare':
        default:
            return poincarePosition(point.depth, point.angle || (index/64)*Math.PI*2, 0);
    }
}

// Compute hyperbolic distance in Poincare disk
export function hyperbolicDistance(p1, p2) {
    const x1 = p1.x, z1 = p1.z;
    const x2 = p2.x, z2 = p2.z;

    const r1 = Math.sqrt(x1*x1 + z1*z1);
    const r2 = Math.sqrt(x2*x2 + z2*z2);

    const eucDist = Math.sqrt((x2-x1)**2 + (z2-z1)**2);

    // Handle edge cases
    if (eucDist < 0.0001) return 0; // Same point
    if (r1 >= 0.99 || r2 >= 0.99) return Infinity; // At boundary

    // Poincare disk metric: d = 2 * arctanh(|z1 - z2| / |1 - z1*conj(z2)|)
    // Simplified approximation
    const arg = eucDist / (1 + r1 * r2);
    if (arg >= 1) return Infinity; // Clamp to avoid NaN

    return 2 * Math.atanh(arg);
}
