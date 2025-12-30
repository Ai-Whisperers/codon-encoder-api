/**
 * API Client
 */

export async function fetchVisualizationData() {
    const response = await fetch('/api/visualization');
    if (!response.ok) throw new Error('Failed to load visualization data');
    return await response.json();
}

export async function fetchEdges(mode) {
    const response = await fetch(`/api/edges?mode=${mode}`);
    if (!response.ok) throw new Error('Failed to load edges');
    return await response.json();
}

export async function encodeDnaSequence(sequence) {
    const response = await fetch('/api/encode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ sequence })
    });
    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Encoding failed');
    }
    return await response.json();
}

export async function fetchAngularVariance() {
    const response = await fetch('/api/angular_variance');
    if (!response.ok) throw new Error('Failed to load variance data');
    return await response.json();
}

export async function fetchActbReference() {
    const response = await fetch('/api/reference/actb');
    if (!response.ok) throw new Error('Failed to load ACTB reference');
    return await response.json();
}
