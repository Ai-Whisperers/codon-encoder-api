import * as THREE from 'three';
import * as Globals from './globals.js?v=5';
import * as API from './api.js?v=5';
import * as Geometry from './geometry.js?v=5';
import * as UI from './ui.js?v=5';
import { setupScene, onWindowResize } from './scene.js?v=5';

// Initialize Scene
const { scene, camera, renderer, controls, raycaster, groups } = setupScene('container');
const mouse = new THREE.Vector2();

// Module-level variables
let pointsMesh;
let pointsGeometry;
let pointsMaterial;

// State convenience
const state = Globals.state;

// Event Listeners
window.addEventListener('resize', () => onWindowResize(camera, renderer));
renderer.domElement.addEventListener('mousemove', onMouseMove);

document.getElementById('projectionMode').addEventListener('change', onProjectionChange);
document.getElementById('colorBy').addEventListener('change', onColorModeChange);
document.getElementById('trajectoryStyle').addEventListener('change', () => {
    if (state.currentEncodedSequence.length > 0) drawTrajectory(state.currentEncodedSequence);
});
document.getElementById('showDepthRings').addEventListener('change', toggleDepthRings);
document.getElementById('showHierarchyTree').addEventListener('change', toggleHierarchyTree);
document.getElementById('showAllCodons').addEventListener('change', toggleAllCodons);
document.getElementById('showFibers').addEventListener('change', toggleFibers);
document.getElementById('fiberMode').addEventListener('change', onFiberModeChange);
document.getElementById('showClusterCenters').addEventListener('change', toggleClusterCenters);

// Filter controls
document.getElementById('depthFilterMin').addEventListener('input', onDepthFilterChange);
document.getElementById('depthFilterMax').addEventListener('input', onDepthFilterChange);
document.getElementById('aminoAcidFilter').addEventListener('change', onAminoAcidFilterChange);
document.getElementById('resetFiltersBtn').addEventListener('click', resetFilters);

document.getElementById('encodeBtn').addEventListener('click', encodeSequence);
document.getElementById('clearBtn').addEventListener('click', clearSequence);
document.getElementById('resetView').addEventListener('click', resetView);
document.getElementById('dnaInput').addEventListener('input', UI.validateDnaInput);
document.getElementById('sequenceList').addEventListener('change', onSequenceListChange);
document.getElementById('dnaForm').addEventListener('submit', (e) => {
    e.preventDefault();
    encodeSequence();
});
document.getElementById('loadActbBtn').addEventListener('click', loadActbReference);
document.getElementById('clearOverlayBtn').addEventListener('click', clearOverlay);
document.getElementById('loadVarianceBtn').addEventListener('click', loadVarianceData);
document.getElementById('variantSelect').addEventListener('change', onVariantSelectChange);

// New feature event listeners
document.getElementById('codonSearch').addEventListener('input', onCodonSearch);
document.getElementById('measureMode').addEventListener('change', onMeasureModeChange);
document.getElementById('showMutationNetwork').addEventListener('change', toggleMutationNetwork);
document.getElementById('animateBtn').addEventListener('click', toggleAnimation);
document.getElementById('animationSpeed').addEventListener('input', onAnimationSpeedChange);
document.getElementById('exportPngBtn').addEventListener('click', exportPNG);
document.getElementById('exportSvgBtn').addEventListener('click', exportSVG);
document.getElementById('exportCsvBtn').addEventListener('click', exportCSV);
document.getElementById('exportJsonBtn').addEventListener('click', exportJSON);
document.getElementById('organismSelect').addEventListener('change', onOrganismChange);

// Click handler for codon selection
renderer.domElement.addEventListener('click', onCodonClick);

// Keyboard shortcuts
document.addEventListener('keydown', onKeyDown);

// Initialize
loadData();
animate();

// Animation Loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Mouse Move Logic (Raycasting)
function onMouseMove(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    if (pointsMesh && pointsMesh.visible) {
        const intersects = raycaster.intersectObject(pointsMesh);
        
        if (intersects.length > 0) {
            const idx = intersects[0].index;
            const point = state.visualizationData.points[idx];
            
            UI.displayTooltip(
                `<div class="codon">${point.codon}</div>
                 <div class="aa">AA: ${point.amino_acid}</div>
                 <div>Depth: ${point.depth}</div>
                 <div>Conf: ${(point.confidence || 0).toFixed(2)}</div>`,
                event.clientX, event.clientY
            );
        } else {
            UI.displayTooltip(null);
        }
    }
}

// Logic Functions

async function loadData() {
    try {
        state.visualizationData = await API.fetchVisualizationData();
        state.serverConfig = state.visualizationData.config || {};
        Globals.applyServerColors(state.serverConfig);

        UI.updateInfoPanel({
            version: state.visualizationData.metadata.version || 'unknown',
            structure: state.visualizationData.metadata.structure_score || 0
        });

        // Initialize dynamic UI elements
        const projectionMode = document.getElementById('projectionMode').value;
        updateDynamicLabels(projectionMode);
        updateControlRelevance();

        UI.buildLegend();
        createDepthRings();
        createPoints();

        await loadEdges('hierarchical');

        // Initialize filter info
        UI.updateFilterInfo(64, 64);

        checkDeepLink();

    } catch (error) {
        console.error("Failed to load data", error);
    }
}

async function loadEdges(mode) {
    if (mode === 'none') {
        state.currentEdges = [];
        groups.fibers.clear();
        return;
    }
    try {
        state.currentEdges = await API.fetchEdges(mode);
        if (document.getElementById('showFibers').checked) {
            renderFibers();
        }
    } catch (e) {
        console.error(e);
    }
}

function createDepthRings() {
    groups.depthRings.clear();
    const projectionMode = document.getElementById('projectionMode').value;

    // Depth rings only make sense for Poincare and Hemisphere modes
    if (projectionMode === 'pca') {
        // For PCA, show simple axis lines instead
        const axisMaterial = new THREE.LineBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.3 });
        const axisLength = 1.0;

        // X axis
        const xAxisGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-axisLength, 0, 0),
            new THREE.Vector3(axisLength, 0, 0)
        ]);
        groups.depthRings.add(new THREE.Line(xAxisGeo, axisMaterial));

        // Y axis
        const yAxisGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, -axisLength, 0),
            new THREE.Vector3(0, axisLength, 0)
        ]);
        groups.depthRings.add(new THREE.Line(yAxisGeo, axisMaterial));

        // Z axis
        const zAxisGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, -axisLength),
            new THREE.Vector3(0, 0, axisLength)
        ]);
        groups.depthRings.add(new THREE.Line(zAxisGeo, axisMaterial));

        return;
    }

    const material = new THREE.LineBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.3 });

    if (projectionMode === 'hemisphere') {
        // For hemisphere, draw latitude circles at different heights
        for (let v = 0; v <= 9; v++) {
            const r = Geometry.depthToPoincareRadius(v);
            const y = Math.sqrt(Math.max(0, 1 - r * r));
            const curve = new THREE.EllipseCurve(0, 0, r, r, 0, 2 * Math.PI, false, 0);
            const points = curve.getPoints(64).map(p => new THREE.Vector3(p.x, y, p.y));
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const ring = new THREE.Line(geometry, material);
            groups.depthRings.add(ring);
        }
        // Equator ring
        const equatorGeo = new THREE.BufferGeometry().setFromPoints(
            new THREE.EllipseCurve(0, 0, 1.0, 1.0, 0, 2 * Math.PI).getPoints(64).map(p => new THREE.Vector3(p.x, 0, p.y))
        );
        const equatorMat = new THREE.LineBasicMaterial({ color: 0x666666 });
        groups.depthRings.add(new THREE.Line(equatorGeo, equatorMat));
    } else {
        // Poincare mode - flat rings
        for (let v = 0; v <= 9; v++) {
            const radius = Geometry.depthToPoincareRadius(v);
            const curve = new THREE.EllipseCurve(0, 0, radius, radius, 0, 2 * Math.PI, false, 0);
            const points = curve.getPoints(64).map(p => new THREE.Vector3(p.x, 0, p.y));
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const ring = new THREE.Line(geometry, material);
            groups.depthRings.add(ring);
        }
        // Boundary ring
        const boundaryGeo = new THREE.BufferGeometry().setFromPoints(
            new THREE.EllipseCurve(0, 0, 1.0, 1.0, 0, 2 * Math.PI).getPoints(64).map(p => new THREE.Vector3(p.x, 0, p.y))
        );
        const boundaryMat = new THREE.LineBasicMaterial({ color: 0x666666 });
        groups.depthRings.add(new THREE.Line(boundaryGeo, boundaryMat));
    }
}

function createPoints() {
    if (!state.visualizationData) return;
    
    if (pointsMesh) {
        scene.remove(pointsMesh);
        if (pointsGeometry) pointsGeometry.dispose();
        if (pointsMaterial) pointsMaterial.dispose();
    }
    
    const data = state.visualizationData.points;
    const geometry = new THREE.BufferGeometry();
    
    const positions = [];
    const colors = [];
    const sizes = [];
    
    const colorMode = document.getElementById('colorBy').value;
    const projectionMode = document.getElementById('projectionMode').value;

    data.forEach((p, i) => {
        const pos = Geometry.getProjectedPosition(p, i, projectionMode);
        positions.push(pos.x, pos.y, pos.z);
        
        let color = new THREE.Color();
        if (colorMode === 'depth') {
            color.setHex(Globals.DEPTH_COLORS[p.depth] || 0xffffff);
        } else if (colorMode === 'amino_acid') {
            color.setHex(Globals.AA_COLORS[p.aa] || Globals.AA_COLORS[p.amino_acid] || 0x888888);
        } else {
            color.setHex(Globals.DEPTH_COLORS[p.depth] || 0xffffff);
        }
        
        colors.push(color.r, color.g, color.b);
        sizes.push(0.08);
    });
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
    
    pointsGeometry = geometry;
    
    pointsMaterial = new THREE.ShaderMaterial({
        uniforms: { opacity: { value: 0.85 } },
        vertexShader: `
            attribute float size;
            varying vec3 vColor;
            void main() {
                vColor = color;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (300.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `,
        fragmentShader: `
            uniform float opacity;
            varying vec3 vColor;
            void main() {
                vec2 center = gl_PointCoord - vec2(0.5);
                float dist = length(center);
                if (dist > 0.5) discard;
                gl_FragColor = vec4(vColor, opacity);
            }
        `,
        transparent: true,
        vertexColors: true
    });
    
    pointsMesh = new THREE.Points(geometry, pointsMaterial);
    pointsMesh.name = 'codons';
    scene.add(pointsMesh);
}

function createCircleTexture() {
    const size = 32;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    
    const context = canvas.getContext('2d');
    context.beginPath();
    context.arc(size/2, size/2, size/2, 0, 2 * Math.PI);
    context.fillStyle = '#ffffff';
    context.fill();
    
    const texture = new THREE.CanvasTexture(canvas);
    return texture;
}

function renderFibers() {
    groups.fibers.clear();
    const edges = state.currentEdges;
    if (!edges || edges.length === 0) return;

    const points = state.visualizationData.points;
    const codonMap = {};
    const indexMap = {};
    points.forEach((p, i) => {
        codonMap[p.codon] = p;
        indexMap[p.codon] = i;
    });

    const projectionMode = document.getElementById('projectionMode').value;
    const positions = [];
    const colors = [];

    edges.forEach(edge => {
        const p1Data = codonMap[edge.source];
        const p2Data = codonMap[edge.target];

        if (!p1Data || !p2Data) return;

        const idx1 = indexMap[edge.source];
        const idx2 = indexMap[edge.target];

        const pos1 = Geometry.getProjectedPosition(p1Data, idx1, projectionMode);
        const pos2 = Geometry.getProjectedPosition(p2Data, idx2, projectionMode);
        
        const arcPoints = Geometry.computeGeodesicArc(pos1, pos2, 10);
        
        for (let i = 0; i < arcPoints.length - 1; i++) {
            positions.push(arcPoints[i].x, arcPoints[i].y, arcPoints[i].z);
            positions.push(arcPoints[i+1].x, arcPoints[i+1].y, arcPoints[i+1].z);
            
            colors.push(0.3, 0.3, 0.3);
            colors.push(0.3, 0.3, 0.3);
        }
    });
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    
    const material = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.15
    });
    
    const lines = new THREE.LineSegments(geometry, material);
    groups.fibers.add(lines);
}

function checkDeepLink() {
    const params = new URLSearchParams(window.location.search);
    const sequences = params.getAll('sequence');

    if (sequences.length > 0) {
        state.loadedSequences = sequences.map((seq, i) => ({
            label: `Seq ${i + 1} (${seq.length}bp)`,
            sequence: seq
        }));

        UI.updateSequenceList(state.loadedSequences);

        if (state.loadedSequences.length > 0) {
            document.getElementById('sequenceList').value = "0";
            onSequenceListChange();
        }
    }
}

function onSequenceListChange() {
    const select = document.getElementById('sequenceList');
    const idx = parseInt(select.value, 10);

    if (!isNaN(idx) && idx >= 0 && idx < state.loadedSequences.length) {
        const seq = state.loadedSequences[idx].sequence;
        document.getElementById('dnaInput').value = seq;
        UI.validateDnaInput();
        encodeSequence();
    }
}

async function encodeSequence() {
    const input = document.getElementById('dnaInput');
    const seq = input.value.replace(/[^ATCG]/gi, '').toUpperCase();
    if (seq.length < 3) return;

    try {
        const result = await API.encodeDnaSequence(seq);
        state.currentEncodedSequence = result;
        drawTrajectory(result);

        // Update sequence panel
        UI.displayEncodedSequence(result);

        // Update info panel with sequence stats
        const geoDist = UI.computeGeodesicDistance(result);
        UI.updateInfoPanel({
            seqLen: `${seq.length} bp (${result.length} codons)`,
            geoDist: geoDist
        });

        // Update colors if in sequence mode
        if (document.getElementById('colorBy').value === 'sequence') {
            updateColors();
        }

    } catch (e) {
        console.error(e);
        alert(e.message);
    }
}

function computeSpiralPath(p1, p2, segments) {
    const points = [];
    for (let i = 0; i <= segments; i++) {
        const t = i / segments;
        // Linear interpolation with sinusoidal height variation
        const x = p1.x + (p2.x - p1.x) * t;
        const z = p1.z + (p2.z - p1.z) * t;
        const baseY = p1.y + (p2.y - p1.y) * t;
        // Add spiral effect: height oscillation
        const spiralHeight = Math.sin(t * Math.PI * 2) * 0.05;
        points.push(new THREE.Vector3(x, baseY + spiralHeight, z));
    }
    return points;
}

function drawTrajectory(encoded) {
    groups.trajectory.clear();
    if (!encoded || encoded.length === 0) return;

    const positions = [];
    const style = document.getElementById('trajectoryStyle').value;
    const projectionMode = document.getElementById('projectionMode').value;

    // Build positions using the current projection mode
    encoded.forEach((item, idx) => {
        // Create a point-like object for getProjectedPosition
        const pointData = {
            depth: item.depth,
            angle: item.angle,
            projection: item.projection,
            poincare_radius: item.poincare_radius
        };
        const pos = Geometry.getProjectedPosition(pointData, idx, projectionMode);
        positions.push(pos);
    });

    const pathPoints = [];

    for (let i = 0; i < positions.length - 1; i++) {
        const p1 = positions[i];
        const p2 = positions[i + 1];

        let segmentPoints = [];
        if (style === 'geodesic' && projectionMode !== 'pca') {
            // Geodesic arcs only make sense for Poincare/Hemisphere
            segmentPoints = Geometry.computeGeodesicArc(p1, p2, 20);
        } else if (style === 'spiral') {
            // Spiral: add intermediate points with height variation
            segmentPoints = computeSpiralPath(p1, p2, 15);
        } else {
            // Linear
            segmentPoints = [p1, p2];
        }

        pathPoints.push(...segmentPoints);
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(pathPoints);
    const material = new THREE.LineBasicMaterial({
        color: 0x00ff88,
        linewidth: 2
    });

    const line = new THREE.Line(geometry, material);
    groups.trajectory.add(line);

    // Markers with codon labels
    encoded.forEach((item, i) => {
        const pos = positions[i];
        if (!pos) return;

        const markerGeo = new THREE.SphereGeometry(0.015, 8, 8);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const marker = new THREE.Mesh(markerGeo, markerMat);
        marker.position.copy(pos);
        marker.userData = { codon: item.codon, aa: item.amino_acid, index: i };
        groups.trajectory.add(marker);
    });
}

function updateColors() {
    if (!pointsMesh || !pointsGeometry || !state.visualizationData) return;

    const mode = document.getElementById('colorBy').value;
    const colors = [];
    const data = state.visualizationData.points;

    // Compute min/max for continuous scales
    const stats = computeColorStats(data);

    data.forEach((p, i) => {
        let color = new THREE.Color();
        const aa = p.amino_acid || p.aa;
        const codon = p.codon;

        switch (mode) {
            // === EMBEDDING PROPERTIES ===
            case 'depth':
                color.setHex(Globals.DEPTH_COLORS[p.depth] || 0xffffff);
                break;

            case 'confidence':
                // Red (low) → Green (high)
                const confT = Globals.normalize(p.confidence, stats.minConf, stats.maxConf);
                color.setHSL(confT * 0.33, 0.8, 0.5);
                break;

            case 'margin':
                // Blue (low) → Red (high)
                const marginT = Globals.normalize(p.margin, stats.minMargin, stats.maxMargin);
                color.setHSL(0.66 - marginT * 0.66, 0.8, 0.5);
                break;

            case 'embedding_norm':
                // Purple (low) → Yellow (high)
                const normT = Globals.normalize(p.embedding_norm, stats.minNorm, stats.maxNorm);
                color.setHSL(0.15 + (1 - normT) * 0.6, 0.8, 0.5);
                break;

            // === GENETIC CODE ===
            case 'amino_acid':
                color.setHex(Globals.AA_COLORS[aa] || 0x888888);
                break;

            case 'degeneracy':
                const deg = Globals.DEGENERACY[aa] || 1;
                color.setHex(Globals.DEGENERACY_COLORS[deg] || 0x888888);
                break;

            case 'gc_content':
                // Blue (0% GC) → Red (100% GC)
                const gc = Globals.getGCContent(codon);
                const gcT = gc / 100;
                color.setHSL(0.66 - gcT * 0.66, 0.9, 0.5);
                break;

            case 'wobble':
                const wobble = Globals.getWobbleBase(codon);
                color.setHex(Globals.WOBBLE_COLORS[wobble] || 0x888888);
                break;

            // === BIOCHEMICAL ===
            case 'hydrophobicity':
                // Blue (hydrophilic, -4.5) → Red (hydrophobic, +4.5)
                const hydro = Globals.HYDROPHOBICITY[aa] || 0;
                const hydroT = Globals.normalize(hydro, -4.5, 4.5);
                color.setHSL(0.66 - hydroT * 0.66, 0.85, 0.5);
                break;

            case 'molecular_weight':
                // Light (low MW) → Dark (high MW)
                const mw = Globals.MOLECULAR_WEIGHT[aa] || 100;
                const mwT = Globals.normalize(mw, 75, 205);
                color.setHSL(0.6, 0.7, 0.7 - mwT * 0.4);
                break;

            case 'chemical_class':
                const chemClass = Globals.CHEMICAL_CLASS[aa] || 'stop';
                color.setHex(Globals.CHEMICAL_CLASS_COLORS[chemClass] || 0x888888);
                break;

            // === SEQUENCE ===
            case 'sequence':
                const seqIdx = state.currentEncodedSequence.findIndex(e => e.codon === codon);
                if (seqIdx >= 0 && state.currentEncodedSequence.length > 0) {
                    const t = seqIdx / (state.currentEncodedSequence.length - 1 || 1);
                    color.setHSL(t * 0.8, 0.9, 0.5);
                } else {
                    const t = (p.position || i) / 63;
                    color.setHSL(t * 0.8, 0.5, 0.5);
                }
                break;

            default:
                color.setHex(0xaaaaaa);
        }

        colors.push(color.r, color.g, color.b);
    });

    pointsGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    pointsGeometry.attributes.color.needsUpdate = true;

    // Update legend
    UI.buildLegend(mode, data);
}

function computeColorStats(data) {
    let minMargin = Infinity, maxMargin = -Infinity;
    let minConf = Infinity, maxConf = -Infinity;
    let minNorm = Infinity, maxNorm = -Infinity;

    data.forEach(p => {
        if (p.margin !== undefined) {
            minMargin = Math.min(minMargin, p.margin);
            maxMargin = Math.max(maxMargin, p.margin);
        }
        if (p.confidence !== undefined) {
            minConf = Math.min(minConf, p.confidence);
            maxConf = Math.max(maxConf, p.confidence);
        }
        if (p.embedding_norm !== undefined) {
            minNorm = Math.min(minNorm, p.embedding_norm);
            maxNorm = Math.max(maxNorm, p.embedding_norm);
        }
    });

    return {
        minMargin: minMargin === Infinity ? 0 : minMargin,
        maxMargin: maxMargin === -Infinity ? 1 : maxMargin,
        minConf: minConf === Infinity ? 0 : minConf,
        maxConf: maxConf === -Infinity ? 1 : maxConf,
        minNorm: minNorm === Infinity ? 0 : minNorm,
        maxNorm: maxNorm === -Infinity ? 1 : maxNorm
    };
}

function toggleAllCodons() {
    if (pointsMesh) {
        pointsMesh.visible = document.getElementById('showAllCodons').checked;
    }
}

function toggleFibers() { 
    groups.fibers.visible = document.getElementById('showFibers').checked; 
    if (groups.fibers.visible) renderFibers(); 
}
function toggleClusterCenters() {
    const visible = document.getElementById('showClusterCenters').checked;
    groups.clusterCenters.visible = visible;
    if (visible && groups.clusterCenters.children.length === 0) {
        createClusterCenters();
    }
}

function createClusterCenters() {
    groups.clusterCenters.clear();
    if (!state.visualizationData) return;

    const projectionMode = document.getElementById('projectionMode').value;
    const points = state.visualizationData.points;

    // Group codons by amino acid
    const aaGroups = {};
    points.forEach((p, i) => {
        const aa = p.amino_acid || p.aa;
        if (!aaGroups[aa]) aaGroups[aa] = [];
        aaGroups[aa].push({ point: p, index: i });
    });

    // Compute centroid for each amino acid group
    for (const [aa, group] of Object.entries(aaGroups)) {
        if (group.length === 0) continue;

        let sumX = 0, sumY = 0, sumZ = 0;
        group.forEach(({ point, index }) => {
            const pos = Geometry.getProjectedPosition(point, index, projectionMode);
            sumX += pos.x;
            sumY += pos.y;
            sumZ += pos.z;
        });

        const centroid = new THREE.Vector3(
            sumX / group.length,
            sumY / group.length,
            sumZ / group.length
        );

        // Create sphere at centroid
        const sphereGeo = new THREE.SphereGeometry(0.03, 16, 16);
        const color = Globals.AA_COLORS[aa] || 0xffffff;
        const sphereMat = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.7
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.copy(centroid);
        sphere.userData = { aa: aa, count: group.length };
        groups.clusterCenters.add(sphere);
    }
}

function toggleHierarchyTree() {
    const visible = document.getElementById('showHierarchyTree').checked;
    groups.hierarchyTree.visible = visible;
    if (visible && groups.hierarchyTree.children.length === 0) {
        createHierarchyTree();
    }
}

function createHierarchyTree() {
    groups.hierarchyTree.clear();
    if (!state.visualizationData) return;

    const projectionMode = document.getElementById('projectionMode').value;
    const points = state.visualizationData.points;

    // Create radial lines from center to each depth level
    const material = new THREE.LineBasicMaterial({
        color: 0x444466,
        transparent: true,
        opacity: 0.3
    });

    // Group points by depth
    const depthGroups = {};
    points.forEach((p, i) => {
        const d = p.depth;
        if (!depthGroups[d]) depthGroups[d] = [];
        depthGroups[d].push({ point: p, index: i });
    });

    // Draw connections between adjacent depth levels
    const center = new THREE.Vector3(0, projectionMode === 'hemisphere' ? 1 : 0, 0);

    for (let depth = 0; depth <= 8; depth++) {
        const currentGroup = depthGroups[depth] || [];
        const nextGroup = depthGroups[depth + 1] || [];

        if (currentGroup.length === 0 || nextGroup.length === 0) continue;

        // Connect some points between levels (sample to avoid clutter)
        const sampleSize = Math.min(3, currentGroup.length, nextGroup.length);
        for (let i = 0; i < sampleSize; i++) {
            const p1Data = currentGroup[i % currentGroup.length];
            const p2Data = nextGroup[i % nextGroup.length];

            const pos1 = Geometry.getProjectedPosition(p1Data.point, p1Data.index, projectionMode);
            const pos2 = Geometry.getProjectedPosition(p2Data.point, p2Data.index, projectionMode);

            const lineGeo = new THREE.BufferGeometry().setFromPoints([pos1, pos2]);
            const line = new THREE.Line(lineGeo, material);
            groups.hierarchyTree.add(line);
        }
    }

    // Draw radial spokes from center to outermost ring
    const outerPoints = depthGroups[0] || [];
    const spokeCount = Math.min(8, outerPoints.length);
    for (let i = 0; i < spokeCount; i++) {
        const idx = Math.floor(i * outerPoints.length / spokeCount);
        const pData = outerPoints[idx];
        const pos = Geometry.getProjectedPosition(pData.point, pData.index, projectionMode);

        const lineGeo = new THREE.BufferGeometry().setFromPoints([center, pos]);
        const line = new THREE.Line(lineGeo, material);
        groups.hierarchyTree.add(line);
    }
}

function toggleDepthRings() {
    groups.depthRings.visible = document.getElementById('showDepthRings').checked;
}

function onFiberModeChange() {
    const mode = document.getElementById('fiberMode').value;
    const checkbox = document.getElementById('showFibers');

    // Auto-enable fibers when selecting a mode other than 'none'
    if (mode !== 'none' && !checkbox.checked) {
        checkbox.checked = true;
        groups.fibers.visible = true;
    }

    // Auto-disable when selecting 'none'
    if (mode === 'none' && checkbox.checked) {
        checkbox.checked = false;
        groups.fibers.visible = false;
    }

    loadEdges(mode);
}

function onProjectionChange() {
    const mode = document.getElementById('projectionMode').value;

    // Update dynamic labels based on projection mode
    updateDynamicLabels(mode);

    // Recreate visualizations
    createDepthRings();
    createPoints();
    renderFibers();
    if (groups.clusterCenters.visible) createClusterCenters();
    if (groups.hierarchyTree.visible) createHierarchyTree();
    if (groups.mutationNetwork && groups.mutationNetwork.visible) renderMutationNetwork();
    if (state.currentEncodedSequence.length > 0) {
        drawTrajectory(state.currentEncodedSequence);
    }
}

function updateDynamicLabels(projectionMode) {
    const depthRingsLabel = document.getElementById('labelDepthRings');
    const hierarchyLabel = document.getElementById('labelHierarchyTree');
    const clusterLabel = document.getElementById('labelClusterCenters');

    if (depthRingsLabel) {
        switch (projectionMode) {
            case 'pca':
                depthRingsLabel.textContent = 'Axis Guides';
                break;
            case 'hemisphere':
                depthRingsLabel.textContent = 'Latitude Rings';
                break;
            default:
                depthRingsLabel.textContent = 'Depth Rings';
        }
    }

    if (hierarchyLabel) {
        switch (projectionMode) {
            case 'pca':
                hierarchyLabel.textContent = 'Structure Lines';
                break;
            default:
                hierarchyLabel.textContent = 'Hierarchy Tree';
        }
    }

    if (clusterLabel) {
        clusterLabel.textContent = 'AA Centroids';
    }
}

function onColorModeChange() {
    updateColors();
    updateControlRelevance();
}

function updateControlRelevance() {
    const colorMode = document.getElementById('colorBy').value;
    const clusterRow = document.getElementById('clusterCentersRow');

    // Dim cluster centers when not in amino acid mode
    if (clusterRow) {
        if (colorMode === 'amino_acid') {
            clusterRow.style.opacity = '1';
        } else {
            clusterRow.style.opacity = '0.5';
        }
    }
}

// ============== FILTER FUNCTIONS ==============

function getFilterState() {
    const minDepth = parseInt(document.getElementById('depthFilterMin').value, 10);
    const maxDepth = parseInt(document.getElementById('depthFilterMax').value, 10);
    const aaFilter = document.getElementById('aminoAcidFilter').value;

    return {
        minDepth: Math.min(minDepth, maxDepth),
        maxDepth: Math.max(minDepth, maxDepth),
        aminoAcid: aaFilter
    };
}

function onDepthFilterChange() {
    const minDepth = parseInt(document.getElementById('depthFilterMin').value, 10);
    const maxDepth = parseInt(document.getElementById('depthFilterMax').value, 10);

    // Ensure min <= max
    const actualMin = Math.min(minDepth, maxDepth);
    const actualMax = Math.max(minDepth, maxDepth);

    // Update label
    const label = document.getElementById('depthFilterLabel');
    if (label) {
        if (actualMin === actualMax) {
            label.textContent = `${actualMin}`;
        } else {
            label.textContent = `${actualMin}-${actualMax}`;
        }
    }

    applyFilters();
}

function onAminoAcidFilterChange() {
    applyFilters();
}

function applyFilters() {
    if (!pointsMesh || !pointsGeometry || !state.visualizationData) return;

    const filter = getFilterState();
    const data = state.visualizationData.points;
    const sizes = [];

    data.forEach((p, i) => {
        const aa = p.amino_acid || p.aa;
        const depth = p.depth;

        // Check if point passes filters
        const passesDepth = depth >= filter.minDepth && depth <= filter.maxDepth;
        const passesAA = filter.aminoAcid === 'all' || aa === filter.aminoAcid;
        const visible = passesDepth && passesAA;

        // Set size to 0 for filtered out points, normal size for visible
        sizes.push(visible ? 0.08 : 0);
    });

    pointsGeometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
    pointsGeometry.attributes.size.needsUpdate = true;

    // Update info with filter stats
    const visibleCount = sizes.filter(s => s > 0).length;
    UI.updateFilterInfo(visibleCount, data.length);
}

function resetFilters() {
    // Reset depth sliders
    document.getElementById('depthFilterMin').value = 0;
    document.getElementById('depthFilterMax').value = 9;
    document.getElementById('depthFilterLabel').textContent = '0-9';

    // Reset AA filter
    document.getElementById('aminoAcidFilter').value = 'all';

    // Reapply filters (which will show all)
    applyFilters();
}

function clearSequence() {
    document.getElementById('dnaInput').value = '';
    state.currentEncodedSequence = [];
    groups.trajectory.clear();
}

function resetView() {
    controls.reset();
}

// ACTB Reference and Overlay Functions

async function loadActbReference() {
    try {
        const data = await API.fetchActbReference();
        if (data && data.variants) {
            state.actbVariants = data.variants;
            populateVariantSelector(data.variants);
        }
    } catch (e) {
        console.error('Failed to load ACTB reference:', e);
        alert('Failed to load ACTB reference data');
    }
}

function populateVariantSelector(variants) {
    const select = document.getElementById('variantSelect');
    if (!select) return;

    select.innerHTML = '<option value="">Select variant...</option>';
    variants.forEach((v, idx) => {
        const opt = document.createElement('option');
        opt.value = idx.toString();
        opt.textContent = v.name || `Variant ${idx + 1}`;
        select.appendChild(opt);
    });
    select.style.display = 'block';
}

async function onVariantSelectChange() {
    const select = document.getElementById('variantSelect');
    const idx = parseInt(select.value, 10);

    if (isNaN(idx) || !state.actbVariants || idx >= state.actbVariants.length) {
        return;
    }

    const variant = state.actbVariants[idx];
    if (variant.sequence) {
        try {
            const result = await API.encodeDnaSequence(variant.sequence);
            drawOverlayTrajectory(result, variant.name);
        } catch (e) {
            console.error('Failed to encode variant:', e);
        }
    }
}

function drawOverlayTrajectory(encoded, label) {
    groups.overlayTrajectory.clear();
    if (!encoded || encoded.length === 0) return;

    const positions = [];

    encoded.forEach(item => {
        const proj = item.projection;
        if (proj && proj.length >= 3) {
            positions.push(new THREE.Vector3(proj[0], proj[1], proj[2]));
        }
    });

    if (positions.length < 2) return;

    const pathPoints = [];
    const style = document.getElementById('trajectoryStyle').value;

    for (let i = 0; i < positions.length - 1; i++) {
        const p1 = positions[i];
        const p2 = positions[i+1];

        let segmentPoints = [];
        if (style === 'geodesic') {
            segmentPoints = Geometry.computeGeodesicArc(p1, p2, 20);
        } else {
            segmentPoints = [p1, p2];
        }

        pathPoints.push(...segmentPoints);
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(pathPoints);
    const material = new THREE.LineBasicMaterial({
        color: 0xff6600,  // Orange for overlay
        linewidth: 2
    });

    const line = new THREE.Line(geometry, material);
    groups.overlayTrajectory.add(line);

    // Markers
    encoded.forEach((item, i) => {
        const pos = positions[i];
        if (pos) {
            const markerGeo = new THREE.SphereGeometry(0.012, 8, 8);
            const markerMat = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
            const marker = new THREE.Mesh(markerGeo, markerMat);
            marker.position.copy(pos);
            groups.overlayTrajectory.add(marker);
        }
    });
}

function clearOverlay() {
    groups.overlayTrajectory.clear();
    const select = document.getElementById('variantSelect');
    if (select) {
        select.value = '';
    }
}

async function loadVarianceData() {
    try {
        const data = await API.fetchAngularVariance();
        if (data) {
            state.varianceData = data;
            displayVarianceInfo(data);
        }
    } catch (e) {
        console.error('Failed to load variance data:', e);
        alert('Failed to load variance data');
    }
}

function displayVarianceInfo(data) {
    // API returns { summary: { mean_intra_variance, mean_inter_variance, mean_ratio } }
    const summary = data.summary || data;
    UI.updateInfoPanel({
        intraVariance: summary.mean_intra_variance || summary.intra_class_variance || 0,
        interVariance: summary.mean_inter_variance || summary.inter_class_variance || 0,
        varianceRatio: summary.mean_ratio || summary.variance_ratio || 0
    });
}

// =============================================================================
// NEW FEATURES - Interactive Analysis Tools
// =============================================================================

// State for new features
let selectedCodon = null;
let measureModeActive = false;
let measurePoint1 = null;
let animationInterval = null;
let animationIndex = 0;
let highlightedIndices = new Set();

// ============== CLICK TO HIGHLIGHT SYNONYMOUS CODONS ==============

function onCodonClick(event) {
    if (!pointsMesh || !pointsMesh.visible || !state.visualizationData) return;

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);

    const intersects = raycaster.intersectObject(pointsMesh);
    if (intersects.length > 0) {
        const idx = intersects[0].index;
        const point = state.visualizationData.points[idx];

        if (measureModeActive) {
            handleMeasureClick(point, idx);
        } else {
            highlightSynonymousCodons(point);
        }
    } else if (!measureModeActive) {
        clearHighlight();
    }
}

function highlightSynonymousCodons(clickedPoint) {
    if (!state.visualizationData) return;

    selectedCodon = clickedPoint.codon;
    const aa = clickedPoint.amino_acid;
    const synonymousCodons = Globals.getCodonsForAA(aa);

    // Find all indices of synonymous codons
    highlightedIndices.clear();
    state.visualizationData.points.forEach((p, idx) => {
        if (synonymousCodons.includes(p.codon)) {
            highlightedIndices.add(idx);
        }
    });

    // Update point sizes to highlight
    updatePointHighlights();

    // Show info
    const searchResults = document.getElementById('searchResults');
    if (searchResults) {
        searchResults.innerHTML = `<span style="color: #00ff88;">${aa} (${Globals.AA_NAMES[aa]}): ${synonymousCodons.join(', ')}</span>`;
    }
}

function updatePointHighlights() {
    if (!pointsGeometry) return;

    const sizes = [];
    const data = state.visualizationData.points;

    data.forEach((_, idx) => {
        if (highlightedIndices.has(idx)) {
            sizes.push(0.15); // Larger size for highlighted
        } else if (highlightedIndices.size > 0) {
            sizes.push(0.04); // Smaller for non-highlighted when something is selected
        } else {
            sizes.push(0.08); // Default size
        }
    });

    pointsGeometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));
    pointsGeometry.attributes.size.needsUpdate = true;
}

function clearHighlight() {
    selectedCodon = null;
    highlightedIndices.clear();
    updatePointHighlights();

    const searchResults = document.getElementById('searchResults');
    if (searchResults) searchResults.innerHTML = '';
}

// ============== SEARCH FUNCTIONALITY ==============

function onCodonSearch() {
    const input = document.getElementById('codonSearch');
    const resultsDiv = document.getElementById('searchResults');
    const query = input.value.toUpperCase().trim();

    if (!query || !state.visualizationData) {
        clearHighlight();
        return;
    }

    highlightedIndices.clear();

    // Search by codon, amino acid letter, or amino acid name
    state.visualizationData.points.forEach((p, idx) => {
        const aa = p.amino_acid;
        const aaName = Globals.AA_NAMES[aa] || '';

        if (p.codon.includes(query) ||
            aa === query ||
            aaName.toUpperCase().includes(query)) {
            highlightedIndices.add(idx);
        }
    });

    updatePointHighlights();

    if (highlightedIndices.size > 0) {
        resultsDiv.innerHTML = `<span style="color: #88ccff;">Found ${highlightedIndices.size} codons</span>`;
    } else {
        resultsDiv.innerHTML = `<span style="color: #ff8888;">No matches</span>`;
    }
}

// ============== KEYBOARD SHORTCUTS ==============

function onKeyDown(event) {
    // Don't trigger if typing in input
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        if (event.key === 'Escape') {
            event.target.blur();
            clearHighlight();
        }
        return;
    }

    switch (event.key.toLowerCase()) {
        case 'f':
            // Focus search
            event.preventDefault();
            document.getElementById('codonSearch').focus();
            break;
        case 'r':
            // Reset view
            resetView();
            break;
        case ' ':
            // Play/Pause animation
            event.preventDefault();
            toggleAnimation();
            break;
        case '1':
            document.getElementById('projectionMode').value = 'poincare';
            onProjectionChange();
            break;
        case '2':
            document.getElementById('projectionMode').value = 'pca';
            onProjectionChange();
            break;
        case '3':
            document.getElementById('projectionMode').value = 'hemisphere';
            onProjectionChange();
            break;
        case 'escape':
            clearHighlight();
            if (measureModeActive) {
                document.getElementById('measureMode').checked = false;
                onMeasureModeChange();
            }
            break;
        case 'm':
            // Toggle measure mode
            const measureCheckbox = document.getElementById('measureMode');
            measureCheckbox.checked = !measureCheckbox.checked;
            onMeasureModeChange();
            break;
        case 'n':
            // Toggle mutation network
            const networkCheckbox = document.getElementById('showMutationNetwork');
            networkCheckbox.checked = !networkCheckbox.checked;
            toggleMutationNetwork();
            break;
    }
}

// ============== DISTANCE MEASUREMENT ==============

function onMeasureModeChange() {
    measureModeActive = document.getElementById('measureMode').checked;
    measurePoint1 = null;

    const resultDiv = document.getElementById('measureResult');
    if (measureModeActive) {
        resultDiv.innerHTML = 'Click first codon...';
        renderer.domElement.style.cursor = 'crosshair';
    } else {
        resultDiv.innerHTML = '';
        renderer.domElement.style.cursor = 'default';
        groups.measureLine && groups.measureLine.clear && groups.measureLine.clear();
    }
}

function handleMeasureClick(point, idx) {
    const resultDiv = document.getElementById('measureResult');

    if (!measurePoint1) {
        measurePoint1 = { point, idx };
        resultDiv.innerHTML = `First: ${point.codon} (${point.amino_acid}). Click second...`;
    } else {
        const p1 = measurePoint1.point;
        const p2 = point;

        // Calculate Euclidean distance in projected space
        const projMode = document.getElementById('projectionMode').value;
        const pos1 = Geometry.getProjectedPosition(p1, measurePoint1.idx, projMode);
        const pos2 = Geometry.getProjectedPosition(p2, idx, projMode);
        const proj1 = [pos1.x, pos1.y, pos1.z];
        const proj2 = [pos2.x, pos2.y, pos2.z];

        const dx = proj2[0] - proj1[0];
        const dy = proj2[1] - proj1[1];
        const dz = proj2[2] - proj1[2];
        const euclidean = Math.sqrt(dx * dx + dy * dy + dz * dz);

        // Calculate geodesic distance (hyperbolic)
        const geodesic = Geometry.hyperbolicDistance ?
            Geometry.hyperbolicDistance(proj1, proj2) : euclidean;

        // Draw measurement line
        drawMeasureLine(proj1, proj2);

        resultDiv.innerHTML = `
            <div>${p1.codon}→${p2.codon}</div>
            <div>Euclidean: ${euclidean.toFixed(4)}</div>
            <div>Depth diff: ${Math.abs(p1.depth - p2.depth)}</div>
        `;

        measurePoint1 = null;
    }
}

function drawMeasureLine(p1, p2) {
    // Create or clear measure line group
    if (!groups.measureLine) {
        groups.measureLine = new THREE.Group();
        scene.add(groups.measureLine);
    }
    groups.measureLine.clear();

    const material = new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 2 });
    const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(p1[0], p1[1], p1[2]),
        new THREE.Vector3(p2[0], p2[1], p2[2])
    ]);
    const line = new THREE.Line(geometry, material);
    groups.measureLine.add(line);
}

// ============== MUTATION NETWORK ==============

function toggleMutationNetwork() {
    const show = document.getElementById('showMutationNetwork').checked;

    if (!groups.mutationNetwork) {
        groups.mutationNetwork = new THREE.Group();
        scene.add(groups.mutationNetwork);
    }

    if (show) {
        renderMutationNetwork();
    } else {
        groups.mutationNetwork.clear();
    }

    groups.mutationNetwork.visible = show;
}

function renderMutationNetwork() {
    if (!groups.mutationNetwork) return;
    groups.mutationNetwork.clear();

    const projectionMode = document.getElementById('projectionMode').value;
    const points = state.visualizationData.points;

    // Build position lookup
    const codonToPosition = {};
    points.forEach((p, idx) => {
        const pos = Geometry.getProjectedPosition(p, idx, projectionMode);
        codonToPosition[p.codon] = [pos.x, pos.y, pos.z];
    });

    // Create edges for single-nucleotide mutations
    const edgeSet = new Set();
    const positions = [];

    for (const codon of Object.keys(codonToPosition)) {
        const neighbors = Globals.getSingleMutationNeighbors(codon);
        for (const neighbor of neighbors) {
            if (codonToPosition[neighbor.codon]) {
                const edgeKey = [codon, neighbor.codon].sort().join('-');
                if (!edgeSet.has(edgeKey)) {
                    edgeSet.add(edgeKey);

                    const p1 = codonToPosition[codon];
                    const p2 = codonToPosition[neighbor.codon];

                    positions.push(p1[0], p1[1], p1[2]);
                    positions.push(p2[0], p2[1], p2[2]);
                }
            }
        }
    }

    if (positions.length > 0) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

        const material = new THREE.LineBasicMaterial({
            color: 0x444466,
            transparent: true,
            opacity: 0.3
        });

        const lines = new THREE.LineSegments(geometry, material);
        groups.mutationNetwork.add(lines);
    }
}

// ============== SEQUENCE ANIMATION ==============

function toggleAnimation() {
    const btn = document.getElementById('animateBtn');
    const statusDiv = document.getElementById('animationStatus');

    if (animationInterval) {
        // Stop animation
        clearInterval(animationInterval);
        animationInterval = null;
        btn.textContent = '▶ Play';
        statusDiv.innerHTML = '';
        clearHighlight();
    } else {
        // Start animation
        if (state.currentEncodedSequence.length === 0) {
            statusDiv.innerHTML = '<span style="color: #ff8888;">Encode a sequence first!</span>';
            return;
        }

        animationIndex = 0;
        const speed = parseInt(document.getElementById('animationSpeed').value, 10);
        btn.textContent = '⏸ Pause';

        animationInterval = setInterval(() => {
            if (animationIndex >= state.currentEncodedSequence.length) {
                // Loop back
                animationIndex = 0;
            }

            const item = state.currentEncodedSequence[animationIndex];
            highlightedIndices.clear();

            // Find this codon in visualization data
            state.visualizationData.points.forEach((p, idx) => {
                if (p.codon === item.codon) {
                    highlightedIndices.add(idx);
                }
            });

            updatePointHighlights();

            statusDiv.innerHTML = `Position ${animationIndex + 1}/${state.currentEncodedSequence.length}: <span style="color: #00ff88;">${item.codon}</span> → ${item.amino_acid}`;

            animationIndex++;
        }, speed);
    }
}

function onAnimationSpeedChange() {
    const speed = document.getElementById('animationSpeed').value;
    document.getElementById('speedLabel').textContent = speed + 'ms';

    // If animation is running, restart with new speed
    if (animationInterval) {
        toggleAnimation(); // Stop
        toggleAnimation(); // Restart with new speed
    }
}

// ============== EXPORT FUNCTIONS ==============

function exportPNG() {
    // Render at higher resolution
    const originalWidth = renderer.domElement.width;
    const originalHeight = renderer.domElement.height;

    renderer.setSize(originalWidth * 2, originalHeight * 2);
    renderer.render(scene, camera);

    const link = document.createElement('a');
    link.download = `codon-visualization-${Date.now()}.png`;
    link.href = renderer.domElement.toDataURL('image/png');
    link.click();

    // Restore original size
    renderer.setSize(originalWidth, originalHeight);
}

function exportSVG() {
    // SVG export is complex for 3D; provide a simplified 2D projection
    const points = state.visualizationData.points;
    const projectionMode = document.getElementById('projectionMode').value;

    let svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1.5 -1.5 3 3" width="800" height="800">
<rect x="-1.5" y="-1.5" width="3" height="3" fill="#0a0a0f"/>
<g transform="scale(1,-1)">`;

    points.forEach((p, idx) => {
        const pos = Geometry.getProjectedPosition(p, idx, projectionMode);
        const color = '#' + Globals.DEPTH_COLORS[p.depth].toString(16).padStart(6, '0');
        svg += `<circle cx="${pos.x.toFixed(3)}" cy="${pos.z.toFixed(3)}" r="0.02" fill="${color}" title="${p.codon}"/>`;
    });

    svg += '</g></svg>';

    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const link = document.createElement('a');
    link.download = `codon-visualization-${Date.now()}.svg`;
    link.href = URL.createObjectURL(blob);
    link.click();
}

function exportCSV() {
    if (!state.visualizationData) return;

    const projectionMode = document.getElementById('projectionMode').value;
    let csv = 'codon,amino_acid,depth,confidence,margin,x,y,z\n';

    state.visualizationData.points.forEach((p, idx) => {
        const pos = Geometry.getProjectedPosition(p, idx, projectionMode);
        csv += `${p.codon},${p.amino_acid},${p.depth},${(p.confidence || 0).toFixed(4)},${(p.margin || 0).toFixed(4)},${pos.x.toFixed(6)},${pos.y.toFixed(6)},${pos.z.toFixed(6)}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.download = `codon-data-${Date.now()}.csv`;
    link.href = URL.createObjectURL(blob);
    link.click();
}

function exportJSON() {
    if (!state.visualizationData) return;

    const projectionMode = document.getElementById('projectionMode').value;
    const exportData = {
        metadata: state.visualizationData.metadata,
        projection: projectionMode,
        timestamp: new Date().toISOString(),
        points: state.visualizationData.points.map((p, idx) => {
            const pos = Geometry.getProjectedPosition(p, idx, projectionMode);
            return {
                codon: p.codon,
                amino_acid: p.amino_acid,
                depth: p.depth,
                confidence: p.confidence,
                margin: p.margin,
                embedding: p.embedding,
                projection: { x: pos.x, y: pos.y, z: pos.z }
            };
        })
    };

    if (state.currentEncodedSequence.length > 0) {
        exportData.encoded_sequence = state.currentEncodedSequence;
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.download = `codon-data-${Date.now()}.json`;
    link.href = URL.createObjectURL(blob);
    link.click();
}

// ============== ORGANISM CODON USAGE ==============

function onOrganismChange() {
    const organism = document.getElementById('organismSelect').value;
    updateUsageStats(organism);

    // If we have an encoded sequence, recalculate stats
    if (state.currentEncodedSequence.length > 0) {
        calculateSequenceStats(organism);
    }
}

function updateUsageStats(organism) {
    const statsDiv = document.getElementById('usageStats');
    const usage = Globals.getCodonUsage(organism);

    if (!usage) {
        statsDiv.innerHTML = '';
        return;
    }

    // Show top 3 most/least used codons
    const sorted = Object.entries(usage).sort((a, b) => b[1] - a[1]);
    const top3 = sorted.slice(0, 3);
    const bottom3 = sorted.filter(([c]) => Globals.CODON_TABLE[c] !== '*').slice(-3);

    statsDiv.innerHTML = `
        <div style="color: #00ff88;">Top: ${top3.map(([c, f]) => `${c}(${f.toFixed(0)})`).join(', ')}</div>
        <div style="color: #ff8888;">Rare: ${bottom3.map(([c, f]) => `${c}(${f.toFixed(0)})`).join(', ')}</div>
    `;
}

function calculateSequenceStats(organism) {
    const usage = Globals.getCodonUsage(organism);
    const codons = state.currentEncodedSequence.map(e => e.codon);
    const statsDiv = document.getElementById('usageStats');

    if (!usage || codons.length === 0) return;

    const cai = Globals.calculateCAI(codons, usage);
    const enc = Globals.calculateENC(codons);

    statsDiv.innerHTML += `
        <div style="margin-top: 4px; border-top: 1px solid #333; padding-top: 4px;">
            <div>CAI: <span style="color: #ffcc44;">${cai.toFixed(3)}</span></div>
            <div>ENC: <span style="color: #ffcc44;">${enc.toFixed(1)}</span></div>
        </div>
    `;
}
