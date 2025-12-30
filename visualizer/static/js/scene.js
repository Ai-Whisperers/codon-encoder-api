import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function setupScene(containerId) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0f);

    const camera = new THREE.PerspectiveCamera(
        60, window.innerWidth / window.innerHeight, 0.1, 1000
    );
    camera.position.set(0, 2, 2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById(containerId).appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0);

    // Zoom limits
    controls.minDistance = 0.5;
    controls.maxDistance = 10;

    // Enable all controls
    controls.enableZoom = true;
    controls.enableRotate = true;
    controls.enablePan = true;

    // Zoom speed (slower for smoother control)
    controls.zoomSpeed = 0.5;
    controls.rotateSpeed = 0.8;

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.05;

    // Groups
    const groups = {
        depthRings: new THREE.Group(),
        hierarchyTree: new THREE.Group(),
        trajectory: new THREE.Group(),
        fibers: new THREE.Group(),
        clusterCenters: new THREE.Group(),
        overlayTrajectory: new THREE.Group()
    };

    scene.add(groups.depthRings);
    scene.add(groups.hierarchyTree);
    scene.add(groups.trajectory);
    scene.add(groups.overlayTrajectory);
    scene.add(groups.fibers);
    scene.add(groups.clusterCenters);
    
    groups.fibers.visible = false;
    groups.clusterCenters.visible = false;

    // Lighting
    const ambient = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambient);
    const directional = new THREE.DirectionalLight(0xffffff, 0.5);
    directional.position.set(1, 2, 1);
    scene.add(directional);

    return { scene, camera, renderer, controls, raycaster, groups };
}

export function onWindowResize(camera, renderer) {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}
