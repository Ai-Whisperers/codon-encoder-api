/**
 * UI Utilities
 */
import {
    DEPTH_COLORS,
    DEGENERACY_COLORS,
    WOBBLE_COLORS,
    CHEMICAL_CLASS_COLORS
} from './globals.js';
import * as Geometry from './geometry.js';

export function buildLegend(mode = 'depth', data = null) {
    const container = document.getElementById('legendContent');
    const title = document.querySelector('#legend h4');
    if (!container) return;

    container.innerHTML = '';

    if (mode === 'depth') {
        if (title) title.textContent = 'Depth Level';
        for (let v = 0; v <= 9; v++) {
            const item = document.createElement('div');
            item.className = 'legend-item';
            const r = Geometry.depthToPoincareRadius(v);
            item.innerHTML = `
                <div class="legend-color" style="background: #${DEPTH_COLORS[v].toString(16).padStart(6, '0')}"></div>
                <span>v=${v} (r=${r.toFixed(2)})</span>
            `;
            container.appendChild(item);
        }
    } else if (mode === 'amino_acid') {
        if (title) title.textContent = 'Amino Acid';
        const aaGroups = {
            'Hydrophobic': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'],
            'Polar': ['S', 'T', 'N', 'Q', 'Y', 'C', 'G'],
            'Basic (+)': ['K', 'R', 'H'],
            'Acidic (-)': ['D', 'E'],
            'Stop': ['*']
        };
        const groupColors = {
            'Hydrophobic': '#8B8B00',
            'Polar': '#00CED1',
            'Basic (+)': '#0000FF',
            'Acidic (-)': '#FF0000',
            'Stop': '#808080'
        };
        for (const [group, aas] of Object.entries(aaGroups)) {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${groupColors[group]}"></div>
                <span>${group} (${aas.join(', ')})</span>
            `;
            container.appendChild(item);
        }
    } else if (mode === 'margin') {
        if (title) title.textContent = 'Basin Margin';
        const gradientSteps = ['Low', 'Medium', 'High'];
        const colors = ['#0066ff', '#9933ff', '#ff0000'];
        gradientSteps.forEach((label, i) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${colors[i]}"></div>
                <span>${label}</span>
            `;
            container.appendChild(item);
        });
    } else if (mode === 'confidence') {
        if (title) title.textContent = 'Confidence';
        const gradientSteps = ['Low', 'Medium', 'High'];
        const colors = ['#ff0000', '#ffff00', '#00ff00'];
        gradientSteps.forEach((label, i) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${colors[i]}"></div>
                <span>${label}</span>
            `;
            container.appendChild(item);
        });
    } else if (mode === 'sequence') {
        if (title) title.textContent = 'Sequence Position';
        const gradientSteps = ['Start', 'Middle', 'End'];
        const colors = ['#ff0000', '#00ff00', '#0000ff'];
        gradientSteps.forEach((label, i) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${colors[i]}"></div>
                <span>${label}</span>
            `;
            container.appendChild(item);
        });
    } else if (mode === 'degeneracy') {
        if (title) title.textContent = 'Codon Degeneracy';
        const degLevels = [
            { fold: 1, label: '1-fold (M, W)', color: DEGENERACY_COLORS[1] },
            { fold: 2, label: '2-fold (C, D, E, F, H, K, N, Q, Y)', color: DEGENERACY_COLORS[2] },
            { fold: 3, label: '3-fold (I, Stop)', color: DEGENERACY_COLORS[3] },
            { fold: 4, label: '4-fold (A, G, P, T, V)', color: DEGENERACY_COLORS[4] },
            { fold: 6, label: '6-fold (L, R, S)', color: DEGENERACY_COLORS[6] }
        ];
        degLevels.forEach(({ label, color }) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: #${color.toString(16).padStart(6, '0')}"></div>
                <span>${label}</span>
            `;
            container.appendChild(item);
        });
    } else if (mode === 'gc_content') {
        if (title) title.textContent = 'GC Content';
        const gcLevels = [
            { pct: '0%', label: 'AT-rich (AAA, AAT...)', color: '#ff0000' },
            { pct: '33%', label: '1 G/C', color: '#ff8800' },
            { pct: '67%', label: '2 G/C', color: '#88ff00' },
            { pct: '100%', label: 'GC-rich (GGG, GGC...)', color: '#00ff00' }
        ];
        gcLevels.forEach(({ pct, label, color }) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${color}"></div>
                <span>${pct} - ${label}</span>
            `;
            container.appendChild(item);
        });
    } else if (mode === 'wobble') {
        if (title) title.textContent = 'Wobble Base (3rd Position)';
        const wobbleBases = [
            { base: 'A', label: 'Adenine', color: WOBBLE_COLORS['A'] },
            { base: 'T', label: 'Thymine', color: WOBBLE_COLORS['T'] },
            { base: 'G', label: 'Guanine', color: WOBBLE_COLORS['G'] },
            { base: 'C', label: 'Cytosine', color: WOBBLE_COLORS['C'] }
        ];
        wobbleBases.forEach(({ base, label, color }) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: #${color.toString(16).padStart(6, '0')}"></div>
                <span>${base} - ${label}</span>
            `;
            container.appendChild(item);
        });
    } else if (mode === 'hydrophobicity') {
        if (title) title.textContent = 'Hydrophobicity (Kyte-Doolittle)';
        const hydroLevels = [
            { label: 'Hydrophilic (-4.5)', color: '#0000ff' },
            { label: 'Neutral (0)', color: '#ffffff' },
            { label: 'Hydrophobic (+4.5)', color: '#ff0000' }
        ];
        hydroLevels.forEach(({ label, color }) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${color}; border: 1px solid #444;"></div>
                <span>${label}</span>
            `;
            container.appendChild(item);
        });
    } else if (mode === 'molecular_weight') {
        if (title) title.textContent = 'Molecular Weight (Da)';
        const mwLevels = [
            { label: 'Light (~75 Da, Gly)', color: '#00ffff' },
            { label: 'Medium (~130 Da)', color: '#00ff00' },
            { label: 'Heavy (~200 Da, Trp)', color: '#ff00ff' }
        ];
        mwLevels.forEach(({ label, color }) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${color}"></div>
                <span>${label}</span>
            `;
            container.appendChild(item);
        });
    } else if (mode === 'chemical_class') {
        if (title) title.textContent = 'Chemical Classification';
        const classes = [
            { name: 'Aliphatic', aas: 'G,A,V,L,I,P', color: CHEMICAL_CLASS_COLORS['aliphatic'] },
            { name: 'Aromatic', aas: 'F,Y,W', color: CHEMICAL_CLASS_COLORS['aromatic'] },
            { name: 'Sulfur', aas: 'C,M', color: CHEMICAL_CLASS_COLORS['sulfur'] },
            { name: 'Hydroxyl', aas: 'S,T', color: CHEMICAL_CLASS_COLORS['hydroxyl'] },
            { name: 'Amide', aas: 'N,Q', color: CHEMICAL_CLASS_COLORS['amide'] },
            { name: 'Basic (+)', aas: 'K,R,H', color: CHEMICAL_CLASS_COLORS['basic'] },
            { name: 'Acidic (-)', aas: 'D,E', color: CHEMICAL_CLASS_COLORS['acidic'] },
            { name: 'Stop', aas: '*', color: CHEMICAL_CLASS_COLORS['stop'] }
        ];
        classes.forEach(({ name, aas, color }) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: #${color.toString(16).padStart(6, '0')}"></div>
                <span>${name} (${aas})</span>
            `;
            container.appendChild(item);
        });
    } else if (mode === 'embedding_norm') {
        if (title) title.textContent = 'Embedding Norm';
        const normLevels = [
            { label: 'Low (near origin)', color: '#0000ff' },
            { label: 'Medium', color: '#00ff00' },
            { label: 'High (peripheral)', color: '#ff0000' }
        ];
        normLevels.forEach(({ label, color }) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${color}"></div>
                <span>${label}</span>
            `;
            container.appendChild(item);
        });
    }
}

export function validateDnaInput() {
    const input = document.getElementById('dnaInput');
    const validation = document.getElementById('dnaValidation');
    if (!input || !validation) return;

    const seq = input.value.toUpperCase();
    const clean = seq.replace(/[^ATCG]/g, '');
    
    if (seq !== clean) {
        validation.textContent = "Invalid characters detected (only A, T, C, G allowed)";
        validation.style.color = "#ff4444";
    } else if (clean.length > 0 && clean.length % 3 !== 0) {
        validation.textContent = `Length ${clean.length} is not a multiple of 3 (frameshift risk)`;
        validation.style.color = "#ffcc00";
    } else {
        validation.textContent = clean.length > 0 ? `${clean.length} bp (${clean.length/3} codons)` : "";
        validation.style.color = "#88ccff";
    }
}

export function updateSequenceList(loadedSequences) {
    const select = document.getElementById('sequenceList');
    if (!select) return;

    // Rebuild options
    select.innerHTML = '';
    loadedSequences.forEach((item, idx) => {
        const opt = document.createElement('option');
        opt.value = idx.toString();
        opt.textContent = item.label;
        select.appendChild(opt);
    });

    if (loadedSequences.length > 0) {
         select.style.display = 'block';
    } else {
         select.style.display = 'none';
    }
}

export function displayTooltip(text, x, y, color) {
    const tooltip = document.getElementById('tooltip');
    if (!tooltip) return;

    if (text) {
        tooltip.style.display = 'block';
        tooltip.style.left = x + 10 + 'px';
        tooltip.style.top = y + 10 + 'px';
        tooltip.innerHTML = text; // Assumes text contains safe HTML or is sanitized
        if (color) tooltip.style.borderColor = color;
    } else {
        tooltip.style.display = 'none';
    }
}

export function updateInfoPanel(stats) {
    if (stats.version) document.getElementById('infoVersion').textContent = stats.version;
    if (stats.structure !== undefined) document.getElementById('infoStructure').textContent = stats.structure.toFixed(4);
    if (stats.seqLen !== undefined) document.getElementById('infoSeqLen').textContent = stats.seqLen;
    if (stats.geoDist !== undefined) document.getElementById('infoGeoDist').textContent = stats.geoDist.toFixed(4);

    // Variance
    if (stats.intraVariance !== undefined) document.getElementById('infoIntraVar').textContent = stats.intraVariance.toFixed(4);
    if (stats.interVariance !== undefined) document.getElementById('infoInterVar').textContent = stats.interVariance.toFixed(4);
    if (stats.varianceRatio !== undefined) document.getElementById('infoVarRatio').textContent = stats.varianceRatio.toFixed(4);
}

export function displayEncodedSequence(encodedSequence) {
    const output = document.getElementById('sequence-output');
    if (!output) return;

    if (!encodedSequence || encodedSequence.length === 0) {
        output.innerHTML = 'Enter DNA sequence above and click Encode';
        return;
    }

    let html = '<table class="sequence-table"><thead><tr>';
    html += '<th>#</th><th>Codon</th><th>AA</th><th>Depth</th><th>Conf</th>';
    html += '</tr></thead><tbody>';

    encodedSequence.forEach((item, idx) => {
        const conf = item.confidence !== undefined ? item.confidence.toFixed(2) : '-';
        html += `<tr>
            <td>${idx + 1}</td>
            <td class="codon-cell">${item.codon}</td>
            <td>${item.amino_acid || '-'}</td>
            <td>${item.depth !== undefined ? item.depth : '-'}</td>
            <td>${conf}</td>
        </tr>`;
    });

    html += '</tbody></table>';

    // Summary
    const protein = encodedSequence.map(e => e.amino_acid || '?').join('');
    html += `<div class="sequence-summary">
        <strong>Protein:</strong> ${protein}<br>
        <strong>Codons:</strong> ${encodedSequence.length}
    </div>`;

    output.innerHTML = html;
}

export function computeGeodesicDistance(encodedSequence) {
    if (!encodedSequence || encodedSequence.length < 2) return 0;

    let totalDist = 0;
    for (let i = 0; i < encodedSequence.length - 1; i++) {
        const p1 = encodedSequence[i].projection;
        const p2 = encodedSequence[i + 1].projection;
        if (p1 && p2) {
            const dx = p2[0] - p1[0];
            const dy = p2[1] - p1[1];
            const dz = p2[2] - p1[2];
            totalDist += Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
    return totalDist;
}

export function updateFilterInfo(visibleCount, totalCount) {
    // Could add a filter info display if desired
    // For now, just log or update existing info panel
    const filterInfo = document.getElementById('filterInfo');
    if (filterInfo) {
        filterInfo.textContent = `${visibleCount}/${totalCount} codons`;
    }
}
