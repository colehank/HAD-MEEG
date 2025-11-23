// Global state
let currentBase = null;
let currentData = null;
let manualLabels = [];
let componentGrid = null;
let allBases = [];
let currentIndex = 0;
let statusData = null; // Store latest status data

// Initialize app
async function init() {
    await loadStatus();
    setupEventListeners();
}

// Load status and current BIDS
async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();

        // Store status data globally
        statusData = status;

        // Update global state
        allBases = status.all_bases;

        // Update progress display
        updateProgressDisplay();

        // Check if there's a saved position in localStorage
        const savedBase = localStorage.getItem('currentBase');
        let baseToLoad = null;

        if (savedBase && allBases.includes(savedBase)) {
            // Use saved position
            baseToLoad = savedBase;
        } else if (status.current_base) {
            // Use default (first unprocessed)
            baseToLoad = status.current_base;
        }

        if (baseToLoad) {
            await loadBIDS(baseToLoad);
        } else {
            alert('All BIDS processed!');
        }

        updateNavigationButtons();
    } catch (error) {
        console.error('Error loading status:', error);
    }
}

// Load BIDS data
async function loadBIDS(base) {
    try {
        currentBase = base;

        // Save current position to localStorage
        localStorage.setItem('currentBase', base);

        // Update current index
        currentIndex = allBases.indexOf(base);

        // Update basename display
        document.getElementById('current-basename').textContent = base;

        // Update position info
        document.getElementById('position-info').textContent =
            `Position: ${currentIndex + 1} / ${allBases.length}`;

        // Load data
        const response = await fetch(`/api/load/${base}`);
        currentData = await response.json();

        // Initialize manual labels with existing labels if available, otherwise use auto labels
        if (currentData.existing_labels && currentData.existing_labels.length > 0) {
            manualLabels = [...currentData.existing_labels];
        } else {
            manualLabels = [...currentData.auto_labels];
        }

        // Load main image
        const mainImage = document.getElementById('main-image');
        mainImage.src = `/api/image/main/${base}`;
        mainImage.onload = setupImageGrid;

        // Update labels table
        updateLabelsTable();

        // Update navigation buttons
        updateNavigationButtons();

    } catch (error) {
        console.error('Error loading BIDS:', error);
    }
}

// Setup image grid for click detection
function setupImageGrid() {
    const img = document.getElementById('main-image');
    const overlay = document.getElementById('image-overlay');

    // Calculate grid based on component count and data type
    const nComponents = currentData.n_components;
    const dtype = currentData.dtype;

    let cols, rows;

    // Determine grid layout based on data type
    if (dtype === 'meg') {
        // MEG: 40 components in 10 columns x 4 rows
        cols = 10;
        rows = 4;
    } else if (dtype === 'eeg') {
        // EEG: 20 components in 5 columns x 4 rows
        cols = 5;
        rows = 4;
    } else {
        // Fallback: auto-calculate
        cols = Math.ceil(Math.sqrt(nComponents));
        rows = Math.ceil(nComponents / cols);
    }

    componentGrid = {
        cols: cols,
        rows: rows,
        width: img.offsetWidth,
        height: img.offsetHeight
    };

    // Clear overlay
    overlay.innerHTML = '';
    overlay.style.width = img.offsetWidth + 'px';
    overlay.style.height = img.offsetHeight + 'px';

    // Create clickable cells
    const cellWidth = componentGrid.width / cols;
    const cellHeight = componentGrid.height / rows;

    for (let i = 0; i < nComponents; i++) {
        const row = Math.floor(i / cols);
        const col = i % cols;
        const compId = currentData.comp_ids[i];

        const cell = document.createElement('div');
        cell.className = 'grid-cell';
        cell.style.left = (col * cellWidth) + 'px';
        cell.style.top = (row * cellHeight) + 'px';
        cell.style.width = cellWidth + 'px';
        cell.style.height = cellHeight + 'px';
        cell.dataset.compId = compId;
        cell.dataset.compIndex = i;

        // Check if modified
        updateCellHighlight(cell, i);

        // Left click - show detail
        cell.addEventListener('click', (e) => {
            e.preventDefault();
            showComponentDetail(compId);
        });

        // Right click - show label menu
        cell.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            showContextMenu(e, compId, i);
        });

        // Hover - show tooltip
        cell.addEventListener('mouseenter', (e) => {
            showTooltip(e, compId, i);
        });

        cell.addEventListener('mouseleave', () => {
            hideTooltip();
        });

        overlay.appendChild(cell);
    }
}

// Show component detail modal
function showComponentDetail(compId) {
    const modal = document.getElementById('modal');
    const modalImage = document.getElementById('modal-image');
    const modalTitle = document.getElementById('modal-title');

    modalTitle.textContent = `Component ${compId}`;
    modalImage.src = `/api/image/comp/${currentBase}/${compId}`;
    modal.classList.remove('hidden');
}

// Update cell highlight based on modification status
function updateCellHighlight(cell, compIndex) {
    const isModified = manualLabels[compIndex] !== currentData.auto_labels[compIndex];
    if (isModified) {
        cell.classList.add('modified');
    } else {
        cell.classList.remove('modified');
    }
}

// Show tooltip on hover
function showTooltip(event, compId, compIndex) {
    const tooltip = document.getElementById('grid-tooltip');
    const label = manualLabels[compIndex];
    const autoLabel = currentData.auto_labels[compIndex];
    const isModified = label !== autoLabel;

    // Build tooltip content
    let content = `Component ${compId}: ${label}`;
    if (isModified) {
        content += ` (was: ${autoLabel})`;
    }

    tooltip.textContent = content;

    // Position tooltip
    const rect = event.target.getBoundingClientRect();
    tooltip.style.left = (rect.left + rect.width / 2) + 'px';
    tooltip.style.top = (rect.top - 30) + 'px';
    tooltip.style.transform = 'translateX(-50%)';

    // Show tooltip
    tooltip.classList.add('visible');
}

// Hide tooltip
function hideTooltip() {
    const tooltip = document.getElementById('grid-tooltip');
    tooltip.classList.remove('visible');
}

// Show context menu for label selection
function showContextMenu(event, compId, compIndex) {
    const menu = document.getElementById('context-menu');
    const menuItems = document.getElementById('context-menu-items');

    // Clear previous items
    menuItems.innerHTML = '';

    // Add title
    const title = document.createElement('div');
    title.className = 'context-menu-title';
    title.textContent = `Component ${compId}`;
    menuItems.appendChild(title);

    // Add label options
    currentData.candidate_labels.forEach(label => {
        const item = document.createElement('div');
        item.className = 'context-menu-item';
        item.textContent = label;

        // Highlight current label
        if (manualLabels[compIndex] === label) {
            item.classList.add('active');
        }

        item.addEventListener('click', () => {
            updateLabel(compIndex, label);
            menu.classList.add('hidden');
        });

        menuItems.appendChild(item);
    });

    // Position menu
    menu.style.left = event.pageX + 'px';
    menu.style.top = event.pageY + 'px';
    menu.classList.remove('hidden');
}

// Update label
function updateLabel(compIndex, label) {
    manualLabels[compIndex] = label;
    updateLabelsTable();
    updateGridHighlights();
}

// Update all grid cell highlights
function updateGridHighlights() {
    const cells = document.querySelectorAll('.grid-cell');
    cells.forEach(cell => {
        const compIndex = parseInt(cell.dataset.compIndex);
        updateCellHighlight(cell, compIndex);
    });
}

// Update labels table
function updateLabelsTable() {
    const table = document.getElementById('labels-table');
    table.innerHTML = '';

    const tableEl = document.createElement('table');
    tableEl.className = 'labels-table';

    // Header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Component</th>
            <th>Auto Label</th>
            <th>Manual Label</th>
        </tr>
    `;
    tableEl.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    currentData.comp_ids.forEach((compId, i) => {
        const row = document.createElement('tr');
        const isModified = manualLabels[i] !== currentData.auto_labels[i];

        row.innerHTML = `
            <td>${compId}</td>
            <td>${currentData.auto_labels[i]}</td>
            <td class="${isModified ? 'modified' : ''}">${manualLabels[i]}</td>
        `;
        tbody.appendChild(row);
    });
    tableEl.appendChild(tbody);

    table.appendChild(tableEl);
}

// Save labels and move to next
async function saveLabels() {
    try {
        const response = await fetch('/api/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                base: currentBase,
                manual_labels: manualLabels
            })
        });

        const result = await response.json();

        if (result.success) {
            // Update progress after save
            await fetchAndUpdateProgress();

            // Move to next BIDS if available
            if (currentIndex < allBases.length - 1) {
                await navigateNext();
            } else {
                alert('All BIDS completed!');
                // Optionally reload to update status
                await loadStatus();
            }
        } else {
            alert('Error: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error saving labels:', error);
        alert('Error saving labels');
    }
}

// Skip current BIDS
async function skipBIDS() {
    if (!confirm('Skip this BIDS and use auto labels?')) {
        return;
    }

    try {
        const response = await fetch('/api/skip', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                base: currentBase
            })
        });

        const result = await response.json();

        if (result.success) {
            // Update progress after skip
            await fetchAndUpdateProgress();

            // Move to next BIDS if available
            if (currentIndex < allBases.length - 1) {
                await navigateNext();
            } else {
                alert('All BIDS completed!');
                await loadStatus();
            }
        } else {
            alert('Error: ' + (result.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error skipping:', error);
        alert('Error skipping BIDS');
    }
}

// Navigate to previous BIDS
async function navigatePrevious() {
    if (currentIndex > 0) {
        const prevBase = allBases[currentIndex - 1];
        await loadBIDS(prevBase);
        // Update progress after navigation
        await fetchAndUpdateProgress();
    }
}

// Navigate to next BIDS
async function navigateNext() {
    if (currentIndex < allBases.length - 1) {
        const nextBase = allBases[currentIndex + 1];
        await loadBIDS(nextBase);
        // Update progress after navigation
        await fetchAndUpdateProgress();
    }
}

// Fetch latest status and update progress display
async function fetchAndUpdateProgress() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        statusData = status;
        updateProgressDisplay();
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

// Update progress display (text and bar)
function updateProgressDisplay() {
    if (!statusData) return;

    const progressText = document.getElementById('progress-text');
    const progressBarFill = document.getElementById('progress-bar-fill');

    const total = statusData.total;
    const processed = statusData.processed;
    const remaining = statusData.remaining;
    const percentage = total > 0 ? Math.round((processed / total) * 100) : 0;

    // Update text
    progressText.textContent = `Processed: ${processed}/${total} | Remaining: ${remaining}`;

    // Update progress bar
    progressBarFill.style.width = percentage + '%';
    progressBarFill.setAttribute('data-percentage', percentage + '%');
}

// Update navigation button states
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    // Disable previous button if at first BIDS
    prevBtn.disabled = (currentIndex <= 0);

    // Disable next button if at last BIDS
    nextBtn.disabled = (currentIndex >= allBases.length - 1);
}

// Setup event listeners
function setupEventListeners() {
    // Modal close
    document.querySelector('.close').addEventListener('click', () => {
        document.getElementById('modal').classList.add('hidden');
    });

    // Click outside modal to close
    window.addEventListener('click', (e) => {
        const modal = document.getElementById('modal');
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });

    // Click outside context menu to close
    document.addEventListener('click', (e) => {
        const menu = document.getElementById('context-menu');
        if (!menu.contains(e.target)) {
            menu.classList.add('hidden');
        }
    });

    // Buttons
    document.getElementById('save-btn').addEventListener('click', saveLabels);
    document.getElementById('skip-btn').addEventListener('click', skipBIDS);
    document.getElementById('reload-btn').addEventListener('click', () => {
        loadBIDS(currentBase);
    });
    document.getElementById('prev-btn').addEventListener('click', navigatePrevious);
    document.getElementById('next-btn').addEventListener('click', navigateNext);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);
