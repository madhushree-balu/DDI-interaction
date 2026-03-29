document.addEventListener('DOMContentLoaded', () => {
    // State
    let selectedBrands = []; // Array of {brand, displayTxt}
    
    // Elements
    const searchInput = document.getElementById('drug-search');
    const searchResults = document.getElementById('search-results');
    const selectedContainer = document.getElementById('selected-drugs-container');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.loader');
    
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsContainer = document.getElementById('results-container');
    
    // Inputs
    const ageInput = document.getElementById('patient-age');
    const genderInput = document.getElementById('patient-gender');
    const conditionsInput = document.getElementById('patient-conditions');
    const labsInput = document.getElementById('patient-labs');

    // Debounce timer
    let searchTimeout = null;

    // OCR Upload Handler
    const ocrUpload = document.getElementById('ocr-upload');
    if (ocrUpload) {
        ocrUpload.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const originalPlaceholder = searchInput.placeholder;
            searchInput.value = "";
            searchInput.placeholder = "Scanning image...";
            searchInput.disabled = true;

            const formData = new FormData();
            formData.append('image', file);

            fetch('/api/ocr', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                
                const nameToSearch = (data.brand_name && data.brand_name !== "Not found" && data.brand_name !== "null") ? data.brand_name : 
                                     ((data.tablet_name && data.tablet_name !== "Not found" && data.tablet_name !== "null") ? data.tablet_name : null);
                
                if (!nameToSearch) {
                    throw new Error("Could not extract a valid name from the image.");
                }

                searchInput.value = nameToSearch;
                searchInput.disabled = false;
                searchInput.placeholder = originalPlaceholder;
                
                // Auto search
                fetch(`/api/brands/search?q=${encodeURIComponent(nameToSearch)}`)
                    .then(res => res.json())
                    .then(searchData => {
                        const results = searchData.results || [];
                        if (results.length > 0) {
                            addBrand({brand: results[0], displayTxt: results[0]});
                            searchInput.value = '';
                        } else {
                            alert(`Extracted "${nameToSearch}" but couldn't find an exact match. Please edit the search query manually.`);
                        }
                    })
                    .catch(err => {
                        console.error("Auto-search error", err);
                    });
            })
            .catch(err => {
                console.error("OCR error", err);
                alert("Error scanning image: " + err.message);
                searchInput.value = '';
                searchInput.disabled = false;
                searchInput.placeholder = originalPlaceholder;
            })
            .finally(() => {
                ocrUpload.value = '';
            });
        });
    }

    // Search input handler
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        clearTimeout(searchTimeout);
        
        if (query.length < 2) {
            searchResults.classList.add('hidden');
            return;
        }
        
        searchTimeout = setTimeout(() => {
            fetch(`/api/brands/search?q=${encodeURIComponent(query)}`)
                .then(res => res.json())
                .then(data => {
                    renderSearchResults(data.results || []);
                })
                .catch(err => console.error("Search error", err));
        }, 300);
    });

    function renderSearchResults(results) {
        searchResults.innerHTML = '';
        if (results.length === 0) {
            searchResults.innerHTML = '<li style="color: rgba(255,255,255,0.5); padding: 1rem; text-align: center;">No matching brands found</li>';
        } else {
            results.forEach(brand => {
                const displayTxt = brand;
                
                const li = document.createElement('li');
                li.innerHTML = `<i class="fa-solid fa-capsules" style="color: var(--primary-color)"></i> <strong>${brand}</strong>`;
                li.addEventListener('click', () => {
                    addBrand({brand, displayTxt});
                    searchInput.value = '';
                    searchResults.classList.add('hidden');
                });
                searchResults.appendChild(li);
            });
        }
        searchResults.classList.remove('hidden');
    }

    // Close dropdown
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.classList.add('hidden');
        }
    });

    function addBrand(brandObj) {
        if (selectedBrands.some(b => b.brand === brandObj.brand)) return;
        selectedBrands.push(brandObj);
        renderSelectedBrands();
        checkAnalyzeButton();
    }

    function removeBrand(brandName) {
        selectedBrands = selectedBrands.filter(b => b.brand !== brandName);
        renderSelectedBrands();
        checkAnalyzeButton();
    }

    function renderSelectedBrands() {
        selectedContainer.innerHTML = '';
        selectedBrands.forEach(brandObj => {
            const chip = document.createElement('div');
            chip.className = 'drug-chip';
            chip.innerHTML = `
                <span>${brandObj.displayTxt}</span>
                <button class="chip-remove" title="Remove"><i class="fa-solid fa-xmark"></i></button>
            `;
            chip.querySelector('.chip-remove').addEventListener('click', () => removeBrand(brandObj.brand));
            selectedContainer.appendChild(chip);
        });
    }

    function checkAnalyzeButton() {
        // Need at least 2 drugs to check interactions
        analyzeBtn.disabled = selectedBrands.length < 2;
    }

    // Analyze handler
    analyzeBtn.addEventListener('click', () => {
        const payload = {
            brand_names: selectedBrands.map(b => b.brand),
            patient_data: null
        };
        
        const age = ageInput.value.trim();
        const gender = genderInput.value;
        const weight = document.getElementById('patient-weight').value.trim();
        
        // Grab checked conditions
        const conditions = [];
        document.querySelectorAll('#patient-conditions input[type=checkbox]:checked').forEach(cb => {
            conditions.push(cb.value);
        });
        const cirrhosis = document.getElementById('patient-cirrhosis').value;
        if (cirrhosis) conditions.push(cirrhosis);
        
        // Grab labs
        const lab_egfr = document.getElementById('lab-egfr').value.trim();
        const lab_alt = document.getElementById('lab-alt').value.trim();
        const lab_platelets = document.getElementById('lab-platelets').value.trim();
        const lab_inr = document.getElementById('lab-inr').value.trim();
        const lab_glucose = document.getElementById('lab-glucose').value.trim();
        
        const parsedLabs = {};
        if (lab_egfr) parsedLabs['eGFR'] = Number(lab_egfr);
        if (lab_alt) parsedLabs['ALT'] = Number(lab_alt);
        if (lab_platelets) parsedLabs['platelet_count'] = Number(lab_platelets);
        if (lab_inr) parsedLabs['INR'] = Number(lab_inr);
        if (lab_glucose) parsedLabs['blood_glucose'] = Number(lab_glucose);
        
        if (age || gender || weight || conditions.length > 0 || Object.keys(parsedLabs).length > 0) {
            payload.patient_data = {
                age: age ? parseInt(age) : null,
                gender: gender || null,
                weight: weight ? Number(weight) : null,
                conditions: conditions,
                lab_values: parsedLabs
            };
        }

        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        analyzeBtn.disabled = true;
        
        resultsPlaceholder.classList.add('hidden');
        resultsContainer.innerHTML = '';
        resultsContainer.classList.add('hidden');

        fetch('/api/analyse', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            analyzeBtn.disabled = false;
            
            renderDashboard(data);
        })
        .catch(err => {
            console.error(err);
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            analyzeBtn.disabled = false;
            
            resultsContainer.innerHTML = `<div class="alert-item">An error occurred while running the analysis.</div>`;
            resultsContainer.classList.remove('hidden');
        });
    });

    function renderDashboard(data) {
        if (data.status === 'NO_INTERACTIONS') {
            resultsContainer.innerHTML = `
                <div class="placeholder-state card glass-card fade-in">
                    <div class="placeholder-content">
                        <div class="icon-ring" style="border-color: rgba(16, 185, 129, 0.4)">
                            <i class="fa-solid fa-check placeholder-icon" style="background: linear-gradient(135deg, #34d399, #10b981); -webkit-background-clip: text;"></i>
                        </div>
                        <h3>No Known Interactions</h3>
                        <p>${data.message}</p>
                    </div>
                </div>
            `;
            resultsContainer.classList.remove('hidden');
            return;
        }

        if (data.error) {
            resultsContainer.innerHTML = `<div class="alert-item">Error: ${data.error}</div>`;
            resultsContainer.classList.remove('hidden');
            return;
        }
        
        const rColor = (data.risk_color === 'RED' || data.risk_color === '🔴') ? 'risk-high' : 
                       (data.risk_color === 'YELLOW' || data.risk_color === '🟡') ? 'risk-moderate' : 
                       (data.risk_color === 'GREEN' || data.risk_color === '⚪') ? 'risk-low' : 'risk-unknown';

        const riskIcon = ["🔴", "🟡", "⚪"].includes(data.risk_color) ? data.risk_color : '';
        const riskLevel = data.risk_level || 'MINIMAL';

        let html = `
            <div class="card glass-card fade-in">
                <div class="results-header">
                    <h1>PolyGuard Insight <span style="font-size: 1rem; color: #94a3b8; font-weight: 500;">v2.0</span></h1>
                    <div class="risk-badge ${rColor}">
                        ${riskIcon} ${riskLevel.toUpperCase()} RISK
                    </div>
                </div>
                
                <div class="summary-stats">
                    <div class="stat-card">
                        <div class="stat-icon"><i class="fa-solid fa-triangle-exclamation"></i></div>
                        <div class="stat-value">${data.num_interactions || 0}</div>
                        <div class="stat-label">Interactions</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon"><i class="fa-solid fa-lungs"></i></div>
                        <div class="stat-value">${data.num_organs_affected || 0}</div>
                        <div class="stat-label">Organs at Risk</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon"><i class="fa-solid fa-link"></i></div>
                        <div class="stat-value">${data.num_cascades || 0}</div>
                        <div class="stat-label">Cascades</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon"><i class="fa-solid fa-clipboard-list"></i></div>
                        <div class="stat-value">${data.total_score || 0}</div>
                        <div class="stat-label">Total Score</div>
                    </div>
                </div>
                
                <div class="alert-item" style="border-left-color: var(--primary-color); background: rgba(59, 130, 246, 0.1);">
                    <div class="alert-title"><i class="fa-solid fa-bullseye"></i> Primary Action Required</div>
                    <div>${data.primary_action || ''}</div>
                </div>
            </div>
        `;
        
        const cascades = data.cascades || [];
        if (cascades.length > 0) {
            html += `<h2 class="section-title fade-in" style="animation-delay: 0.1s;"><i class="fa-solid fa-link"></i> Polypharmacy Cascades</h2>`;
            cascades.forEach(c => {
                html += `
                <div class="list-card fade-in" style="animation-delay: 0.15s; border-left: 4px solid var(--warning);">
                    <div class="list-header">
                        <div class="list-title"><i class="fa-solid fa-exclamation-triangle" style="color: var(--warning)"></i> ${c.organ_system} Alert</div>
                        <div class="badge">${c.alert_level}</div>
                    </div>
                    <div class="xai-info">
                        <span><i class="fa-solid fa-bolt"></i> Cumulative Score: ${c.cumulative_score}</span>
                        <span><i class="fa-solid fa-compress"></i> Included interactions: ${c.num_interactions}</span>
                    </div>
                </div>
                `;
            });
        }
        
        const systems = data.organ_systems || [];
        if (systems.length > 0) {
            html += `<h2 class="section-title fade-in" style="animation-delay: 0.2s;"><i class="fa-solid fa-lungs"></i> Organ System Vulnerability</h2>`;
            systems.forEach((sys, i) => {
                const score = sys.score || 0;
                const confPercent = sys.nlp_confidence ? (sys.nlp_confidence * 100).toFixed(0) + '%' : 'N/A';
                const warningMsg = sys.risk_factors && sys.risk_factors.length ? sys.risk_factors.join(', ') : '';
                
                html += `
                <div class="list-card fade-in" style="animation-delay: ${0.2 + (i*0.05)}s;">
                    <div class="list-header">
                        <div class="list-title">${sys.icon || ''} ${sys.system.replace('_', ' ').toUpperCase()}</div>
                        <div class="badge">Score: ${score}</div>
                    </div>
                    ${warningMsg ? `
                        <div style="color: #fcd34d; font-size: 0.9rem; margin-bottom: 0.5rem;">
                            <i class="fa-solid fa-triangle-exclamation"></i> Risk Factors: ${warningMsg}
                        </div>
                    ` : ''}
                    <div class="xai-info">
                        <span>NLP Confidence: ${confPercent}</span>
                        <span>Base Score: ${sys.base_score || score}</span>
                        ${sys.vulnerability_multiplier ? `<span>Multiplier: ${sys.vulnerability_multiplier.toFixed(2)}x</span>` : ''}
                    </div>
                </div>
                `;
            });
        }

        const details = data.interactions || [];
        if (details.length > 0) {
            html += `<h2 class="section-title fade-in" style="animation-delay: 0.3s;"><i class="fa-solid fa-flask"></i> Interaction Details</h2>`;
            details.forEach((d, i) => {
                const drugTitle = `${d.drug_a} & ${d.drug_b}`;
                html += `
                <div class="list-card fade-in" style="animation-delay: ${0.3 + (i*0.05)}s;">
                    <div class="list-header">
                        <div class="list-title">${d.icon || ''} ${drugTitle}</div>
                        <div class="badge">${d.severity} (+${d.score})</div>
                    </div>
                    ${d.description && d.description !== 'No description available' ? 
                        `<div style="margin-bottom: 1rem; color: #cbd5e1">${d.description}</div>` : ''}
                    ${d.mechanism && d.mechanism !== 'Unknown' ? 
                        `<div class="mechanism-text"><i class="fa-solid fa-gears"></i> Mechanism: ${d.mechanism}</div>` : ''}
                </div>
                `;
            });
        }

        resultsContainer.innerHTML = html;
        resultsContainer.classList.remove('hidden');
    }
});
