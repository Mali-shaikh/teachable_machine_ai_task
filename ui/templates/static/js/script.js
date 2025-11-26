// Global variables
let webcamStream = null;
let isTraining = false;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadClasses();
    setupEventListeners();
    loadTrainingHistory();
});

// Event listeners setup
function setupEventListeners() {
    // Upload zone click
    document.getElementById('uploadZone').addEventListener('click', function() {
        document.getElementById('predictUpload').click();
    });
    
    // File upload change
    document.getElementById('predictUpload').addEventListener('change', function(e) {
        handleImageUpload(e.target.files[0]);
    });
    
    // Drag and drop for upload zone
    const uploadZone = document.getElementById('uploadZone');
    uploadZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadZone.style.background = '#f0f4ff';
    });
    
    uploadZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadZone.style.background = '';
    });
    
    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadZone.style.background = '';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageUpload(files[0]);
        }
    });
}

// Tab navigation
function openTab(tabName) {
    // Hide all tab contents
    const tabContents = document.getElementsByClassName('tab-content');
    for (let tab of tabContents) {
        tab.classList.remove('active');
    }
    
    // Remove active class from all tab buttons
    const tabButtons = document.getElementsByClassName('tab-button');
    for (let button of tabButtons) {
        button.classList.remove('active');
    }
    
    // Show selected tab and activate button
    document.getElementById(tabName).classList.add('active');
    event.currentTarget.classList.add('active');
}

// Class management
async function createClass() {
    const classNameInput = document.getElementById('className');
    const className = classNameInput.value.trim();
    
    if (!className) {
        alert('Please enter a class name');
        return;
    }
    
    showLoading('Creating class...');
    
    try {
        const response = await fetch('/create_class', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ class_name: className })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            classNameInput.value = '';
            loadClasses();
            alert('Class created successfully!');
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error creating class: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function loadClasses() {
    try {
        const response = await fetch('/get_classes');
        const data = await response.json();
        
        if (response.ok) {
            displayClasses(data.classes);
            updateClassSelects(data.classes);
        }
    } catch (error) {
        console.error('Error loading classes:', error);
    }
}

function displayClasses(classes) {
    const classesList = document.getElementById('classesList');
    classesList.innerHTML = '';
    
    if (classes.length === 0) {
        classesList.innerHTML = '<p>No classes created yet</p>';
        return;
    }
    
    classes.forEach(cls => {
        const classItem = document.createElement('div');
        classItem.className = 'class-item';
        classItem.innerHTML = `
            <span class="class-name">${cls.name}</span>
            <span class="class-count">${cls.image_count} images</span>
        `;
        classesList.appendChild(classItem);
    });
}

function updateClassSelects(classes) {
    const uploadSelect = document.getElementById('uploadClassSelect');
    uploadSelect.innerHTML = '<option value="">Select a class</option>';
    
    classes.forEach(cls => {
        const option = document.createElement('option');
        option.value = cls.name;
        option.textContent = `${cls.name} (${cls.image_count} images)`;
        uploadSelect.appendChild(option);
    });
}

// Image upload
async function uploadImages() {
    const classSelect = document.getElementById('uploadClassSelect');
    const class_name = classSelect.value;
    const fileInput = document.getElementById('imageUpload');
    
    if (!class_name) {
        alert('Please select a class');
        return;
    }
    
    if (fileInput.files.length === 0) {
        alert('Please select images to upload');
        return;
    }
    
    showLoading('Uploading images...');
    
    const formData = new FormData();
    formData.append('class_name', class_name);
    for (let file of fileInput.files) {
        formData.append('images', file);
    }
    
    try {
        const response = await fetch('/upload_images', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            fileInput.value = '';
            loadClasses();
            alert(`Successfully uploaded ${data.saved_count} images`);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error uploading images: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Model training
async function trainModels() {
    if (isTraining) return;
    
    const selectedModels = Array.from(document.querySelectorAll('input[name="model"]:checked'))
        .map(checkbox => checkbox.value);
    
    if (selectedModels.length === 0) {
        alert('Please select at least one model to train');
        return;
    }
    
    const classes = await getClasses();
    if (classes.length === 0) {
        alert('Please create classes and upload images first');
        return;
    }
    
    isTraining = true;
    const trainBtn = document.getElementById('trainBtn');
    trainBtn.disabled = true;
    trainBtn.textContent = 'Training...';
    
    showLoading('Training models... This may take a few minutes.');
    
    try {
        const response = await fetch('/train_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                class_names: classes,
                models: selectedModels
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            alert('Training completed successfully!');
            loadTrainingHistory();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error training models: ' + error.message);
    } finally {
        isTraining = false;
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train Models';
        hideLoading();
    }
}

async function getClasses() {
    try {
        const response = await fetch('/get_classes');
        const data = await response.json();
        return data.classes.map(cls => cls.name);
    } catch (error) {
        console.error('Error getting classes:', error);
        return [];
    }
}

// Image prediction
function handleImageUpload(file) {
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('uploadPreview');
        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        
        predictUploadedImage(file);
    };
    reader.readAsDataURL(file);
}

async function predictUploadedImage(file) {
    showLoading('Analyzing image...');
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/predict_upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayPredictions(data.predictions, 'uploadPredictions');
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error predicting image: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Webcam functionality
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        
        const webcam = document.getElementById('webcam');
        webcam.srcObject = stream;
        webcamStream = stream;
        
        document.getElementById('startWebcamBtn').disabled = true;
        document.getElementById('stopWebcamBtn').disabled = false;
        document.getElementById('captureBtn').disabled = false;
        
        // Start continuous prediction
        startContinuousPrediction();
    } catch (error) {
        alert('Error accessing webcam: ' + error.message);
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    const webcam = document.getElementById('webcam');
    webcam.srcObject = null;
    
    document.getElementById('startWebcamBtn').disabled = false;
    document.getElementById('stopWebcamBtn').disabled = true;
    document.getElementById('captureBtn').disabled = true;
    
    stopContinuousPrediction();
}

function captureImage() {
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('webcamCanvas');
    const context = canvas.getContext('2d');
    
    context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob(function(blob) {
        predictUploadedImage(blob);
    }, 'image/jpeg');
}

let predictionInterval;
function startContinuousPrediction() {
    predictionInterval = setInterval(async () => {
        if (!webcamStream) return;
        
        const webcam = document.getElementById('webcam');
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        context.drawImage(webcam, 0, 0);
        
        canvas.toBlob(async function(blob) {
            const formData = new FormData();
            formData.append('image', blob);
            
            try {
                const response = await fetch('/predict_upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (response.ok) {
                    displayPredictions(data.predictions, 'webcamPredictions');
                }
            } catch (error) {
                console.error('Prediction error:', error);
            }
        }, 'image/jpeg');
    }, 2000); // Predict every 2 seconds
}

function stopContinuousPrediction() {
    if (predictionInterval) {
        clearInterval(predictionInterval);
    }
}

// Display predictions
function displayPredictions(predictions, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    for (const [modelName, prediction] of Object.entries(predictions)) {
        if (!prediction) continue;
        
        const predictionElement = document.createElement('div');
        predictionElement.className = 'model-prediction';
        
        let modelIcon = 'fas fa-chart-line';
        if (modelName === 'cnn') modelIcon = 'fas fa-network-wired';
        if (modelName === 'random_forest') modelIcon = 'fas fa-tree';
        
        predictionElement.innerHTML = `
            <h4><i class="${modelIcon}"></i> ${formatModelName(modelName)}</h4>
            <div class="prediction-result">${prediction.class} (${(prediction.confidence * 100).toFixed(1)}%)</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${prediction.confidence * 100}%"></div>
            </div>
            <div class="probability-list">
                ${Object.entries(prediction.all_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, prob]) => `
                        <div class="probability-item">
                            <span>${cls}</span>
                            <span>${(prob * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
            </div>
        `;
        
        container.appendChild(predictionElement);
    }
}

function formatModelName(modelName) {
    const names = {
        'logistic': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'cnn': 'Convolutional Neural Network'
    };
    return names[modelName] || modelName;
}

// Training history and results
async function loadTrainingHistory() {
    try {
        const response = await fetch('/get_training_history');
        const data = await response.json();
        
        if (response.ok) {
            displayTrainingHistory(data.history);
        }
    } catch (error) {
        console.error('Error loading training history:', error);
    }
}

function displayTrainingHistory(history) {
    const container = document.getElementById('trainingResults');
    const metricsContainer = document.getElementById('modelMetrics');
    
    if (history.length === 0) {
        container.innerHTML = '<p>No training history yet</p>';
        metricsContainer.innerHTML = '';
        return;
    }
    
    // Display latest training results
    const latest = history[history.length - 1];
    container.innerHTML = `
        <div class="metric-card">
            <h5>Latest Training - ${latest.model_type}</h5>
            <p><strong>Accuracy:</strong> ${(latest.accuracy * 100).toFixed(2)}%</p>
            <p><strong>Date:</strong> ${new Date(latest.timestamp).toLocaleString()}</p>
        </div>
    `;
    
    // Display all models metrics
    metricsContainer.innerHTML = '<h4>All Trained Models</h4>';
    history.forEach(result => {
        const metricCard = document.createElement('div');
        metricCard.className = 'metric-card';
        metricCard.innerHTML = `
            <h5>${result.model_type}</h5>
            <p><strong>Accuracy:</strong> ${(result.accuracy * 100).toFixed(2)}%</p>
            <p><strong>Classes:</strong> ${result.classes.join(', ')}</p>
            <p><strong>Trained:</strong> ${new Date(result.timestamp).toLocaleString()}</p>
        `;
        metricsContainer.appendChild(metricCard);
    });
}

// Utility functions
function showLoading(message = 'Processing...') {
    document.getElementById('loadingMessage').textContent = message;
    document.getElementById('loadingModal').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingModal').style.display = 'none';
}

// Clear all data
async function clearAllData() {
    if (confirm('Are you sure you want to clear all data? This cannot be undone.')) {
        showLoading('Clearing data...');
        
        try {
            const response = await fetch('/clear_data', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (response.ok) {
                loadClasses();
                loadTrainingHistory();
                alert('All data cleared successfully');
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('Error clearing data: ' + error.message);
        } finally {
            hideLoading();
        }
    }
}