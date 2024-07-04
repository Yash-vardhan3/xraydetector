document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById('fileInput');
    formData.append('file', fileInput.files[0]);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `Prediction: ${data.prediction}`;
        resultDiv.style.color = data.prediction === 'Pneumonia' ? 'red' : 'green';
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
