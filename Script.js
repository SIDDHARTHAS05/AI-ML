async function correctText() {
    const inputText = document.getElementById('inputText').value;

    const response = await fetch('/correct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
    });

    const data = await response.json();
    const outputDiv = document.getElementById('outputText');
    outputDiv.innerHTML = `
        <h3>Corrected Text:</h3>
        <p>${data.corrected_text}</p>
        <h3>Errors:</h3>
        <ul>${data.errors.map(error => `<li>${error.message}</li>`).join('')}</ul>
    `;
}

document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
    });

    const data = await response.json();
    const uploadResult = document.getElementById('uploadResult');
    uploadResult.innerHTML = data.corrected_file
        ? `<p>Corrected file saved: ${data.corrected_file}</p>`
        : `<p>Error: ${data.error}</p>`;
});
