//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
    // prevent default behaviour
    e.preventDefault();
    e.stopPropagation();

    fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
    // handle file selecting
    var files = e.target.files || e.dataTransfer.files;
    fileDragHover(e);
    for (var i = 0, f; (f = files[i]); i++) {
        previewFile(f);
    }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplayDAPI = document.getElementById("image-display-DAPI");
var imageDisplayTnT = document.getElementById("image-display-TnT");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");
var resultsDiv = document.getElementById("resultsDiv");


//========================================================================
// Main button events
//========================================================================


function submitPaths() {
    // action for the submit button
    console.log("submit");

    const filePath = document.getElementById("filePath").value;
    const DAPIPath = document.getElementById("DAPIPath").value;
    const TnTPath = document.getElementById("TnTPath").value;

    if (!(filePath && DAPIPath && TnTPath)) {
        window.alert("Please select paths before submit.");
        return;
    }

    // call the predict function of the backend
    predictImage([filePath, DAPIPath, TnTPath]);
}

function clearPaths() {
    // reset selected files
    fileSelect.value = "";

    // remove image sources and hide them
    imagePreview.src = "";
    imageDisplayDAPI.src = "";
    imageDisplayTnT.src = "";
    predResult.innerHTML = "";

    hide(imagePreview);
    hide(imageDisplayDAPI);
    hide(imageDisplayTnT);
    hide(loader);
    hide(predResult);
    show(uploadCaption);

    imageDisplayDAPI.classList.remove("loading");
    imageDisplayTnT.classList.remove("loading");

}

function submitImage() {
    // action for the submit button
    console.log("submit");

    if (!imageDisplayDAPI.src || !imageDisplayTnT.src) {
        window.alert("Please select an image before submit.");
        return;
    }

    loader.classList.remove("hidden");
    imageDisplayDAPI.classList.add("loading");
    imageDisplayTnT.classList.add("loading");

    // call the predict function of the backend
    predictImage(imageDisplayDAPI.src, imageDisplayTnT.src);
}

function clearImage() {
    // reset selected files
    fileSelect.value = "";

    // remove image sources and hide them
    imagePreview.src = "";
    imageDisplayDAPI.src = "";
    imageDisplayTnT.src = "";
    predResult.innerHTML = "";

    hide(imagePreview);
    hide(imageDisplayDAPI);
    hide(imageDisplayTnT);
    hide(loader);
    hide(predResult);
    show(uploadCaption);

    imageDisplayDAPI.classList.remove("loading");
    imageDisplayTnT.classList.remove("loading");

}

function previewFile(file) {
    // show the preview of the image
    console.log(file.name);
    var fileName = encodeURI(file.name);

    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
        imagePreview.src = URL.createObjectURL(file);

        show(imagePreview);
        hide(uploadCaption);

        // reset
        predResult.innerHTML = "";
        imageDisplayDAPI.classList.remove("loading");
        imageDisplayTnT.classList.remove("loading");

        displayImage(reader.result, "image-display");
    };
}

//========================================================================
// Helper functions
//========================================================================

function predictImage(image) {
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(image)
    })
        .then(resp => {
            if (resp.ok)
                resp.json().then(data => {
                    displayResult(data);
                    console.log("results should have displayed")
                });
        })
        .catch(err => {
            console.log("An error occured", err.message);
            window.alert("Oops! Something went wrong.");
        });
}

function displayImage(image, id) {
    // display image on given id <img> element
    let display = document.getElementById(id);
    display.src = image;
    show(display);
}

function displayResult(data) {
    // display the result
    // imageDisplay.classList.remove("loading");
    // hide(loader);
    const cmcount = document.getElementById("cmcount");
    const nucleicount = document.getElementById("nucleicount");
    resultsDiv = document.getElementById("resultsDiv");
    const uploadDiv = document.getElementById("uploadDiv");

    cmcount.innerHTML += data["cmcount"].toString();
    nucleicount.innerHTML += data["nucleicount"].toString();

    show(resultsDiv);
    hide(uploadDiv);
}

function hide(el) {
    // hide an element
    el.classList.add("hidden");
}

function show(el) {
    // show an element
    el.classList.remove("hidden");
}