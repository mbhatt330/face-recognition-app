const loadModels = async () => {
  const MODEL_URL = '/face-recognition-app/models'; // Path to your models folder
  try {
    console.log("Loading models...");
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL); // Load the SSD MobileNetV1 model
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL); // Load the Tiny Face Detector model
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL); // Load the face landmark model
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL); // Load the face recognition model
    console.log("Models loaded successfully.");
    startVideo(); // Start the webcam after models are loaded
  } catch (error) {
    console.error("Error loading models:", error);
  }
};

const startVideo = () => {
  const video = document.getElementById('video');
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      video.srcObject = stream;
      console.log("Webcam initialized");
      recognizeFaces(); // Start face recognition after video stream is ready
    })
    .catch(err => {
      console.error('Error accessing webcam:', err);
    });
};

const recognizeFaces = async () => {
  const video = document.getElementById('video');
  console.log("Loading labeled images...");
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  video.addEventListener('play', () => {
    console.log("Video started");
    const canvas = faceapi.createCanvasFromMedia(video);
    document.body.append(canvas);
    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
      const detections = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      faceapi.draw.drawDetections(canvas, resizedDetections);

      const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));
      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box;
        const text = result.toString();
        const drawBox = new faceapi.draw.DrawBox(box, { label: text });
        drawBox.draw(canvas);
        document.getElementById('match-result').innerText = text;
      });
    }, 100);
  });
};

const loadLabeledImages = async () => {
  const labels = ['Person1']; // Add your own labels
  return Promise.all(
    labels.map(async label => {
      const img = await faceapi.fetchImage(`/face-recognition-app/uploads/${label}.jpg`);
      const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
      if (!detections) {
        throw new Error(`No faces detected for ${label}`);
      }
      return new faceapi.LabeledFaceDescriptors(label, [detections.descriptor]);
    })
  );
};

// Initialize the process
loadModels();  // Start by loading models
