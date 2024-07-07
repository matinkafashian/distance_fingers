Real-time Hand Distance Measurement and Landmark Visualization
This Python script leverages OpenCV and MediaPipe to perform real-time hand detection and distance measurement between fingers. It captures video from a webcam, detects hand landmarks, and calculates the distance between the thumb tip and index finger tip, displaying the results dynamically on the video feed.

Features:
Real-time Hand Detection: Utilizes MediaPipe's Hands solution to detect and track multiple hands in the video stream.
Landmark Visualization: Draws hand landmarks and connections on the video frame for better visualization.
Distance Calculation: Computes the Euclidean distance between the thumb tip and index finger tip.
Distance Threshold: Includes a customizable threshold to consider small distances as zero, useful for gesture recognition.
Visual Feedback: Displays the calculated distance and a connecting line between the thumb and index finger on the video feed.
Dependencies:
OpenCV
MediaPipe
NumPy
