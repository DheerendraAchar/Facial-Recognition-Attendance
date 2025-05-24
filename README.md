# Face Recognition Attendance System

A real-time face recognition attendance system built with Python, OpenCV, and face_recognition library. This system can automatically detect and recognize faces to mark attendance.

## Features

- **Real-time Face Detection**: Uses webcam to detect faces in real-time
- **Face Recognition**: Identifies registered users and marks attendance
- **Attendance Logging**: Automatically logs attendance with timestamps
- **User Management**: Add new users to the system
- **Database Integration**: Stores attendance records in a database
- **Simple GUI**: Easy-to-use interface for managing the system

## Tech Stack

- **Python 3.8+**
- **OpenCV** - Computer vision and image processing
- **face_recognition** - Face detection and recognition
- **NumPy** - Numerical computations
- **SQLite/MySQL** - Database for storing attendance records
- **Tkinter** - GUI framework (if applicable)

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- pip package manager

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/DheerendraAchar/Facial-Recognition-Attendance.git
cd Facial-Recognition-Attendance
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **Install face recognition models:**
```bash
pip install git+https://github.com/ageitgey/face_recognition_models
```

5. **Download the face landmark predictor (if required):**
   - Download `shape_predictor_68_face_landmarks.dat` from [dlib's website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract and place it in the project root directory

## Usage

### 1. Generate Face Embeddings

First, add photos of people you want to recognize in the `known_faces/` directory:

```
known_faces/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
└── person2/
    ├── photo1.jpg
    └── photo2.jpg
```

Then generate embeddings:
```bash
python generate_embeddings.py
```

### 2. Run the Attendance System

```bash
python main.py
```

### 3. Alternative Scripts

- **For real-time recognition:**
```bash
python face_recognition_live.py
```

- **For batch processing:**
```bash
python batch_recognition.py
```

## Project Structure

```
Face-Recognition-Attendance/
├── main.py                    # Main application file
├── generate_embeddings.py     # Generate face embeddings
├── face_recognition_live.py   # Real-time face recognition
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── known_faces/              # Directory for known face images
│   ├── person1/
│   └── person2/
├── database/                 # Database files
│   └── attendance.db
├── models/                   # Trained models and embeddings
│   └── face_embeddings.pkl
└── attendance_logs/          # Attendance log files
    └── attendance_YYYY-MM-DD.csv
```

## Configuration

### Camera Settings
- Default camera index: 0 (change in `main.py` if needed)
- Resolution: 640x480 (adjustable)

### Recognition Settings
- Face recognition tolerance: 0.6 (lower = more strict)
- Minimum face size: 50x50 pixels

### Database Configuration
Edit database settings in `config.py`:
```python
DATABASE_URL = "sqlite:///attendance.db"
# or for MySQL:
# DATABASE_URL = "mysql://username:password@localhost/attendance_db"
```

## API Reference

### Core Functions

#### `generate_embeddings()`
Generates face embeddings from images in the `known_faces/` directory.

#### `recognize_face(face_encoding)`
Compares a face encoding with known faces and returns the person's name.

#### `mark_attendance(name)`
Records attendance for the recognized person with timestamp.

## Troubleshooting

### Common Issues

1. **Camera not working:**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure no other applications are using the camera

2. **Face recognition not accurate:**
   - Add more photos per person (3-5 recommended)
   - Ensure good lighting in photos
   - Use high-quality, clear images
   - Adjust recognition tolerance

3. **Installation errors:**
   - Make sure you have Python 3.8+
   - Install cmake: `pip install cmake`
   - Install dlib: `pip install dlib`
   - On Windows, install Visual Studio Build Tools

4. **Performance issues:**
   - Reduce video resolution
   - Process every 2nd or 3rd frame
   - Use smaller face images for training

## Performance Optimization

- **Frame skipping**: Process every nth frame for better performance
- **Face detection optimization**: Use smaller video resolution
- **Database optimization**: Use indexed queries for faster lookups

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request


## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [dlib](http://dlib.net/) for machine learning algorithms

## Contact

**Developer:** Dheerendra Achar  
**GitHub:** [@DheerendraAchar](https://github.com/DheerendraAchar)  
**Project Link:** [https://github.com/DheerendraAchar/Facial-Recognition-Attendance](https://github.com/DheerendraAchar/Facial-Recognition-Attendance)

---

## Future Enhancements

- [ ] Web-based interface
- [ ] Mobile app integration
- [ ] Multiple camera support
- [ ] Advanced reporting features
- [ ] Cloud database integration
- [ ] Email notifications
- [ ] Anti-spoofing measures
