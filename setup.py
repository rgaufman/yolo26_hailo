from setuptools import setup, find_packages

setup(
    name="yolo26",
    version="0.1.0",
    description="YOLO26 Object Detection on Raspberry Pi 5 with Hailo-8L",
    author="User",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
    ],
    extras_require={
        "dev": ["onnxruntime", "tensorflow"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8',
    scripts=[
        'python/detect_image.py',
        'python/benchmark_inference.py'
    ]
)
