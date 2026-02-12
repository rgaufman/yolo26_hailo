CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -I. $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4) -lhailort

TARGETS = detect_image benchmark_inference run_coco_inference

all: $(TARGETS)

detect_image: detect_image.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

benchmark_inference: benchmark_inference.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

run_coco_inference: run_coco_inference.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS)

.PHONY: all clean
