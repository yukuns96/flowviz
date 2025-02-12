g++ -std=c++17 -I/usr/local/include/opencv4 -fsanitize=address -g testcv.cpp -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -o colorcoder
export DYLD_LIBRARY_PATH=/usr/local/lib
