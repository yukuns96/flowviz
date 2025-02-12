#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <omp.h>  // OpenMP header

using namespace cv;
using namespace std;

const int N_X = 1920;
const int N_Y = 1080;
const int PIXEL_MAX = 255;

// Generate a filename based on frame number.
string fnameGen(int frame){
    ostringstream frameChar;
    frameChar << setw(4) << setfill('0') << frame;
    string frameChar4 = frameChar.str();
    string fname = "frame_" + frameChar4 + ".tif";
    return fname;
}

// Read parameters from the config file.
void config_(string &path_in, string &path_out, string &integration_option,
             int &integration_length, int &number_colored_frames,
             string &configname) {
    ifstream configFile(configname);
    string key, value;
    while (configFile >> key >> value) {
        if (key == "path_in") {
            path_in = value;
        } else if (key == "path_out") {
            path_out = value;
        } else if (key == "integration_length") {
            integration_length = std::stoi(value);
        } else if (key == "number_colored_frames") {
            number_colored_frames = std::stoi(value);
        } else if (key == "integration_option") {
            integration_option = value;
        }
    }
}

int main(){
    clock_t start = clock();

    // Read config parameters.
    string path_in, path_out, integration_option;
    int integration_length, number_colored_frames;
    string configname = "config.txt";
    config_(path_in, path_out, integration_option, integration_length,
            number_colored_frames, configname);
    int N_P = integration_length;
    cout << "Integration length: " << N_P << endl;

    // Define the original plasma colormap (17 x 3).
    const int rows = 17;
    const int cols = 3;
    double plasma_cmap[rows][cols] = {
        {0.0504, 0.0298, 0.5270},
        {0.1292, 0.0559, 0.5668},
        {0.2033, 0.0369, 0.5581},
        {0.2713, 0.0244, 0.5244},
        {0.3442, 0.0725, 0.4696},
        {0.4041, 0.1362, 0.4054},
        {0.4745, 0.1962, 0.3491},
        {0.5457, 0.2573, 0.2959},
        {0.6267, 0.3197, 0.2490},
        {0.7136, 0.3840, 0.2161},
        {0.8035, 0.4553, 0.1965},
        {0.8956, 0.5255, 0.1925},
        {0.9866, 0.5929, 0.2095},
        {0.9932, 0.6948, 0.3218},
        {0.9871, 0.7884, 0.4713},
        {0.9871, 0.8730, 0.6538},
        {0.9914, 0.9506, 0.8581}
    };

    // Generate a colormap (cmap) with N_P rows by interpolating the plasma_cmap.
    double cmap[N_P][3];
    for (int i = 0; i < N_P; i++){
        double cmap_ind = static_cast<double>(i) / (N_P - 1);
        for (int k = 0; k < 3; k++){
            // Linear interpolation from plasma_cmap[rows-1] (max) to plasma_cmap[0] (min).
            cmap[i][k] = (plasma_cmap[rows - 1][k] - plasma_cmap[0][k]) * cmap_ind + plasma_cmap[rows - 1][k];
        }
    }

    // Allocate a 3D vector to hold the pixel values.
    vector<vector<vector<int>>> pixel(
        N_P, vector<vector<int>>(N_Y, vector<int>(N_X, 0))
    );
    string fpath = "../piv_data/subtracted_shifted_adjusted_0/";

    // For each frame (number_colored_frames)...
    for (int nn = 0; nn < number_colored_frames; nn++){
        // Read images for each integration slice in parallel.
        #pragma omp parallel for schedule(dynamic)
        for (int n = 0; n < N_P; n++){
            // Each thread uses its own local readname.
            string local_readname = fpath + fnameGen(n + 1);
            Mat image = imread(local_readname, IMREAD_UNCHANGED);
            if (image.empty()) {
                // If the image cannot be read, skip processing.
                continue;
            }
            for (int iy = 0; iy < N_Y; iy++){
                for (int ix = 0; ix < N_X; ix++){
                    uchar pixelValue = image.at<uchar>(iy, ix);
                    pixel[n][iy][ix] = static_cast<int>(pixelValue);
                }
            }
        }

        // Create matrices to hold index of max value and the maximum pixel value.
        vector<vector<int>> ind_max(N_Y, vector<int>(N_X, 0));
        vector<vector<int>> valK(N_Y, vector<int>(N_X, 0));

        // Process each pixel independently in parallel.
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int iy = 0; iy < N_Y; iy++){
            for (int ix = 0; ix < N_X; ix++){
                int px_t[N_P];
                for (int n = 0; n < N_P; n++){
                    px_t[n] = pixel[n][iy][ix];
                }
                int px_t_orig[N_P];
                copy(px_t, px_t + N_P, px_t_orig);
                sort(px_t, px_t + N_P);
                int max_val = px_t[N_P - 1];
                valK[iy][ix] = max_val;

                int imax = 0;
                while (px_t_orig[imax] != max_val && imax < N_P)
                    imax++;
                ind_max[iy][ix] = imax;
            }
        }

        // Build an RGB image.
        cv::Mat im_rgb(N_Y, N_X, CV_8UC3);
        const int COEFF = 1;
        // Parallelize over rows of the image.
        #pragma omp parallel for schedule(dynamic)
        for (int row = 0; row < im_rgb.rows; ++row){
            for (int col = 0; col < im_rgb.cols; ++col){
                uchar valR = static_cast<uchar>(COEFF * valK[row][col] * cmap[ind_max[row][col]][0]);
                uchar valG = static_cast<uchar>(COEFF * valK[row][col] * cmap[ind_max[row][col]][1]);
                uchar valB = static_cast<uchar>(COEFF * valK[row][col] * cmap[ind_max[row][col]][2]);
                im_rgb.at<cv::Vec3b>(row, col) = cv::Vec3b(valB, valG, valR);
            }
        }
        cv::imwrite(path_out + fnameGen(nn + 1), im_rgb);

        // Optionally, display progress.
        if (nn % 10 == 0){
            cout << "Processed frame: " << nn << endl;
        }
    }
    
    clock_t end = clock();
    double duration = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Time taken: " << duration << " seconds." << endl;
    return 0;
}
