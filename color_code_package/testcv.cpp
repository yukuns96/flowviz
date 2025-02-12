#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <string>
#include <ctime>

using namespace cv;
using namespace std;

const int N_X = 1920;
const int N_Y = 1080;
const int PIXEL_MAX = 255;

string fnameGen(int frame){
    ostringstream frameChar;
    frameChar.str("");
    frameChar.clear();
    frameChar << setw(4) << setfill('0') << frame;
    string frameChar4 = frameChar.str();
    string fname = "frame_"+frameChar4+".tif";
    return fname;
}

// read parameters from the config file
void config_(string &path_in, string &path_out, string &integration_option,
             int &integration_length, int &number_colored_frames,
             string &configname){
    ifstream configFile(configname);
    string key, value;
    while (configFile >> key >> value){
        if (key == "path_in"){
            path_in = value;
        } else if (key == "path_out"){
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

    string path_in, path_out, integration_option;
    int integration_length, number_colored_frames;
    string configname = "config.txt";
    config_(path_in, path_out, integration_option, integration_length,
            number_colored_frames, configname);
    int N_P = integration_length;
    cout << N_P << endl;
    cout << number_colored_frames << endl;

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
    double cmap[N_P][3];
    // double cmap_ind[N];
    for (int i=0; i<N_P; i++){
        double cmap_ind = (1.0-0.0)/(static_cast<double>(N_P)-1.0)*i; 
        for (int k=0; k<3; k++){
            cmap[i][k] = (plasma_cmap[rows-1][k]-plasma_cmap[0][k])*cmap_ind + plasma_cmap[rows-1][k];
            // cout << cmap[i][k] << ", ";
        }
        // cout << endl;
        // cmap_ind[i] = (1.0-0.0)/(static_cast<double>(N)-1.0);
    }


    vector<vector<vector<int>>> pixel(
        N_P, vector<vector<int>>(N_Y, vector<int>(N_X, 0))
    );
    string fpath = "../piv_data/subtracted_shifted_adjusted_0/";
    string readname;
    for (int nn=0; nn<number_colored_frames; nn++){
        int count=0;
        for (int n=nn; n<nn+N_P; n++){
            readname = fpath+fnameGen(n+1);
            Mat image = imread(readname, IMREAD_UNCHANGED);
            for (int iy=0; iy<N_Y; iy++){
                for (int ix=0; ix<N_X; ix++){
                    uchar pixelValue = image.at<uchar>(iy, ix);
                    pixel[count][iy][ix] = static_cast<int>(pixelValue);
                }
            }
            count++;
        }

        // matrix that stores index for maximum value at each pixel
        vector<vector<int>> ind_max(N_Y, vector<int>(N_X, 0));

        // matrix that stores greyscale value for maximum value at each pixel
        vector<vector<int>> valK(N_Y, vector<int>(N_X, 0));

        for (int iy=0; iy<N_Y; iy++){
            for (int ix=0; ix<N_X; ix++){
                int px_t[N_P];
                for (int n=0; n<N_P; n++){
                    px_t[n] = pixel[n][iy][ix];
                }
                int px_t_orig[N_P];
                copy(px_t, px_t+N_P, px_t_orig);
                sort(px_t, px_t+N_P);
                int max_val = px_t[N_P-1];
                valK[iy][ix] = max_val;
                // int min_val = px_t[0];
                
                int imax=0;
                while (px_t_orig[imax] != max_val){
                    imax++;
                }
                ind_max[iy][ix] = imax;
            }
        }

        // cv::Mat rgbImage(N_Y, N_X, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat im_rgb(N_Y, N_X, CV_8UC3);
        int iy=0;
        int COEFF = 1;
        for (int row=0; row<im_rgb.rows; ++row){
            int ix=0;
            for (int col=0; col<im_rgb.cols; ++col){
                uchar valR = COEFF*valK[iy][ix]*static_cast<uchar>((cmap[ind_max[iy][ix]][0]));
                uchar valG = COEFF*valK[iy][ix]*static_cast<uchar>((cmap[ind_max[iy][ix]][1]));
                uchar valB = COEFF*valK[iy][ix]*static_cast<uchar>((cmap[ind_max[iy][ix]][2]));
                im_rgb.at<cv::Vec3b>(row, col) = cv::Vec3b(valB, valG, valR);
                ix++;
            }
            iy++;
        }
        cv::imwrite(path_out+fnameGen(nn+1), im_rgb);
        // display progress
        // if (nn%10 == 0){
        //     cout << nn << endl;
        // }
    }
    
    clock_t end = clock();
    double duration = (end - start) / CLOCKS_PER_SEC; // seconds
    std::cout << "Time taken: " << duration << " seconds." << std::endl;
    return 0;
}