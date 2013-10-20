/* OpenCV */
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
/* System file and IO */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
/* */
#include "mnist.h"

using namespace std;
using namespace cv;

#define GENERATE_IMAGE (0)
#define DEBUG          (1)

// Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

static Mat toGrayscale(InputArray _src) {
    Mat src = _src.getMat();
    // only allow one channel
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

struct params
{
    Mat data;
    int ch;
    int rows;
    PCA pca;
    string winName;
};


static  Mat formatImagesForPCA(const vector<Mat> &data)
{
    Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_8U);
    for(unsigned int i = 0; i < data.size(); i++)
    {
        Mat image_row = data[i].clone().reshape(1,1);
        Mat row_i = dst.row(i);
        image_row.convertTo(row_i,CV_8U);
    }
    return dst;
}


int main(int agrc, char** argv)
{
    string prefix = "DataSet/";
#if GENERATE_IMAGE
    cout << "--- Generate digits jpg images and save to DataSet file\n";
    parse_and_save_idx3("train-images.idx3-ubyte");
#endif

    // Holds some images:
    vector<Mat> db;
    int component_num ;
    if ( !argv[1] ) {

        component_num = 1000 ;    // whole data set
    } else
    {
        component_num = atoi(argv[1]);    // images in data set
    }
#if DEBUG
    //cout << "Dim ( " << mPCA_set.cols << "," << mPCA_set.rows << " )" << endl;
#endif
    try {
            cout << "Start Load Digit Images Set\n";
             for (int i=0; i < component_num; i++ )
            {
                db.push_back( imread(prefix + to_string(i) + ".jpg") );
#if DEBUG
                cout << "--> Load " << prefix + to_string(i) << "\n" ;
#endif
            }
    } catch (cv::Exception& e) {
        cerr << "Error opening file." << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    /* Build a matrix with the observations in row:*/
//    Mat data = asRowMatrix(db, CV_8U);
    Mat data = formatImagesForPCA(db);
    cout << "All images save in the data row vector.\n";

    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, component_num);

    // And copy the PCA results:
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();
    cout << "Solve PCA results ( Eigenvalues and Eigenvectors )\n";

    /* Display the result */
    cout << "The dim of eigenvectors are (" << eigenvectors.rows << "," << eigenvectors.cols << ")--> \n";
    /*
    int key = 0;
    while(key != 'q')
        key = waitKey();
    */
    return 0;
}
