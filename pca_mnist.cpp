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

static void onTrackbar(int pos, void* ptr)
{
    cout << "Retained Variance = " << pos << "%   ";
    cout << "re-calculating PCA..." << std::flush;

    double var = pos / 100.0;

    struct params *p = (struct params *)ptr;

    p->pca = PCA(p->data, cv::Mat(), CV_PCA_DATA_AS_ROW, var);

    Mat point = p->pca.project(p->data.row(0));
    Mat reconstruction = p->pca.backProject(point);
    reconstruction = reconstruction.reshape(p->ch, p->rows);
    reconstruction = toGrayscale(reconstruction);

    imshow(p->winName, reconstruction);
    cout << "done!   # of principal components: " << p->pca.eigenvectors.rows << endl;
}

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

int main()
{
    string prefix = "DataSet/";
#if GENERATE_IMAGE
    cout << "--- Generate digits jpg images and save to DataSet file\n";
    parse_and_save_idx3("train-images.idx3-ubyte");
#endif

    // Holds some images:
    vector<Mat> db;

    const int component_num = 1000;    // images in data set

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
        cerr << "Error opening file \"" << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    /* Build a matrix with the observations in row:*/
//    Mat data = asRowMatrix(db, CV_8U);
    Mat data = formatImagesForPCA(db);
    cout << "All images save in the data vector.\n";

    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, component_num );

    // And copy the PCA results:
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();
    cout << "Solve PCA results ( Eigenvalues and Eigenvectors )\n";
    // The mean face:
    imshow("avg", norm_0_255(mean.reshape(1, db[0].rows)));

    // The first three eigenfaces:
    imshow("pc1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
    imshow("pc2", norm_0_255(pca.eigenvectors.row(1)).reshape(1, db[0].rows));
    imshow("pc3", norm_0_255(pca.eigenvectors.row(2)).reshape(1, db[0].rows));

    // Demonstration of the effect of retainedVariance on the first image
    Mat point = pca.project(data.row(0)); // project into the eigenspace, thus the image becomes a "point"
    Mat reconstruction = pca.backProject(point); // re-create the image from the "point"
    reconstruction = reconstruction.reshape(db[0].channels(), db[0].rows); // reshape from a row vector into image shape
    reconstruction = toGrayscale(reconstruction); // re-scale for displaying

    // display until user presses q
    string winName = "Reconstruction | press 'q' to quit";
    imshow(winName, reconstruction);

    // params struct to pass to the trackbar handler
    params p;
    p.data = data;
    p.ch = db[0].channels();
    p.rows = db[0].rows;
    p.pca = pca;
    p.winName = winName;

    // create the tracbar
    int pos = 95;
    createTrackbar("Retained Variance (%)", winName, &pos, 100, onTrackbar, (void*)&p);


    int key = 0;
    while(key != 'q')
        key = waitKey();
    return 0;
}
