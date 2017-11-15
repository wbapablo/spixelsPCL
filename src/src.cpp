#include <iostream>
#include <fstream>


#include <opencv2/core.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;


const char *windowDisparity = "Disparity";

void readme();

int main(){
    Mat img1, img2, conc;
    img1 = imread("../Lu.png"); //CV_LOAD_IMAGE_GRAYSCALE);
    img2 = imread("../Ru.png"); //CV_LOAD_IMAGE_GRAYSCALE);

    hconcat(img1,img2,conc);

    imshow("Raw Images",conc);

    waitKey(0);

    double cm1[3][3] = {{351.4026, 0.000000e+00, 339.6319}, {0.000000e+00, 350.6625, 193.8976}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double cm2[3][3] = {{349.9746, 0.000000e+00, 334.0690 }, {0.000000e+00, 349.1968, 196.7178}, {0.000000e+00, 0.000000e+00, 1.000000e+00}};
    double d1[1][5] = {{ -0.1669, 0.0083, -0.0000074943, -0.000071809, 0.0127}};
    double d2[1][5] = {{-0.1661, 0.0079, -0.00034609, -0.0001746, 0.0126}};

    Mat CM1 (3,3, CV_64FC1, cm1);
    Mat CM2 (3,3, CV_64FC1, cm2);
    Mat D1(1,5, CV_64FC1, d1);
    Mat D2(1,5, CV_64FC1, d2);

 
    double r[3][3] = {{1.0, 0.0008, -0.0093},{-0.0008, 1, 0.0034},{0.0093, -0.0034, 1}};
    double t[3][1] = {{ -119.7992}, {-0.1643}, {-0.2634}};
    Mat R (3,3, CV_64FC1, r);
    Mat T (3,1, CV_64FC1, t);


    Mat R1, R2, T1, T2, Q, P1, P2;
    
    stereoRectify(CM1, D1, CM2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);

    Mat map11, map12, map21, map22;
    Size img_size = img1.size();
    initUndistortRectifyMap(CM1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(CM2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    Mat img1r, img2r, imgLr, imgRr;
    remap(img1, img1r, map11, map12, 1);
    remap(img2, img2r, map21, map22, 1);


    hconcat(img1r,img2r,conc);

    imshow("Undistorted Images",conc);

    waitKey(0);

    imgLr = img1r(Rect(4,34,672-24,376-62 ));
    imgRr = img2r(Rect(16 ,28,672-20,376-60 ));




    cv::resize(imgLr, imgLr, img_size);
    cv::resize(imgRr, imgRr, img_size);


    hconcat(imgLr,imgRr,conc);

    imshow("Undistorted Images ROI",conc);

    waitKey(0);

    int sadSize = 1;
    Ptr<StereoSGBM> sbm = StereoSGBM::create(0,16,1);

    sbm->setBlockSize (sadSize);
    sbm->setNumDisparities(16*1);//128
    sbm->setPreFilterCap(100 );//63
    sbm->setMinDisparity(10 ); //-39;0 
    sbm->setUniquenessRatio(0.1);
    sbm->setSpeckleWindowSize(100);
    sbm->setSpeckleRange(128);
    sbm->setDisp12MaxDiff(300);
    sbm->setMode( 1 );

    sbm->setP1 (sadSize*sadSize*16);
    sbm->setP2 (sadSize*sadSize*64);

    Mat disp, disp8;
    sbm->compute(imgLr, imgRr, disp);

    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    cout<< "Hddk " << disp8.size() << endl;

    Mat roi = disp8(Rect(26,0,672-26,376));
    cv::resize(roi,roi,img_size);
    imshow("Hey" , roi);



    
    waitKey(0);

}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SBMSample <imgLeft> <imgRight>" << std::endl; }