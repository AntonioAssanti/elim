#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;
int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
//Mat src, src_gray, dst;
char* window_name = "Threshold Demo";
char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

/*void Threshold_Demo( int, void* )
{
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

 /* threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

  imshow( window_name, dst );
}*/

int ind = 0;
Mat src, dst, map_x, map_y;
char* remap_window = "Remap demo";

void update_map( void ){
    ind = ind%4;

    for( int j = 0; j < src.rows; j++ ){
        for( int i = 0; i < src.cols; i++ ){
         switch(ind){
            case 0:
                if( i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75 ){
                    map_x.at<float>(j,i) = 2*( i - src.cols*0.25 ) + 0.5 ;
                    map_y.at<float>(j,i) = 2*( j - src.rows*0.25 ) + 0.5 ;
                    }
                else{
                    map_x.at<float>(j,i) = 0 ;
                    map_y.at<float>(j,i) = 0 ;
                }
                break;
            case 1:
                 map_x.at<float>(j,i) = i ;
                 map_y.at<float>(j,i) = src.rows - j ;
                 break;
           case 2:
                 map_x.at<float>(j,i) = src.cols - i ;
                 map_y.at<float>(j,i) = j ;
                 break;
           case 3:
                 map_x.at<float>(j,i) = src.cols - i ;
                 map_y.at<float>(j,i) = src.rows - j ;
                 break;
        } // end of switch
       }
    }
    ind++;
}

int main()
{   /*
    ///EROSIONE-DILATAZIONE
    Mat image;
    image = imread("erosione.png");
    namedWindow("ORIGINE", CV_WINDOW_AUTOSIZE);
    imshow("ORIGINE", image);
    waitKey(0);
    Mat erd;
    erode(image, erd, Mat(), Point(-1, -1), 2, 1, 1);
    namedWindow("EROSIONE", CV_WINDOW_AUTOSIZE);
    imshow("EROSIONE", erd);
    waitKey(0);
    Mat dil;
    dilate(image, dil, Mat(), Point(-1, -1), 2, 1, 1);
    imwrite("DILATAZIONE.png", dil);
    imshow("DILATAZIONE", dil);
    waitKey(0);
    Mat image2;
    dilate(erd, image2, Mat(), Point(-1, -1), 2, 1, 1);
    imwrite("ORIGINE2.png", image2);
    imshow("ORIGINE2", image2);
    waitKey(0);
    */
    /*
    ///ZOOM IN AND OUT
    cout << endl << "Zoom In-Out demo" <<endl;
    cout << endl << "------------------" <<endl;
    cout << "* [1] -> Zoom in" << endl;
    cout << "* [2] -> Zoom out" << endl;
    cout << "* [0] -> Close program" << endl << endl;
    Mat src, tmp, dst;
    src = imread("chicky_512.png");
    tmp = src;
    dst = tmp;
    namedWindow("ORIGINE", CV_WINDOW_AUTOSIZE);
    imshow("ORIGINE", dst);
    while(true){
        int c;
        cin >> c;
        waitKey(10);
        if(c == 0)
            break;
        if(c == 1){
            pyrUp( tmp, dst, Size( tmp.cols*2, tmp.rows*2 ) );
            cout << "** Zoom In: Image x 2" << endl;
        }
        else if(c == 2){
            pyrDown( tmp, dst, Size( tmp.cols/2, tmp.rows/2 ) );
            cout << "** Zoom Out: Image / 2 \n" << endl;
        }

        imshow("RISULTATO", dst);
        tmp = dst;
    }
    */
/*
    src = imread("chicky_512.png");
    cvtColor(src, src_gray, CV_BGR2GRAY);
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    createTrackbar(trackbar_type, window_name, &threshold_type, max_type, Threshold_Demo);
    createTrackbar(trackbar_value, window_name, &threshold_value, max_value, Threshold_Demo);
    Threshold_Demo( 0, 0 );
    while(true){
        int c;
        c = waitKey( 20 );
        if((char)c == 27)
            break;
   }
*/
    /*
    ///FILTRI BLUR
    Mat image;
    image = imread("erosione.png",1);
    namedWindow("ORIGINE", CV_WINDOW_AUTOSIZE);
    imshow("ORIGINE", image);
    waitKey(0);
    Mat med, blu;
    medianBlur(image, med, 15);
    namedWindow("MEDIANO", CV_WINDOW_AUTOSIZE);
    imshow("MEDIANO", med);
    waitKey(0);
    for ( int i = 1; i < 31; i = i + 2 )
        blur(image, blu, Size(i, i), Point(-1,-1));
    namedWindow("BLUR", CV_WINDOW_AUTOSIZE);
    imshow("BLUR", blu);
    waitKey();
    Mat gaussian;
    for ( int i = 1; i < 31; i = i + 2 )
        GaussianBlur(image, gaussian, Size( i, i ), 0, 0 );
    namedWindow("GAUSSIANO", CV_WINDOW_AUTOSIZE);
    imshow("GAUSSIANO", gaussian);
    waitKey();
    Mat bilateral;
    for ( int i = 1; i < 31; i = i + 2 )
        bilateralFilter (image, bilateral, i, i*2, i/2 );
    namedWindow("BILATERALE", CV_WINDOW_AUTOSIZE);
    imshow("BILATERALE", bilateral);
    waitKey();
    Mat image;// new blank image
    image = imread("test.png");// read the file
    namedWindow("ORIGINE", CV_WINDOW_AUTOSIZE);// create a window for display.
    imshow("ORIGINE", image);// show our image inside it.
    waitKey(0);// wait for a keystroke in the window
    Mat image2;
    Laplacian(image, image2, CV_8U, 3);
    imshow("LAPLACIAN", image2);
    waitKey(0);
    Mat grey;
    cvtColor(image, grey, CV_BGR2GRAY);
    Mat sobelx;
    Sobel(grey, sobelx, CV_32F, 1, 0);
    imshow("SOBEL", sobelx);
    waitKey(0);
    /*double minVal, maxVal;
    minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
    Mat draw;
    sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));

    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", draw);
    Mat edge;
    Canny( grey, edge, 50, 150, 3);
    namedWindow("CANNY",  CV_WINDOW_AUTOSIZE);
    imshow("CANNY",edge);
    waitKey(0);
*/
    /*
    ///FILTER 2D
    Mat src, dst, kernel;
    Point anchor;
    double delta;
    int ddepth;
    int kernel_size;
    char* window_name = "filter2D Demo";

    int c;
    src = imread("test.png");
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );
    anchor = Point( -1, -1 );
    delta = 0;
    ddepth = -1;
    int ind = 0;
    while( true ){
        c = waitKey(500);
        if( (char)c == 27 )
            break;
        kernel_size = 3 + 2*( ind%5 );
        kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

        filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
        imshow( window_name, dst );
        ind++;
    }
    */
    /*
    ///HOUGH
    Mat src;
    src = imread("hough.jpg");
    Mat dst, cdst;
    Canny(src, dst, 50, 200, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    vector<Vec2f> lines;
    HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

    for( size_t i = 0; i < lines.size(); i++ ){
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
    }

    vector<Vec4i> lines2;
    HoughLinesP(dst, lines2, 1, CV_PI/180, 50, 50, 10 );
    for( size_t i = 0; i < lines2.size(); i++ ){
        Vec4i l = lines2[i];
        line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
    }
    imshow("source", src);
    imshow("detected lines", cdst);

    waitKey(0);
    */

    ///REMAP
    src = imread("remap.jpg");
    dst.create( src.size(), src.type() );
    map_x.create( src.size(), CV_32FC1 );
    map_y.create( src.size(), CV_32FC1 );
    namedWindow( remap_window, CV_WINDOW_AUTOSIZE );
    while( true ){
        int c = waitKey( 1000 );
        if( (char)c == 27 )
            break;
    update_map();
    remap( src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );
    imshow( remap_window, dst );
  }

    return 0;
}
