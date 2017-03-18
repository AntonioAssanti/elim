----BLUR
Mat image, blu;
image = imread("erosione.png");
Mat kernel = Mat::once(3,3, CV_8U);
int borderType = BORDER_CONSTANT;
Point anchor = Point(kernel.cols/2, kernel.rows/2);
namedWindow("MEDIANO", CV_WINDOW_AUTOSIZE);
imshow("MEDIANO", med);
waitKey(0);
for ( int i = 1; i < 31; i = i + 2 )
    //blur(image, blu, Size(i, i), Point(-1,-1));
    boxFilter(image, blu, image.type(), anchor, true, borderType);
namedWindow("BLUR", CV_WINDOW_AUTOSIZE);
imshow("BLUR", blu);
waitKey();



----EROSIONE
Mat src, dst, _src;    //L'immagine di input va caricata in _src
Mat kernel = Mat::once(3,3, CV_8U);
Scalar borderValue;
int borderType = BORDER_CONSTANT;
Point anchor = Point(kernel.cols/2, kernel.rows/2);
borderValue = getMaxVal(src.depth());
copyMakeBorder(_src, src, anchor.y, kernel.rows - anchor.y - 1, anchor.x,
    kernel.cols - anchor.x -1, borderType, borderValue);
dst.create(_src.size(), src.type());
vector <int> ofs;
int step = (int)(src.step/src.elemSize1());
int cn = src.channels();
for(int i = 0; i < kernel.rows; i++)
    for(int j = 0, j < kernel.cols; j++)
        if( kernel.at<uchar>(i,j) != 0)
            ofs.push_back(i*step + j*cn);
if( ofs.empty() )
    ofs.push_back(anchor.y*step + anchor.x*cn);
if( src.depth() )
    erode_<uchar>(src, dst, ofs);



----DILATAZIONE
Mat kernel = Mat::once(3, 3, CV_8U);    //L'immagine di input va caricata in _src
Mat src, dst, _src;
Scalar borderValue;
int borderType = BORDER_CONSTANT;
Point anchor = Point(kernel.cols/2, kernel.rows/2);
borderValue = getMinVal(src.depth());
copyMakeBorder(_src, src, anchor.y, kernel.rows - anchor.y -1, anchor.x,
    kernel.cols - anchor.x - 1, borderType, borderValue);
dst.create(_src.size(), src.type());
vector <int> ofs;
int step = (int)(src.step/src.elemSize1());
int cn = src.channels;
for(int i = 0; i < kernel.rows; i++)
    for(int j = 0; j < kernel.cols; j++)
        if(kernel.at<uchar>(i, j) != 0)
              ofs.push_back(i*step + j*cn);
if( ofs.empty() )
    ofs.push_back(anchor.y*step + anchor.x*cn);
if( src.depth() )
    erode_<uchar>(src, dst, ofs);



------APERTURA E CHIUSURA
morphologyEx( src, dst, operation, element );
//operation = 4 apertura
//operation = 5 chiusura
//element è il kernel usato, per trovarlo possiamo usare la funzione
//getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
//morph_elem = 0, morph_size = non lo so forse 0


------THRESHOLDING
/// Convert the image to Gray
cvtColor( src, src_gray, CV_BGR2GRAY );
e poi usare la funzione compare()



-----CONVOLUZIONE
Point anchor = Point( -1, -1 );
double delta = 0;
int ddepth = -1;
/// Apply filter
filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT);



-------SOBEL
  Mat src, src_gray;
  Mat grad;
  char* window_name = "Sobel Demo - Simple Edge Detector";
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  int c;
  /// Load an image
  src = imread( argv[1] );
  if( !src.data )
  { return -1; }
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
  /// Convert it to gray
  cvtColor( src, src_gray, CV_BGR2GRAY );
  /// Create window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );
  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  /// Gradient X
  Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );
  /// Gradient Y
  Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y ); //We convert our partial results back to CV_8U:
  imshow( window_name, grad );
  waitKey(0);
  return 0;
  }




------BILATERAL
Mat src, dst;
int d;
double sigma_color, sigma_space;
int borderType;
int cn = src.channels();
int i, j, k, maxk, radius;
double minValSrc = -1, maxValSrc = 1;
const int kExpNumBinsPerChannel = 1 << 12;
int kExpNumBins = 0;
float lastExpVal = 1.f;
float len, scale_index;
Size size = src.size();

dst.create(size, src.type());

CV_Assert( (src.type() == CV_32FC1 || src.type() == CV_32FC3) &&
    src.type() == dst.type() && src.size() == dst.size() &&
    src.data != dst.data );

if( sigma_color <= 0 )
    sigma_color = 1;
if( sigma_space <= 0 )
    sigma_space = 1;

double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

if( d <= 0 )
    radius = cvRound(sigma_space*1.5);
else
    radius = d/2;
radius = MAX(radius, 1);
d = radius*2 + 1;
// compute the min/max range for the input image (even if multichannel)

minMaxLoc( src.reshape(1), &minValSrc, &maxValSrc );
if(std::abs(minValSrc - maxValSrc) < FLT_EPSILON)
{
    src.copyTo(dst);
    return;
}

// temporary copy of the image with borders for easy processing
Mat temp;
copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );
patchNaNs(temp);

// allocate lookup tables
vector<float> _space_weight(d*d);
vector<int> _space_ofs(d*d);
float* space_weight = &_space_weight[0];
int* space_ofs = &_space_ofs[0];

// assign a length which is slightly more than needed
len = (float)(maxValSrc - minValSrc) * cn;
kExpNumBins = kExpNumBinsPerChannel * cn;
vector<float> _expLUT(kExpNumBins+2);
float* expLUT = &_expLUT[0];

scale_index = kExpNumBins/len;

// initialize the exp LUT
for( i = 0; i < kExpNumBins+2; i++ )
{
    if( lastExpVal > 0.f )
    {
        double val =  i / scale_index;
        expLUT[i] = (float)std::exp(val * val * gauss_color_coeff);
        lastExpVal = expLUT[i];
    }
    else
        expLUT[i] = 0.f;
}

// initialize space-related bilateral filter coefficients
for( i = -radius, maxk = 0; i <= radius; i++ )
    for( j = -radius; j <= radius; j++ )
    {
        double r = std::sqrt((double)i*i + (double)j*j);
        if( r > radius )
            continue;
        space_weight[maxk] = (float)std::exp(r*r*gauss_space_coeff);
        space_ofs[maxk++] = (int)(i*(temp.step/sizeof(float)) + j*cn);
    }

for( i = 0; i < size.height; i++ )
{
    const float* sptr = temp.ptr<float>(i+radius) + radius*cn;
    float* dptr = dst.ptr<float>(i);

    if( cn == 1 )
    {
        for( j = 0; j < size.width; j++ )
        {
            float sum = 0, wsum = 0;
            float val0 = sptr[j];
            for( k = 0; k < maxk; k++ )
            {
                float val = sptr[j + space_ofs[k]];
                float alpha = (float)(std::abs(val - val0)*scale_index);
                int idx = cvFloor(alpha);
                alpha -= idx;
                float w = space_weight[k]*(expLUT[idx] + alpha*(expLUT[idx+1] - expLUT[idx]));
                sum += val*w;
                wsum += w;
            }
            dptr[j] = (float)(sum/wsum);
        }
    }
    else
    {
        assert( cn == 3 );
        for( j = 0; j < size.width*3; j += 3 )
        {
            float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
            float b0 = sptr[j], g0 = sptr[j+1], r0 = sptr[j+2];
            for( k = 0; k < maxk; k++ )
            {
                const float* sptr_k = sptr + j + space_ofs[k];
                float b = sptr_k[0], g = sptr_k[1], r = sptr_k[2];
                float alpha = (float)((std::abs(b - b0) +
                    std::abs(g - g0) + std::abs(r - r0))*scale_index);
                int idx = cvFloor(alpha);
                alpha -= idx;
                float w = space_weight[k]*(expLUT[idx] + alpha*(expLUT[idx+1] - expLUT[idx]));
                sum_b += b*w; sum_g += g*w; sum_r += r*w;
                wsum += w;
            }
            wsum = 1.f/wsum;
            b0 = sum_b*wsum;
            g0 = sum_g*wsum;
            r0 = sum_r*wsum;
            dptr[j] = b0; dptr[j+1] = g0; dptr[j+2] = r0;
        }
    }
}
}




---HOUGH CERCHI (forse)
cvHoughCircles( CvArr* src_image, void* circle_storage,
                int method, double dp, double min_dist,
                double param1, double param2,
                int min_radius, int max_radius )
{
    CvSeq* result = 0;

    CvMat stub, *img = (CvMat*)src_image;
    CvMat* mat = 0;
    CvSeq* circles = 0;
    CvSeq circles_header;
    CvSeqBlock circles_block;
    int circles_max = INT_MAX;
    int canny_threshold = cvRound(param1);
    int acc_threshold = cvRound(param2);

    img = cvGetMat( img, &stub );

    if( !CV_IS_MASK_ARR(img))
        CV_Error( CV_StsBadArg, "The source image must be 8-bit, single-channel" );

    if( !circle_storage )
        CV_Error( CV_StsNullPtr, "NULL destination" );

    if( dp <= 0 || min_dist <= 0 || canny_threshold <= 0 || acc_threshold <= 0 )
        CV_Error( CV_StsOutOfRange, "dp, min_dist, canny_threshold and acc_threshold must be all positive numbers" );

    min_radius = MAX( min_radius, 0 );
    if( max_radius <= 0 )
        max_radius = MAX( img->rows, img->cols );
    else if( max_radius <= min_radius )
        max_radius = min_radius + 2;

    if( CV_IS_STORAGE( circle_storage ))
    {
        circles = cvCreateSeq( CV_32FC3, sizeof(CvSeq),
            sizeof(float)*3, (CvMemStorage*)circle_storage );
    }
    else if( CV_IS_MAT( circle_storage ))
    {
        mat = (CvMat*)circle_storage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) ||
            CV_MAT_TYPE(mat->type) != CV_32FC3 )
            CV_Error( CV_StsBadArg,
            "The destination matrix should be continuous and have a single row or a single column" );

        circles = cvMakeSeqHeaderForArray( CV_32FC3, sizeof(CvSeq), sizeof(float)*3,
                mat->data.ptr, mat->rows + mat->cols - 1, &circles_header, &circles_block );
        circles_max = circles->total;
        cvClearSeq( circles );
    }
    else
        CV_Error( CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*" );

    switch( method )
    {
    case CV_HOUGH_GRADIENT:
        icvHoughCirclesGradient( img, (float)dp, (float)min_dist,
                                min_radius, max_radius, canny_threshold,
                                acc_threshold, circles, circles_max );
          break;
    default:
        CV_Error( CV_StsBadArg, "Unrecognized method id" );
    }

    if( mat )
    {
        if( mat->cols > mat->rows )
            mat->cols = circles->total;
        else
            mat->rows = circles->total;
    }
    else
        result = circles;

    return result;
}



-----HOUGH RETTE
void cv::HoughLines( InputArray _image, OutputArray _lines,
                    double rho, double theta, int threshold,
                    double srn, double stn, double min_theta, double max_theta )
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(srn == 0 && stn == 0 && _image.isUMat() && _lines.isUMat(),
               ocl_HoughLines(_image, _lines, rho, theta, threshold, min_theta, max_theta));

    Mat image = _image.getMat();
    std::vector<Vec2f> lines;

    if( srn == 0 && stn == 0 )
        HoughLinesStandard(image, (float)rho, (float)theta, threshold, lines, INT_MAX, min_theta, max_theta );
    else
        HoughLinesSDiv(image, (float)rho, (float)theta, threshold, cvRound(srn), cvRound(stn), lines, INT_MAX, min_theta, max_theta);

    Mat(lines).copyTo(_lines);
}


void cv::HoughLinesP(InputArray _image, OutputArray _lines,
                     double rho, double theta, int threshold,
                     double minLineLength, double maxGap )
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(_image.isUMat() && _lines.isUMat(),
               ocl_HoughLinesP(_image, _lines, rho, theta, threshold, minLineLength, maxGap));

    Mat image = _image.getMat();
    std::vector<Vec4i> lines;
    HoughLinesProbabilistic(image, (float)rho, (float)theta, threshold, cvRound(minLineLength), cvRound(maxGap), lines, INT_MAX);
    Mat(lines).copyTo(_lines);
}



------LAPLACIAN
src = imread( argv[1] );
  if( !src.data )
    { return -1; }
  /// Remove noise by blurring with a Gaussian filter
  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );
  /// Create window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );
  /// Apply Laplace function
  Mat abs_dst;
  Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( dst, abs_dst );
  /// Show what you got
  imshow( window_name, abs_dst );
  waitKey(0);
  return 0;


------EQUALIZEHIST
  Mat src, dst;
  char* source_window = "Source image";
  char* equalized_window = "Equalized Image";
  /// Load image
  src = imread( argv[1], 1 );
  if( !src.data )
    { cout<<"Usage: ./Histogram_Demo <path_to_image>"<<endl;
      return -1;}
  /// Convert to grayscale
  cvtColor( src, src, CV_BGR2GRAY );
  /// Apply Histogram Equalization
  equalizeHist( src, dst ); alternativa....   threshold( src_gray, dst, 0, 255,3 );
  /// Display results
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  namedWindow( equalized_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );
  imshow( equalized_window, dst );
  /// Wait until user exits the program
  waitKey(0);
  return 0;



------FINDING CONTOURS
src = imread( argv[1], 1 );
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
  /// Convert image to gray and blur it
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );
  Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
       {
         Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
         drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
       }
