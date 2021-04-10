#include <opencv2/opencv.hpp>
#include <chrono>

#ifdef __ANDROID__
#include <android/log.h>
#endif

using namespace cv;
using namespace std;

long long int get_now() {
    return chrono::duration_cast<std::chrono::milliseconds>(
            chrono::system_clock::now().time_since_epoch()
    ).count();
}

void platform_log(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
#ifdef __ANDROID__
    __android_log_vprint(ANDROID_LOG_VERBOSE, "ndk", fmt, args);
#else
    vprintf(fmt, args);
#endif
    va_end(args);
}

// Avoiding name mangling
extern "C" {
    // Attributes to prevent 'unused' function from being removed and to make it visible
    __attribute__((visibility("default"))) __attribute__((used))
    const char* version() {
        return CV_VERSION;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    bool compareContourAreas ( vector<Point> contour1, vector<Point> contour2 ) {
        double i = fabs( contourArea(Mat(contour1)) );
        double j = fabs( contourArea(Mat(contour2)) );
        return ( i > j );
    }

    __attribute__((visibility("default"))) __attribute__((used))
    int convertToInt (float num){
        return static_cast<int>(num);
    }

    __attribute__((visibility("default"))) __attribute__((used))
    void thresholdSegmentation(Mat input, Mat input_gray, char* outputImagePath){
        Mat grad, threshed, blurred, kernel, closed;
        vector<vector<Point>> contours;
        vector<Point> largestContour;
        vector<Vec4i> hierarchy;
        RotatedRect rect;
        vector<int> points;
        //vector<int> intPoints;
        int ddepth = CV_32F;

        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        Sobel(input_gray, grad_x, ddepth, 1, 0, -1);
        Sobel(input_gray, grad_y, ddepth, 0, 1, -1);
        
        // Convert output to a CV_8U image
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);

        //Gradient
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

        // blurring
        blur(grad, blurred, Size(9, 9));

        // Binary thresholding
        threshold(blurred, threshed, 230, 255, THRESH_BINARY);

        // get kernel for morphological operations
        kernel = getStructuringElement(MORPH_RECT, Size(21, 7));

        // closing morphological op
        morphologyEx( threshed, closed, MORPH_CLOSE, kernel );

        // find contours
        findContours(closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        //sort contours according to area
        sort(contours.begin(), contours.end(), compareContourAreas);

        // get largest contour
        largestContour = contours[0];

        rect = minAreaRect(largestContour);

        boxPoints(rect, points);

        // transform box point values to integers
        //intPoints(points.begin(), points.end());

        drawContours(input, points, -1, Scalar(0, 255, 0), 3);

        imwrite(outputImagePath, input);
    }

    __attribute__((visibility("default"))) __attribute__((used))
    class ShapeDetector {
        public:
            string detectShape(std::vector<Point> contour){
                string shape = "unidentified";
                vector<Point> approx;
                approxPolyDP(contour, approx, 0.04 * arcLength(contour, true), true);
                platform_log("Number of vertices: %d\n", approx.size());

                if(approx.size() == 3){
                    shape = "triangle";
                } else if(approx.size() == 4){
                    shape = "square or rectangle";
                } else if(approx.size() == 5){
                    shape = "pentagon";
                } else {
                    shape = "circle";
                }

                return shape;
            }
    };

    __attribute__((visibility("default"))) __attribute__((used))
    void process_image(char* inputImagePath, char* outputImagePath) {
        long long start = get_now();

        Mat input = imread(inputImagePath);

        Mat input_gray = imread(inputImagePath, IMREAD_GRAYSCALE);
        Mat threshed, withContours, blurred;

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        // image preprocessing
        // GaussianBlur(input, blurred, Size(5, 5), 0);
        // threshold(blurred, threshed, 60, 255, THRESH_BINARY);
        // findContours(threshed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // drawContours(threshed, contours, -1, Scalar(0, 255, 0), 4);

        // ShapeDetector sd;
        // for(int i = 0; i < contours.size(); i++){
        //     string shape = sd.detectShape(contours[i]);
        //     platform_log("Shape: %s\n", shape.c_str());
        // }

        // imwrite(outputImagePath, grad);
        thresholdSegmentation(input, input_gray, outputImagePath);
        
        int evalInMillis = static_cast<int>(get_now() - start);
        platform_log("Processing done in %dms\n", evalInMillis);
    }
}
